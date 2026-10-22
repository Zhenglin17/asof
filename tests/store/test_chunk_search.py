"""Guards against look-ahead leakage in full-text retrieval: future or replaced chunks must not
appear, must not take a slot of `limit`, and must not move any score through the BM25 corpus
statistics (N, document frequency, average length are taken over chunks visible at as_of only);
free-form query text must never crash the search.
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import Session

from asof.store.db import init_db, make_engine
from asof.store.models import Chunk, Document, Entity, Source
from asof.store.queries import ChunkHit, base_form, visible_chunk_search
from tests.store.factories import T0, new_chunk, new_document, new_entity, new_source, pk

DAY = timedelta(days=1)
MICROSECOND = timedelta(microseconds=1)


def filed(
    session: Session,
    source: Source,
    entity: Entity,
    text: str,
    *,
    available_at: datetime = T0 - DAY,
    doc_type: str = "10-Q",
    supersedes: Document | None = None,
) -> tuple[Document, Chunk]:
    doc = new_document(
        session,
        source_id=pk(source),
        entity_id=pk(entity),
        available_at=available_at,
        doc_type=doc_type,
        supersedes_id=pk(supersedes) if supersedes is not None else None,
    )
    return doc, new_chunk(session, document_id=pk(doc), text=text)


@pytest.fixture
def microsoft(session: Session) -> Entity:
    return new_entity(session, 789019, "MSFT", name="Microsoft Corporation")


def fillers(session: Session, source: Source, entity: Entity, n: int = 4) -> None:
    """Unrelated chunks, so every query term is rare and BM25 term weights are not clamped."""
    for i in range(n):
        filed(session, source, entity, f"Unrelated filler paragraph number {i} about weather.")


def ids(hits: list[ChunkHit]) -> list[int]:
    return [h.chunk_id for h in hits]


def test_hit_carries_document_context(session: Session, source: Source, apple: Entity) -> None:
    doc, chunk = filed(session, source, apple, "Gross margin expanded on services mix.")
    filed(session, source, apple, "The board approved a new buyback.")

    hits = visible_chunk_search(session, T0, "gross margin")

    assert len(hits) == 1
    hit = hits[0]
    assert hit.chunk_id == chunk.id
    assert hit.document_id == doc.id
    assert hit.entity_id == apple.id
    assert hit.doc_type == "10-Q"
    assert hit.available_at == T0 - DAY
    assert hit.available_at.utcoffset() == timedelta(0)
    assert hit.section_path == chunk.section_path
    assert hit.text == "Gross margin expanded on services mix."
    assert isinstance(hit.score, float)


def test_hit_is_immutable(session: Session, source: Source, apple: Entity) -> None:
    filed(session, source, apple, "Gross margin expanded.")
    (hit,) = visible_chunk_search(session, T0, "margin")
    with pytest.raises(AttributeError):
        hit.score = 0.0  # pyright: ignore[reportAttributeAccessIssue]


def test_future_best_match_neither_appears_nor_takes_the_limit_slot(
    session: Session, source: Source, apple: Entity
) -> None:
    fillers(session, source, apple)
    _, weak = filed(session, source, apple, "Gross margin was stable this quarter.")
    filed(
        session,
        source,
        apple,
        "Gross margin expanded on data center demand.",
        available_at=T0 + DAY,
    )
    query = "gross margin expanded data center demand"

    assert ids(visible_chunk_search(session, T0, query, limit=1)) == [weak.id]
    # Once it is public it is the better match.
    later = visible_chunk_search(session, T0 + DAY, query, limit=1)
    assert ids(later) != [weak.id]


def test_only_future_match_returns_nothing(session: Session, source: Source, apple: Entity) -> None:
    _, future = filed(session, source, apple, "Vision Pro shipments began.", available_at=T0 + DAY)
    assert visible_chunk_search(session, T0, "vision pro") == []
    assert ids(visible_chunk_search(session, T0 + DAY, "vision pro")) == [future.id]


def test_available_exactly_at_as_of_is_visible_one_microsecond_earlier_is_not(
    session: Session, source: Source, apple: Entity
) -> None:
    _, chunk = filed(session, source, apple, "Wearables revenue declined.", available_at=T0)
    assert ids(visible_chunk_search(session, T0, "wearables")) == [chunk.id]
    assert visible_chunk_search(session, T0 - MICROSECOND, "wearables") == []


def test_as_of_in_another_timezone_compares_as_the_same_instant(
    session: Session, source: Source, apple: Entity
) -> None:
    from zoneinfo import ZoneInfo

    _, chunk = filed(session, source, apple, "Wearables revenue declined.", available_at=T0)
    new_york = T0.astimezone(ZoneInfo("America/New_York"))
    assert ids(visible_chunk_search(session, new_york, "wearables")) == [chunk.id]
    assert visible_chunk_search(session, new_york - MICROSECOND, "wearables") == []


def test_revision_hides_the_old_version_only_once_it_is_public(
    session: Session, source: Source, apple: Entity
) -> None:
    original, old = filed(
        session,
        source,
        apple,
        "Net sales were restated lower for the quarter.",
        available_at=T0 - 10 * DAY,
    )
    _, new = filed(
        session,
        source,
        apple,
        "Net sales were confirmed for the quarter.",
        available_at=T0 + 10 * DAY,
        supersedes=original,
    )

    assert ids(visible_chunk_search(session, T0, "net sales quarter")) == [old.id]
    assert ids(visible_chunk_search(session, T0 + 10 * DAY, "net sales quarter")) == [new.id]
    # Words only the old version had are gone too, not just outranked.
    assert visible_chunk_search(session, T0 + 10 * DAY, "restated") == []


def test_entity_filter_applies_before_the_limit(
    session: Session, source: Source, apple: Entity, microsoft: Entity
) -> None:
    _, mine = filed(session, source, apple, "Cloud revenue was flat.")
    filed(session, source, microsoft, "Cloud revenue grew on Azure cloud demand for cloud.")

    assert ids(
        visible_chunk_search(session, T0, "cloud revenue", entity_id=pk(apple), limit=1)
    ) == [mine.id]
    assert len(visible_chunk_search(session, T0, "cloud revenue")) == 2


def test_unknown_entity_returns_nothing(session: Session, source: Source, apple: Entity) -> None:
    filed(session, source, apple, "Cloud revenue was flat.")
    assert visible_chunk_search(session, T0, "cloud", entity_id=999_999) == []


@pytest.mark.parametrize(
    ("doc_type", "expected"),
    [
        ("10-K", "10-K"),
        ("10-K/A", "10-K"),
        ("10-KT", "10-K"),
        ("10-KT/A", "10-K"),
        ("10-Q", "10-Q"),
        ("10-Q/A", "10-Q"),
        ("10-QT", "10-Q"),
        ("8-K", "8-K"),
        ("8-K/A", "8-K"),
        ("20-F/A", "20-F"),
        ("40-F", "40-F"),
        ("6-K/A", "6-K"),
    ],
)
def test_base_form(doc_type: str, expected: str) -> None:
    assert base_form(doc_type) == expected


def test_base_forms_filter_counts_amendments_and_transition_reports(
    session: Session, source: Source, apple: Entity
) -> None:
    by_type = {
        doc_type: filed(
            session, source, apple, f"Risk factors discussed in {doc_type}.", doc_type=doc_type
        )[1]
        for doc_type in ["10-K", "10-K/A", "10-KT", "10-Q", "10-Q/A", "8-K", "8-K/A", "20-F/A"]
    }

    def found(base_forms: list[str]) -> set[int]:
        hits = visible_chunk_search(session, T0, "risk factors", base_forms=base_forms, limit=50)
        return set(ids(hits))

    def chunk_ids(*doc_types: str) -> set[int]:
        return {pk(by_type[t]) for t in doc_types}

    assert found(["10-K"]) == chunk_ids("10-K", "10-K/A", "10-KT")
    assert found(["8-K"]) == chunk_ids("8-K", "8-K/A")
    assert found(["10-Q", "20-F"]) == chunk_ids("10-Q", "10-Q/A", "20-F/A")
    assert found(["40-F"]) == set()
    assert len(visible_chunk_search(session, T0, "risk factors", limit=50)) == 8


def test_base_forms_filter_applies_before_the_limit(
    session: Session, source: Source, apple: Entity
) -> None:
    _, annual = filed(session, source, apple, "Supply chain risk.", doc_type="10-K")
    filed(session, source, apple, "Supply chain risk, supply chain risk.", doc_type="8-K")
    hits = visible_chunk_search(session, T0, "supply chain risk", base_forms=["10-K"], limit=1)
    assert ids(hits) == [annual.id]


@pytest.mark.parametrize(
    "query",
    [
        'gross "margin',
        "gross-margin",
        "margin: up",
        "text:margin",
        "gross AND margin",
        "gross OR NOT margin",
        "NEAR(gross margin)",
        "(margin",
        "margin*",
        "^margin",
        "+margin",
        "-margin",
        "margin's",
        "{gross margin}",
        "gross\tmargin\n",
    ],
)
def test_query_syntax_characters_are_plain_words(
    session: Session, source: Source, apple: Entity, query: str
) -> None:
    _, chunk = filed(session, source, apple, "Gross margin expanded.")
    assert ids(visible_chunk_search(session, T0, query)) == [chunk.id]


@pytest.mark.parametrize("query", ["", "   ", '"" ** () :- ^', "\n\t"])
def test_query_without_words_returns_nothing(
    session: Session, source: Source, apple: Entity, query: str
) -> None:
    filed(session, source, apple, "Gross margin expanded.")
    assert visible_chunk_search(session, T0, query) == []


def test_more_matching_terms_rank_higher(session: Session, source: Source, apple: Entity) -> None:
    # Same length, so only the number of matched query terms differs.
    fillers(session, source, apple)
    _, fewer = filed(session, source, apple, "Revenue was reported for the quarter today.")
    _, more = filed(session, source, apple, "Revenue from data center showed growth today.")

    hits = visible_chunk_search(session, T0, "revenue data center growth")

    assert ids(hits) == [more.id, fewer.id]
    assert hits[0].score > hits[1].score


def test_equal_scores_are_ordered_by_chunk_id(
    session: Session, source: Source, apple: Entity
) -> None:
    chunks = [
        filed(session, source, apple, "Dividend raised by four percent.")[1] for _ in range(3)
    ]
    hits = visible_chunk_search(session, T0, "dividend raised")
    assert ids(hits) == sorted(pk(c) for c in chunks)
    assert len({h.score for h in hits}) == 1
    assert (
        ids(visible_chunk_search(session, T0, "dividend raised", limit=2))
        == sorted(pk(c) for c in chunks)[:2]
    )


def test_limit_caps_the_result(session: Session, source: Source, apple: Entity) -> None:
    for n in range(5):
        filed(session, source, apple, f"Buyback number {n} completed.")
    hits = visible_chunk_search(session, T0, "buyback", limit=3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_naive_as_of_is_rejected(session: Session) -> None:
    with pytest.raises(ValueError, match="timezone"):
        visible_chunk_search(session, datetime(2024, 7, 15, 13, 30), "margin")


# --- BM25 statistics come from the as_of corpus only ---
# score = sum over distinct query terms of idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len / avg))
# idf = ln(1 + (N - n + 0.5) / (n + 0.5)); len = char_end - char_start; N, n and avg over the
# chunks visible at as_of (every entity, replaced versions excluded).
K1 = 1.2
B = 0.75


def scored(hits: list[ChunkHit]) -> list[tuple[int, float]]:
    return [(h.chunk_id, h.score) for h in hits]


def test_future_documents_leave_scores_exactly_unchanged(
    session: Session, source: Source, apple: Entity, microsoft: Entity
) -> None:
    filed(session, source, apple, "Copper output rose sharply in the quarter.")
    filed(session, source, apple, "Copper prices fell.")
    filed(session, source, microsoft, "Output of zircon was flat.")
    fillers(session, source, apple)
    query = "copper zircon output"
    before = scored(visible_chunk_search(session, T0, query))
    assert len(before) == 3

    # Future rows that match the query (raise n) and that do not (raise N, change avg length).
    for i in range(6):
        filed(session, source, apple, f"Copper zircon output {i}.", available_at=T0 + DAY)
        filed(
            session,
            source,
            microsoft,
            "An unrelated and much longer paragraph about the weather " * (i + 1),
            available_at=T0 + MICROSECOND,
        )

    assert scored(visible_chunk_search(session, T0, query)) == before


def test_term_that_becomes_common_only_later_keeps_its_past_weight(
    session: Session, source: Source, apple: Entity
) -> None:
    # At T0 "zircon" is in one visible chunk and "copper" in four, so the zircon chunk wins.
    # Twenty future zircon chunks would flip that if statistics counted the future.
    _, zircon = filed(session, source, apple, "Zircon output rose.")  # 19 characters
    _, copper = filed(session, source, apple, "Copper output rose.")  # 19 characters
    filed(session, source, apple, "Copper prices fell sharply.")  # 27
    filed(session, source, apple, "Copper demand was steady.")  # 25
    filed(session, source, apple, "Copper mines restarted.")  # 23
    for i in range(20):
        filed(session, source, apple, f"Zircon zircon note {i}.", available_at=T0 + DAY)

    hits = visible_chunk_search(session, T0, "zircon copper")

    assert hits[0].chunk_id == zircon.id
    by_id = {h.chunk_id: h.score for h in hits}
    assert by_id[pk(zircon)] > by_id[pk(copper)]

    # Hand-computed: N = 5, avg = (19 + 19 + 27 + 25 + 23) / 5 = 22.6, tf = 1, len = 19.
    norm = 1 * (K1 + 1) / (1 + K1 * (1 - B + B * 19 / 22.6))
    zircon_expected = math.log(1 + (5 - 1 + 0.5) / (1 + 0.5)) * norm  # n = 1
    copper_expected = math.log(1 + (5 - 4 + 0.5) / (4 + 0.5)) * norm  # n = 4
    assert zircon_expected == pytest.approx(1.482929338099849, rel=1e-12)
    # approx only to allow a different floating-point operation order, not other statistics.
    assert by_id[pk(zircon)] == pytest.approx(zircon_expected, rel=1e-12)
    assert by_id[pk(copper)] == pytest.approx(copper_expected, rel=1e-12)


def test_replaced_version_does_not_count_once_the_revision_is_public(
    session: Session, source: Source, apple: Entity, tmp_path: Path
) -> None:
    def common(s: Session, src: Source, entity: Entity) -> None:
        filed(s, src, entity, "Copper prices fell sharply.", available_at=T0 - 20 * DAY)
        filed(s, src, entity, "Zircon demand was steady.", available_at=T0 - 20 * DAY)

    # Store 1: an original with "copper" and another length, replaced before T0.
    common(session, source, apple)
    original, _ = filed(
        session,
        source,
        apple,
        "Copper output rose sharply in the quarter, copper copper.",
        available_at=T0 - 10 * DAY,
    )
    filed(session, source, apple, "Copper output rose.", available_at=T0 - DAY, supersedes=original)

    # Store 2: the same rows except that the original never existed.
    other = make_engine(tmp_path / "other.db")
    init_db(other)
    try:
        with Session(other) as s2:
            src2 = new_source(s2)
            apple2 = new_entity(s2)
            common(s2, src2, apple2)
            filed(s2, src2, apple2, "Copper output rose.", available_at=T0 - DAY)
            expected = [(h.text, h.score) for h in visible_chunk_search(s2, T0, "copper output")]
    finally:
        other.dispose()

    got = [(h.text, h.score) for h in visible_chunk_search(session, T0, "copper output")]
    assert got == expected
    # Before the revision the original counts, so the same chunk scores differently.
    earlier = {h.text: h.score for h in visible_chunk_search(session, T0 - 2 * DAY, "copper")}
    assert earlier["Copper prices fell sharply."] != dict(got)["Copper prices fell sharply."]


# --- top-k pruning returns the same hits as the exhaustive ranking ---
# Reference argument: with limit >= number of candidates the k-th score threshold is 0, so no
# term is ever skipped and the result is the full ranking of every visible matching chunk. Any
# pruned top-k must therefore equal that list cut at k, hit for hit (scores, ties, order).
EXHAUSTIVE = 10_000
COMMON_FILLERS = [
    "metal",
    "market",
    "price",
    "supply",
    "demand",
    "copper",
    "steel",
    "freight",
    "harbor",
    "rail",
    "storage",
    "smelter",
    "furnace",
    "quota",
    "tariff",
    "barge",
]


def pruning_corpus(
    session: Session, source: Source, apple: Entity, microsoft: Entity
) -> dict[str, Chunk]:
    """About 40 visible chunks: "the", "of", "and" are in nearly all, a few words are rare."""
    named: dict[str, Chunk] = {}

    def put(text: str, available_at: datetime = T0 - DAY) -> Chunk:
        return filed(session, source, apple, text, available_at=available_at)[1]

    # Common-only chunks of varied length; each filler appears in two of them, and the first
    # pair is identical so equal scores occur among common-word hits too.
    for i in range(32):
        word = COMMON_FILLERS[i % len(COMMON_FILLERS)]
        tail = " and the " + " ".join(COMMON_FILLERS[: i % 5]) if i >= 2 else ""
        filed(session, source, apple, f"The {word} of the year{tail}.")
    # Only common words, many times: high tf, low idf.
    named["high_tf"] = filed(
        session, source, apple, "The the the of of of and and and the of and the of and."
    )[1]
    # Rare words, a few chunks each; three identical hafnium chunks tie at any cut.
    named["zircon"] = put("The zircon output of the mine and the plant.")
    for n in range(3):
        named[f"hafnium{n}"] = put("The hafnium yield of the year and.")
    named["niobium"] = put("Niobium of the quarter.")
    named["both"] = put("The zircon and hafnium of the mine.")
    # Another company's strong matches: only the entity filter keeps them out.
    filed(session, source, microsoft, "Zircon zircon zircon hafnium of the and.")
    filed(session, source, microsoft, "Niobium niobium of the.")
    # Future strong matches: never visible at T0.
    for i in range(5):
        put(f"Zircon hafnium niobium {i} the of and.", available_at=T0 + DAY)
    return named


@pytest.mark.parametrize("by_entity", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "zircon the of",
        "hafnium and the",
        "zircon hafnium niobium of the and",
        "the of and",
        "niobium the of and",
    ],
)
def test_pruned_top_k_equals_the_exhaustive_ranking_cut_at_k(
    session: Session,
    source: Source,
    apple: Entity,
    microsoft: Entity,
    query: str,
    by_entity: bool,
) -> None:
    pruning_corpus(session, source, apple, microsoft)
    entity_id = pk(apple) if by_entity else None
    reference = visible_chunk_search(session, T0, query, entity_id=entity_id, limit=EXHAUSTIVE)
    assert len(reference) > 10
    assert all(h.available_at <= T0 for h in reference)
    if by_entity:
        assert {h.entity_id for h in reference} == {pk(apple)}

    for k in (1, 2, 3, 5, 10):
        got = visible_chunk_search(session, T0, query, entity_id=entity_id, limit=k)
        assert got == reference[:k], f"k={k}"


def test_tied_rare_chunks_are_cut_by_chunk_id(
    session: Session, source: Source, apple: Entity, microsoft: Entity
) -> None:
    named = pruning_corpus(session, source, apple, microsoft)
    hafnium = sorted(pk(named[f"hafnium{n}"]) for n in range(3))
    reference = visible_chunk_search(
        session, T0, "hafnium yield", entity_id=pk(apple), limit=EXHAUSTIVE
    )
    tied = [h for h in reference if h.chunk_id in hafnium]
    assert len({h.score for h in tied}) == 1
    for k in (1, 2):
        got = visible_chunk_search(session, T0, "hafnium yield", entity_id=pk(apple), limit=k)
        assert ids(got) == hafnium[:k]
        assert got == reference[:k]


def test_common_word_chunk_fills_slots_the_rare_chunks_cannot(
    session: Session, source: Source, apple: Entity, microsoft: Entity
) -> None:
    # Only one visible apple chunk has "niobium"; with k=3 the other two slots go to the best
    # common-word chunks, and the high-tf chunk is the best of those. Pruning must not drop it
    # just because each of its terms has a tiny idf.
    named = pruning_corpus(session, source, apple, microsoft)
    query = "niobium the of and"
    got = visible_chunk_search(session, T0, query, entity_id=pk(apple), limit=3)
    reference = visible_chunk_search(session, T0, query, entity_id=pk(apple), limit=EXHAUSTIVE)

    assert got == reference[:3]
    assert got[0].chunk_id == named["niobium"].id
    assert named["high_tf"].id in ids(got)


# --- characters where FTS5's tokenizer and a Python regex disagree ---
# Measured on the real store 2026-10-22 (chunks containing each): ligatures 170 (chunk 8730
# "ﬃcer"), "™"/"®" glued to a word 5,741 (chunk 204 "Bandwidth™"), private-use glyphs 533 (chunk
# 8388, U+F06F), superscripts 548 (chunk 29190 "Count²"). Fullwidth letters: assumption, 0 found;
# kept as a defensive case. tf, n and the
# candidate set all come from the index itself, so whatever FTS5 makes of these, a chunk it
# returns must score above zero, and pruning must agree with the exhaustive ranking. Which query
# spelling recalls a ligature word is not specified here.
ODD_TEXTS = [
    "Terms not otherwise deﬁned have the meaning of the plan.",
    "EYLEA™ sales of the quarter rose and the outlook held.",
    "At fair value of the notes and the swaps.",
    "Area of the site is 500 m² and the lease runs on.",
    "ＡＢＣ revenue of the segment and the region grew.",
    "Defined terms of the plan and the notes are listed.",
    "Fair value of the notes and the swaps was measured.",
]
ODD_QUERIES = [
    "eylea",
    "eylea sales the",
    "deﬁned terms of the",
    "defined terms",
    "At fair value",
    "fair value of the notes",
    "m² area of the",
    "ＡＢＣ revenue the",
    "abc revenue",
]


def odd_corpus(session: Session, source: Source, apple: Entity) -> dict[str, Chunk]:
    named = {text: filed(session, source, apple, text)[1] for text in ODD_TEXTS}
    for i in range(20):
        word = COMMON_FILLERS[i % len(COMMON_FILLERS)]
        filed(session, source, apple, f"The {word} of the plan and the year {i}.")
    filed(session, source, apple, "EYLEA™ deﬁned ＡＢＣ m² notes.", available_at=T0 + DAY)
    return named


@pytest.mark.parametrize("query", ODD_QUERIES)
def test_odd_characters_never_give_a_zero_score_and_pruning_agrees(
    session: Session, source: Source, apple: Entity, query: str
) -> None:
    odd_corpus(session, source, apple)
    reference = visible_chunk_search(session, T0, query, limit=EXHAUSTIVE)
    assert all(h.score > 0 for h in reference), [(h.text, h.score) for h in reference]
    assert all(h.available_at <= T0 for h in reference)
    for k in (1, 2, 3, 5):
        assert visible_chunk_search(session, T0, query, limit=k) == reference[:k], f"k={k}"


def test_brand_with_trademark_sign_is_found_by_its_plain_name(
    session: Session, source: Source, apple: Entity
) -> None:
    named = odd_corpus(session, source, apple)
    eylea = named["EYLEA™ sales of the quarter rose and the outlook held."]
    hits = {h.chunk_id: h for h in visible_chunk_search(session, T0, "eylea", limit=EXHAUSTIVE)}
    assert pk(eylea) in hits
    assert hits[pk(eylea)].score > 0
