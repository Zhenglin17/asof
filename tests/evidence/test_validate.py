"""Guards against conclusions that their evidence does not support: a cited chunk that was not
public at as_of (or was already replaced, or never existed: all read the same), belongs to another
company, does not contain the quote on token boundaries, has a quote outside the length limits,
or whose quote lacks a number (value, unit and sign) the claim states.
"""

import re
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session

from asof.evidence import validate as v
from asof.store.models import Chunk, Document, Entity, Source
from tests.store.factories import T0, new_chunk, new_document, new_entity, pk

DAY = timedelta(days=1)
MICROSECOND = timedelta(microseconds=1)

# Mimics an EDGAR results chunk. Line break inside a sentence: measured, e.g. chunk 463551 of
# NVDA 10-Q 0001045810-26-000075 reads "jurisdiction of ... |\nincorporation or organization".
# Non-breaking space: assumption only; on 2026-10-22 none of NVDA's real chunks held one. It is a
# defensive case; tests/evidence/test_golden_evidence.py checks a real quote against a real chunk.
RESULTS = (
    "Total net sales were $85,777 million in the third quarter of fiscal 2024,\n"
    "up 5% from a year ago. Gross margin was 46.3 percent of net sales."
)
NUMBERS = (
    "During the quarter the company returned approximately $26.0 billion to shareholders. "
    "It repurchased 1,200 thousand shares at an average price of $3.50 per share. "
    "Paid subscriptions reached 12 million accounts. "
    "Operating expenses were 26,000 million dollars in total."
)
# Exactly 20 characters, no digits.
QUOTE_20 = "Total net sales were"


def cite(
    chunk: Chunk | None,
    quote: str | None,
    claim: str = "Net sales were reported.",
    *,
    kind: str = "chunk",
    evidence_ref: str | None = None,
) -> v.ClaimCitation:
    return v.ClaimCitation(
        claim=claim,
        evidence_kind=kind,
        chunk_id=pk(chunk) if chunk is not None else None,
        quote=quote,
        evidence_ref=evidence_ref,
    )


def check(
    session: Session, as_of: datetime, entity: Entity | None, *citations: v.ClaimCitation
) -> v.ValidationResult:
    draft = v.DecisionDraft(
        as_of=as_of, entity_id=pk(entity) if entity is not None else None, citations=list(citations)
    )
    return v.validate(session, draft)


def reasons(result: v.ValidationResult) -> list[tuple[int, v.FailureReason]]:
    return [(f.index, f.reason) for f in result.failures]


def filed(
    session: Session,
    source: Source,
    entity: Entity,
    text: str,
    *,
    available_at: datetime = T0 - DAY,
    supersedes: Document | None = None,
) -> tuple[Document, Chunk]:
    doc = new_document(
        session,
        source_id=pk(source),
        entity_id=pk(entity),
        available_at=available_at,
        supersedes_id=pk(supersedes) if supersedes is not None else None,
    )
    return doc, new_chunk(session, document_id=pk(doc), text=text)


@pytest.fixture
def results(session: Session, source: Source, apple: Entity) -> Chunk:
    return filed(session, source, apple, RESULTS)[1]


@pytest.fixture
def numbers(session: Session, source: Source, apple: Entity) -> Chunk:
    return filed(session, source, apple, NUMBERS)[1]


# --- normal path ---


def test_verbatim_quote_with_matching_numbers_passes(
    session: Session, apple: Entity, results: Chunk
) -> None:
    quote = "Total net sales were $85,777 million in the third quarter of fiscal 2024, up 5%"
    claim = "Net sales were $85,777 million in fiscal 2024, up 5% year over year."
    result = check(session, T0, apple, cite(results, quote, claim))
    assert result.ok
    assert result.failures == []


def test_quote_copied_with_the_original_line_break_passes(
    session: Session, apple: Entity, results: Chunk
) -> None:
    quote = "in the third quarter of fiscal 2024,\nup 5% from a year ago."
    assert check(session, T0, apple, cite(results, quote, "Sales rose 5% in fiscal 2024.")).ok


def test_several_valid_citations_pass(
    session: Session, apple: Entity, results: Chunk, numbers: Chunk
) -> None:
    result = check(
        session,
        T0,
        apple,
        # "46.3 percent" is the token 46.3%, so the claim must say % (a bare 46.3 would fail).
        cite(results, "Gross margin was 46.3 percent of net sales.", "Gross margin was 46.3%."),
        cite(
            numbers, "returned approximately $26.0 billion to shareholders", "Returned $26 billion."
        ),
    )
    assert result.ok


# --- draft-level rules ---


def test_draft_without_citations_is_rejected(session: Session, apple: Entity) -> None:
    result = check(session, T0, apple)
    assert not result.ok
    assert [f.reason for f in result.failures] == [v.FailureReason.NO_CITATIONS]


def test_naive_as_of_is_rejected(session: Session, apple: Entity, results: Chunk) -> None:
    with pytest.raises(ValueError, match="timezone"):
        check(session, datetime(2024, 7, 15, 13, 30), apple, cite(results, QUOTE_20))


def test_every_failing_citation_is_reported_and_one_failure_rejects_all(
    session: Session, apple: Entity, results: Chunk
) -> None:
    good = cite(results, QUOTE_20)
    wrong_number = cite(results, QUOTE_20 + " $85,777 million", "Net sales were $90,000 million.")
    missing = v.ClaimCitation(
        claim="Net sales were reported.", evidence_kind="chunk", chunk_id=999_999, quote=QUOTE_20
    )
    result = check(session, T0, apple, good, wrong_number, missing)
    assert not result.ok
    assert reasons(result) == [
        (1, v.FailureReason.NUMBER_NOT_IN_QUOTE),
        (2, v.FailureReason.NOT_VISIBLE),
    ]
    assert all(f.detail for f in result.failures)


# --- chunk existence and kind ---


def test_unknown_chunk_reads_as_not_visible(session: Session, apple: Entity) -> None:
    citation = v.ClaimCitation(
        claim="Net sales were reported.", evidence_kind="chunk", chunk_id=999_999, quote=QUOTE_20
    )
    assert reasons(check(session, T0, apple, citation)) == [(0, v.FailureReason.NOT_VISIBLE)]


def test_chunk_citation_without_chunk_id_reads_as_not_visible(
    session: Session, apple: Entity
) -> None:
    citation = v.ClaimCitation(
        claim="Net sales were reported.", evidence_kind="chunk", chunk_id=None, quote=QUOTE_20
    )
    assert reasons(check(session, T0, apple, citation)) == [(0, v.FailureReason.NOT_VISIBLE)]


def test_future_and_unknown_chunks_are_indistinguishable(
    session: Session, source: Source, apple: Entity
) -> None:
    # A caller probing ids must not learn that a chunk exists before it is public, or that it
    # was replaced: reason and detail match an id that never existed, up to the id itself.
    original, replaced = filed(session, source, apple, RESULTS, available_at=T0 - 10 * DAY)
    filed(session, source, apple, RESULTS, available_at=T0 - DAY, supersedes=original)
    _, future = filed(session, source, apple, RESULTS, available_at=T0 + DAY)
    unknown_id = 999_999

    def failure(chunk_id: int) -> v.CitationFailure:
        citation = v.ClaimCitation(
            claim="Net sales were reported.",
            evidence_kind="chunk",
            chunk_id=chunk_id,
            quote=QUOTE_20,
        )
        (only,) = check(session, T0, apple, citation).failures
        return only

    def masked(f: v.CitationFailure, chunk_id: int) -> tuple[v.FailureReason, str]:
        # Digit-bounded, so an id of 3 does not also mask the 3 in "13:30".
        return f.reason, re.sub(rf"(?<!\d){chunk_id}(?!\d)", "<id>", f.detail)

    expected = masked(failure(unknown_id), unknown_id)
    assert expected[0] is v.FailureReason.NOT_VISIBLE
    assert masked(failure(pk(future)), pk(future)) == expected
    assert masked(failure(pk(replaced)), pk(replaced)) == expected


@pytest.mark.parametrize("kind", ["bar", "fact", "option_snapshot", "macro_obs", "tweet"])
def test_non_chunk_evidence_is_unsupported_in_v1(
    session: Session, apple: Entity, kind: str
) -> None:
    citation = cite(
        None, None, "The stock closed higher.", kind=kind, evidence_ref="AAPL/2024-07-12"
    )
    assert reasons(check(session, T0, apple, citation)) == [(0, v.FailureReason.UNSUPPORTED_KIND)]


# --- visibility (leakage) ---


def test_chunk_published_after_as_of_is_not_visible(
    session: Session, source: Source, apple: Entity
) -> None:
    _, future = filed(session, source, apple, RESULTS, available_at=T0 + DAY)
    citation = cite(future, QUOTE_20)
    assert reasons(check(session, T0, apple, citation)) == [(0, v.FailureReason.NOT_VISIBLE)]
    assert check(session, T0 + DAY, apple, citation).ok


def test_visibility_boundary_is_inclusive_at_the_microsecond(
    session: Session, source: Source, apple: Entity
) -> None:
    _, chunk = filed(session, source, apple, RESULTS, available_at=T0)
    citation = cite(chunk, QUOTE_20)
    assert check(session, T0, apple, citation).ok
    assert reasons(check(session, T0 - MICROSECOND, apple, citation)) == [
        (0, v.FailureReason.NOT_VISIBLE)
    ]


def test_chunk_of_a_replaced_version_is_valid_only_before_the_revision(
    session: Session, source: Source, apple: Entity
) -> None:
    original, old = filed(session, source, apple, RESULTS, available_at=T0 - 10 * DAY)
    filed(session, source, apple, RESULTS, available_at=T0 + 10 * DAY, supersedes=original)
    citation = cite(old, QUOTE_20)

    assert check(session, T0, apple, citation).ok
    assert reasons(check(session, T0 + 10 * DAY, apple, citation)) == [
        (0, v.FailureReason.NOT_VISIBLE)
    ]


# --- entity ---


def test_chunk_of_another_company_is_rejected(
    session: Session, apple: Entity, results: Chunk
) -> None:
    microsoft = new_entity(session, 789019, "MSFT", name="Microsoft Corporation")
    assert reasons(check(session, T0, microsoft, cite(results, QUOTE_20))) == [
        (0, v.FailureReason.ENTITY_MISMATCH)
    ]


def test_draft_without_a_target_entity_is_rejected(session: Session, results: Chunk) -> None:
    assert reasons(check(session, T0, None, cite(results, QUOTE_20))) == [
        (0, v.FailureReason.ENTITY_MISMATCH)
    ]


# --- quote ---


def test_missing_quote_is_rejected(session: Session, apple: Entity, results: Chunk) -> None:
    assert reasons(check(session, T0, apple, cite(results, None))) == [
        (0, v.FailureReason.MISSING_QUOTE)
    ]


def test_quote_of_exactly_the_minimum_length_passes(
    session: Session, apple: Entity, results: Chunk
) -> None:
    assert v.MIN_QUOTE_CHARS == 20
    assert len(QUOTE_20) == v.MIN_QUOTE_CHARS
    assert check(session, T0, apple, cite(results, QUOTE_20)).ok
    # Surrounding whitespace is not counted, and does not hurt.
    assert check(session, T0, apple, cite(results, "  " + QUOTE_20 + "\n")).ok


@pytest.mark.parametrize("quote", ["Total net sales wer", "   Total net sales wer      ", ""])
def test_quote_one_character_short_fails(
    session: Session, apple: Entity, results: Chunk, quote: str
) -> None:
    assert reasons(check(session, T0, apple, cite(results, quote))) == [
        (0, v.FailureReason.QUOTE_TOO_SHORT)
    ]


@pytest.mark.parametrize(
    "quote",
    [
        # One word changed.
        "Total net revenue were $85,777 million in the third quarter",
        # Case differs.
        "total net sales were $85,777 million in the third quarter",
        # Words that only appear in another chunk of the store.
        "During the quarter the company returned approximately",
    ],
)
def test_quote_not_in_the_cited_chunk_fails(
    session: Session, apple: Entity, results: Chunk, numbers: Chunk, quote: str
) -> None:
    assert reasons(check(session, T0, apple, cite(results, quote))) == [
        (0, v.FailureReason.QUOTE_NOT_IN_CHUNK)
    ]


# --- numbers ---


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        # "$26.0" and "26" are the same number: $ dropped, trailing zeros dropped.
        (
            "The company returned $26 billion to shareholders.",
            "the company returned approximately $26.0 billion to shareholders.",
        ),
        # "1,200" and "1200"; "$3.50" and "$3.5".
        (
            "It bought back 1200 thousand shares at $3.5 each.",
            "It repurchased 1,200 thousand shares at an average price of $3.50 per share.",
        ),
        # Same number twice in the claim, once in the quote. ("Twelve million" in words is
        # now UNCHECKED_NUMBER; see test_amount_in_words_is_unchecked.)
        (
            "12 million, i.e. 12 million accounts.",
            "Paid subscriptions reached 12 million accounts.",
        ),
    ],
)
def test_numbers_match_after_normalisation(
    session: Session, apple: Entity, numbers: Chunk, claim: str, quote: str
) -> None:
    assert check(session, T0, apple, cite(numbers, quote, claim)).ok


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        # A different amount.
        (
            "The company returned $25 billion to shareholders.",
            "the company returned approximately $26.0 billion to shareholders.",
        ),
        # A percentage is not the bare number: "12%" is not "12 million".
        ("Paid subscriptions grew 12%.", "Paid subscriptions reached 12 million accounts."),
        # No unit conversion: "$26 billion" is not recognised in "26,000 million".
        (
            "Operating expenses were $26 billion.",
            "Operating expenses were 26,000 million dollars in total.",
        ),
        # Whole numbers, not substrings: "26" does not match inside "26,000".
        (
            "Operating expenses were 26 units.",
            "Operating expenses were 26,000 million dollars in total.",
        ),
        # A year is a number too.
        (
            "In fiscal 2025 the company returned $26 billion.",
            "the company returned approximately $26.0 billion to shareholders.",
        ),
    ],
)
def test_claim_number_missing_from_the_quote_fails(
    session: Session, apple: Entity, numbers: Chunk, claim: str, quote: str
) -> None:
    assert reasons(check(session, T0, apple, cite(numbers, quote, claim))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


def test_number_elsewhere_in_the_chunk_does_not_count(
    session: Session, apple: Entity, results: Chunk
) -> None:
    # 2024 is in the chunk but outside the quote; the quote alone must carry every number.
    quote = "Total net sales were $85,777 million in the third"
    claim = "Net sales were $85,777 million in fiscal 2024."
    assert reasons(check(session, T0, apple, cite(results, quote, claim))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


def test_year_present_in_the_quote_passes(session: Session, apple: Entity, results: Chunk) -> None:
    quote = "in the third quarter of fiscal 2024"
    assert check(session, T0, apple, cite(results, quote, "Results for fiscal 2024.")).ok


def test_result_failures_carry_index_reason_and_detail(
    session: Session, source: Source, apple: Entity
) -> None:
    _, future = filed(
        session, source, apple, RESULTS, available_at=datetime(2030, 1, 1, tzinfo=UTC)
    )
    (failure,) = check(session, T0, apple, cite(future, QUOTE_20)).failures
    assert isinstance(failure, v.CitationFailure)
    assert failure.index == 0
    assert failure.reason is v.FailureReason.NOT_VISIBLE
    assert isinstance(failure.detail, str) and failure.detail


# --- quote length cap ---
# 99 four-letter words, then "abcde" ending at character 500, then "." and more text.
LONG = "abcd " * 99 + "abcde. Further words follow here."


def test_quote_of_exactly_the_maximum_length_passes(
    session: Session, source: Source, apple: Entity
) -> None:
    _, chunk = filed(session, source, apple, LONG)
    quote = LONG[:500]
    assert v.MAX_QUOTE_CHARS == 500
    assert len(quote) == 500 and quote.endswith("abcde")
    assert check(session, T0, apple, cite(chunk, quote)).ok
    # Whitespace runs collapse before counting: a 500-character quote copied with line breaks.
    assert check(session, T0, apple, cite(chunk, quote.replace(" ", " \n  "))).ok


def test_quote_one_character_too_long_fails(
    session: Session, source: Source, apple: Entity
) -> None:
    _, chunk = filed(session, source, apple, LONG)
    quote = LONG[:501]
    assert len(quote) == 501 and quote.endswith("abcde.")
    assert reasons(check(session, T0, apple, cite(chunk, quote))) == [
        (0, v.FailureReason.QUOTE_TOO_LONG)
    ]


# --- quote sits on token boundaries ---
BOUNDARY = (
    "Revenue was $15.2 million, up 3% from (a year ago) and $26.0 billion returned. "
    "xRevenue grew strongly today, and later Revenue grew strongly today. "
    "Costs were 1,234 units in total for the year."
)


@pytest.mark.parametrize(
    ("quote", "claim"),
    [
        # At the chunk start, ending before ", " (a comma not followed by a digit).
        ("Revenue was $15.2 million", "Revenue was $15.2 million."),
        # Right after "$".
        # The quote's first piece is read whole, "$15.2", so the claim keeps the "$".
        ("15.2 million, up 3% from (a year", "Revenue was $15.2 million, up 3%."),
        # Right after "(", ending before ". " and running to the end of a sentence.
        ("a year ago) and $26.0 billion returned", "Returned $26 billion."),
        # Ending before "." at the very end of the chunk.
        ("and later Revenue grew strongly today", "Revenue grew strongly."),
        # The first occurrence is glued to "x"; a later clean occurrence is enough.
        ("Revenue grew strongly today", "Revenue grew strongly."),
    ],
)
def test_quote_on_token_boundaries_passes(
    session: Session, source: Source, apple: Entity, quote: str, claim: str
) -> None:
    _, chunk = filed(session, source, apple, BOUNDARY)
    assert check(session, T0, apple, cite(chunk, quote, claim)).ok


@pytest.mark.parametrize(
    "quote",
    [
        # Starts inside "15.2", right after the digit 1.
        "5.2 million, up 3% from (a year ago)",
        # Starts inside "15.2": the char before is "." preceded by a digit.
        "2 million, up 3% from (a year ago)",
        # Starts inside "1,234": the char before is "," preceded by a digit.
        "234 units in total for the year",
        # Ends inside "$26.0": the char after is "." followed by a digit.
        "from (a year ago) and $26",
        # Starts inside a word.
        "evenue was $15.2 million, up 3%",
        # Ends inside a word.
        "Revenue was $15.2 mill",
    ],
)
def test_quote_cut_inside_a_token_fails(
    session: Session, source: Source, apple: Entity, quote: str
) -> None:
    _, chunk = filed(session, source, apple, BOUNDARY)
    assert len(quote) >= v.MIN_QUOTE_CHARS
    assert reasons(check(session, T0, apple, cite(chunk, quote))) == [
        (0, v.FailureReason.QUOTE_NOT_IN_CHUNK)
    ]


# --- numbers are compared by shape, not by meaning ---
# Every claim piece with a digit must appear among the quote's shapes: the piece as written, with
# thousands commas and trailing decimal zeros dropped, plus a scale word that follows it. Any
# notation the comparison does not know is rejected (fail closed); the price is false rejections.
SHAPES = (
    "The company returned $26.0 billion to shareholders. "
    "It repurchased 1,200 shares at $3.50 each. "
    "Margin rose 5% on pricing. "
    "Sales fell -12% in the quarter on weak demand. "
    "The Canadian unit raised C$5 billion in notes. "
    "The Israeli unit raised NIS 5 million in loans. "
    "Leases run over 5 years in total. "
    "Revenue in Q2 FY2027 was $46.7 billion overall. "
    "Total revenues were RMB99.9 billion for the year. "
    "The group raised $5 billion in bonds. "
    "The dividend was 5 cents per share. "
    "Sales in Europe reached €3.2 billion last year. "
    "Each employee received a $7k bonus in the year. "
    "It is an accelerated filer as defined in Rule 12b-2 of the Exchange Act. "
    "Item 1B. Unresolved Staff Comments: none."
)
RETURNED = "The company returned $26.0 billion to shareholders"
REPURCHASED = "It repurchased 1,200 shares at $3.50 each"
MARGIN_UP = "Margin rose 5% on pricing"
SALES_FELL = "Sales fell -12% in the quarter on weak demand"
AFTER_MINUS = "12% in the quarter on weak demand"
CANADA = "The Canadian unit raised C$5 billion in notes"
ISRAEL = "The Israeli unit raised NIS 5 million in loans"
LEASES = "Leases run over 5 years in total"
Q2 = "Revenue in Q2 FY2027 was $46.7 billion overall"
Q2_AMOUNT = "FY2027 was $46.7 billion overall"
RMB = "Total revenues were RMB99.9 billion for the year"
BONDS = "The group raised $5 billion in bonds"
CENTS = "The dividend was 5 cents per share"
EUR = "Sales in Europe reached €3.2 billion last year"
BONUS = "Each employee received a $7k bonus in the year"
RULE = "an accelerated filer as defined in Rule 12b-2 of the Exchange Act"
ITEM = "Item 1B. Unresolved Staff Comments: none"


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        ("Returned $26 billion.", RETURNED),
        ("Bought back 1200 shares at $3.5 each.", REPURCHASED),
        ("Margin rose 5 percent.", MARGIN_UP),
        ("Sales fell -12%.", SALES_FELL),
        # The quote starts after the minus, but its first piece is read whole: still -12%.
        ("Sales fell -12%.", AFTER_MINUS),
        ("Raised C$5 billion.", CANADA),
        ("Raised NIS 5 million.", ISRAEL),
        ("Revenue in Q2 FY2027 was $46.7 billion.", Q2),
        ("Revenue was RMB99.9 billion.", RMB),
        ("Raised $5 billion.", BONDS),
        ("A dividend of 5 cents.", CENTS),
        ("Sales reached €3.2 billion.", EUR),
        ("A $7k bonus was paid.", BONUS),
        # Rule numbers are pieces with digits like any other: present in the quote, they pass.
        ("It is an accelerated filer under Rule 12b-2.", RULE),
        ("Item 1B lists none.", ITEM),
    ],
)
def test_number_in_the_same_shape_passes(
    session: Session, source: Source, apple: Entity, claim: str, quote: str
) -> None:
    _, chunk = filed(session, source, apple, SHAPES)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    assert result.ok, result.failures


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        # Abbreviated scale: "bn" is not "billion".
        ("Returned $26bn.", RETURNED),
        # Currency dropped: "26 billion" is not "$26 billion".
        ("Returned 26 billion.", RETURNED),
        # A bare number is not the amount.
        ("Returned 26 to holders.", RETURNED),
        ("Returned $26 million.", RETURNED),
        ("Returned $25 billion.", RETURNED),
        # Another currency is not dollars, in either direction.
        ("Raised $5 billion.", CANADA),
        ("Raised $5 million.", ISRAEL),
        ("Raised US$5 billion.", BONDS),
        ("Sales reached $3.2 billion.", EUR),
        # Shape, not meaning: "EUR 3.2" is not "€3.2"; "RMB 99.9" (with a space) is not
        # "RMB99.9". Documented false rejections; the judge copies the filing's notation.
        ("Sales reached EUR 3.2 billion.", EUR),
        ("Revenue was RMB 99.9 billion.", RMB),
        ("Revenue was $99.9 billion.", RMB),
        # Sign dropped.
        ("Sales fell 12%.", SALES_FELL),
        ("Sales fell 12%.", AFTER_MINUS),
        # Labels must appear too: Q3 is not Q2, and a quote without them does not back them.
        ("Q3 FY2027 revenue was $46.7 billion.", Q2),
        ("Revenue in Q2 FY2027 was $46.7 billion.", Q2_AMOUNT),
        ("Revenue in Q2 was $46.7 billion.", Q2_AMOUNT),
        # A range glued into one piece is one shape the quote lacks.
        ("Returned $1B-2 billion.", RETURNED),
        # "5 percentage" is not "5".
        ("Leases grew 5 percentage points.", LEASES),
        # "5¢" is not "5 cents".
        ("A dividend of 5¢.", CENTS),
        # "$7 thousand" is not "$7k"; "7k" without "$" is not "$7k".
        ("A $7 thousand bonus was paid.", BONUS),
        ("The company hired 7k workers.", BONUS),
        # Rule and item numbers are not amounts, and an amount is not a rule number.
        ("Something worth $12 billion.", RULE),
        ("A $1 billion item.", ITEM),
        ("It is an accelerated filer under Rule 12b-2.", RETURNED),
        ("Executives sell under a 10b5-1 plan.", RETURNED),
        # Digits glued to letters are one piece the quote lacks (formerly UNCHECKED_NUMBER;
        # still rejected, now because the shape is absent).
        ("PDD revenue was XYZ999.9 billion.", RMB),
        ("Revenue grew in 1H24.", RMB),
        ("Revenue was 99.9billionish.", RMB),
    ],
)
def test_number_in_another_shape_fails(
    session: Session, source: Source, apple: Entity, claim: str, quote: str
) -> None:
    _, chunk = filed(session, source, apple, SHAPES)
    assert reasons(check(session, T0, apple, cite(chunk, quote, claim))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


@pytest.mark.parametrize(
    "label",
    ["Q2", "4Q", "FY27", "FY2027", "1H", "H2", "10-K", "10-K/A", "10-Q", "8-K", "20-F", "13F"],
)
def test_labels_must_appear_in_the_quote(
    session: Session, source: Source, apple: Entity, label: str
) -> None:
    # Formerly ignored as "known labels"; now a label is a piece with a digit like any other.
    _, chunk = filed(session, source, apple, SHAPES)
    claim = f"The company returned $26 billion, as reported in {label}."
    assert reasons(check(session, T0, apple, cite(chunk, RETURNED, claim))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


# --- a quote's pieces are read whole from the chunk ---
SPAN = (
    "Revenue changed -12% year over year and margins held. "
    "The net loss of $(1,234) million in the quarter was expected. "
    "The company returned $26.0 billion to shareholders this year."
)


@pytest.mark.parametrize(
    ("claim", "quote", "ok"),
    [
        # The minus sits just before the quote; the quoted piece is still "-12%".
        ("Revenue grew 12%.", "12% year over year and margins held", False),
        ("Revenue changed -12%.", "12% year over year and margins held", True),
        # "$(" sits before the quote; the piece is "$(1,234)", and "million" follows it.
        ("A loss of $1,234 million.", "1,234) million in the quarter was expected", False),
        ("A loss of $(1,234) million.", "1,234) million in the quarter was expected", True),
        # Shape, not meaning: "-1,234" is not "$(1,234)" (formerly passed).
        ("A loss of -1,234 million.", "1,234) million in the quarter was expected", False),
        # No digit of 26.0 is inside the quote, so it does not supply 26 billion.
        ("Returned $26 billion.", "billion to shareholders this year", False),
    ],
)
def test_quote_pieces_are_read_whole(
    session: Session, source: Source, apple: Entity, claim: str, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, SPAN)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


def test_scale_word_after_the_quote_end_still_counts(
    session: Session, source: Source, apple: Entity
) -> None:
    _, chunk = filed(session, source, apple, SHAPES)
    # The quote stops at "$26.0"; "billion" right after it is read with it.
    quote = "The company returned $26.0"
    assert check(session, T0, apple, cite(chunk, quote, "Returned $26 billion.")).ok
    assert reasons(check(session, T0, apple, cite(chunk, quote, "Returned $26."))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


# --- tables converted to text: a figure split over cells is joined ---
# Measured: ONTO 10-Q 0000950170-24-057270 onto-20240330.htm, text line 114 (chunk 484269 in the
# real store) reads "Other income (expense), net | 793 | (281 | )". Also measured on
# 2026-10-22: "| 19 | %" (e.g. NXP 8-K exhibit nxp4q22exhibit991.htm) and "(19 | %)" (Corning
# 10-Q glw-20250331.htm) in the real filing texts.
TABLE = (
    "Interest income, net | 7,361 | 3,448\n"
    "Other income (expense), net | 793 | (281 | )\n"
    "Gross margin of the segment | 19 | %\n"
    "Change in operating margin | (19 | %)\n"
    "Gross margin change | (5.2%) | 1.1%\n"
)
OTHER_INCOME = "Other income (expense), net | 793 | (281 | )"
SEGMENT = "Gross margin of the segment | 19 | %"
CHANGE = "Change in operating margin | (19 | %)"
MARGIN = "Gross margin change | (5.2%) | 1.1%"


@pytest.mark.parametrize(
    ("claim", "quote", "ok"),
    [
        ("Other income, net was (281).", OTHER_INCOME, True),
        # The quote may also be written joined, as normalise() reads the chunk.
        ("Other income, net was (281).", "Other income (expense), net | 793 | (281)", True),
        ("Other income, net was 281.", OTHER_INCOME, False),
        # Shape, not meaning: "-281" for "(281)" is a documented false rejection.
        ("Other income, net was -281.", OTHER_INCOME, False),
        ("Segment gross margin was 19%.", SEGMENT, True),
        ("Operating margin changed (19%).", CHANGE, True),
        ("Operating margin changed 19%.", CHANGE, False),
        ("Gross margin moved (5.2%).", MARGIN, True),
        ("Gross margin grew 5.2%.", MARGIN, False),
        # Formerly passed as the same meaning; now a different shape.
        ("Gross margin moved -5.2%.", MARGIN, False),
    ],
)
def test_table_figures_split_over_cells(
    session: Session, source: Source, apple: Entity, claim: str, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, TABLE)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


def test_footnote_marker_is_its_own_shape(session: Session, source: Source, apple: Entity) -> None:
    # "(1)" is the shape "(1)": a claim citing "note 1" is rejected, one copying "(1)" passes.
    _, chunk = filed(session, source, apple, "Revenue grew strongly (1) in the year.")
    quote = "Revenue grew strongly (1) in the year"
    assert check(session, T0, apple, cite(chunk, quote, "Revenue grew strongly (1).")).ok
    assert reasons(check(session, T0, apple, cite(chunk, quote, "See note 1 on revenue."))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


# Real sentence: PDD 20-F 0001104659-26-050727 (chunk 491084 in the store on 2026-10-22).
PDD_SENTENCE = (
    "We recorded a net income of RMB60,026.5 million, RMB112,434.5 million and "
    "RMB97,842.5 million (US$13,991.3 million) in 2023, 2024 and 2025, respectively."
)


@pytest.mark.parametrize(
    ("claim", "ok"),
    [
        # An unmatched "(" around "(US$13,991.3" is stripped: the piece is "US$13991.3".
        ("Net income was US$13,991.3 million in 2025.", True),
        ("Net income was RMB97,842.5 million in 2025.", True),
        ("Net income was -13,991.3 million in 2025.", False),
        # Formerly passed ("US$" read as "$"); now a different shape.
        ("Net income was $13,991.3 million in 2025.", False),
        ("Net income was (US$13,991.3 million) in 2025.", True),
    ],
)
def test_parenthesised_dollar_conversion(
    session: Session, source: Source, apple: Entity, claim: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, PDD_SENTENCE)
    result = check(session, T0, apple, cite(chunk, PDD_SENTENCE, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


# --- an amount in words cannot be compared ---


@pytest.mark.parametrize(
    "claim",
    [
        "Revenue was twenty-seven billion dollars.",
        "The platform has millions of users.",
        "Twelve million accounts were active.",
        "Several hundred stores opened.",
        "Margin rose by five percent.",
        "Sales grew by dozens of units.",
    ],
)
def test_amount_in_words_is_unchecked(
    session: Session, source: Source, apple: Entity, claim: str
) -> None:
    _, chunk = filed(session, source, apple, SHAPES)
    assert reasons(check(session, T0, apple, cite(chunk, RETURNED, claim))) == [
        (0, v.FailureReason.UNCHECKED_NUMBER)
    ]


def test_unchecked_number_comes_after_the_quote_and_before_the_numbers(
    session: Session, source: Source, apple: Entity
) -> None:
    _, chunk = filed(session, source, apple, SHAPES)
    # Both an amount in words and a missing number: the amount in words is reported.
    both = cite(chunk, RMB, "Revenue was $1 billion, or twenty billion in all.")
    assert reasons(check(session, T0, apple, both)) == [(0, v.FailureReason.UNCHECKED_NUMBER)]
    # A quote that is not in the chunk is reported before the claim is read.
    wrong_quote = cite(chunk, "Total revenues were RMB99.9 billion for the decade", "Millions.")
    assert reasons(check(session, T0, apple, wrong_quote)) == [
        (0, v.FailureReason.QUOTE_NOT_IN_CHUNK)
    ]


@pytest.mark.parametrize("gap", ["  ", "\n", "\t", " ", " \n "])
def test_claim_whitespace_is_normalised(
    session: Session, source: Source, apple: Entity, gap: str
) -> None:
    _, chunk = filed(session, source, apple, SHAPES)
    claim = f"The company returned $26{gap}billion."
    assert check(session, T0, apple, cite(chunk, RETURNED, claim)).ok


# --- currency words and codes in a claim must be in the quote ---
MONEY = (
    "Total revenues were RMB 5 billion for the year. "
    "The Israeli unit raised NIS 5 million in loans. "
    "Sales reached 5 billion yuan in China. "
    "The fee was 40 dollar per share in total."
)
RMB_SPACED = "Total revenues were RMB 5 billion for the year"
NIS = "The Israeli unit raised NIS 5 million in loans"
YUAN = "Sales reached 5 billion yuan in China"
DOLLAR = "The fee was 40 dollar per share in total"


@pytest.mark.parametrize(
    ("claim", "quote", "ok"),
    [
        ("Revenue was USD 5 billion.", RMB_SPACED, False),
        ("Revenue was 5 billion dollars.", RMB_SPACED, False),
        ("It raised 5 million dollars.", NIS, False),
        ("Revenue was RMB 5 billion.", RMB_SPACED, True),
        # No currency named in the claim: nothing to contradict.
        ("Revenue was 5 billion.", RMB_SPACED, True),
        ("Sales were 5 billion yuan.", YUAN, True),
        # Plural and singular are the same currency word.
        ("The fee was 40 dollars per share.", DOLLAR, True),
        ("Sales were 5 billion dollars.", YUAN, False),
    ],
)
def test_currency_words_and_codes_must_be_in_the_quote(
    session: Session, source: Source, apple: Entity, claim: str, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, MONEY)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


# --- quantity words without a number are unchecked; ordinary words are not ---
HALF = "Sales in the first half of 2024 rose as planned."
HALF_QUOTE = "Sales in the first half of 2024 rose as planned"


@pytest.mark.parametrize(
    "claim",
    [
        "Rates rose fifty basis points.",
        "The dividend was five cents.",
        "Margin rose ten percentage points.",
        "It announced a twenty-seven-billion-dollar buyback.",
        "Margin rose twenty-five%.",
        "Revenue doubled.",
        "Costs halved.",
        "Sales grew threefold.",
        "Cash of $ MM was held.",
        "Revenue grew twice as fast.",
    ],
)
def test_orphan_quantity_words_are_unchecked(
    session: Session, source: Source, apple: Entity, claim: str
) -> None:
    _, chunk = filed(session, source, apple, HALF)
    assert reasons(check(session, T0, apple, cite(chunk, HALF_QUOTE, claim))) == [
        (0, v.FailureReason.UNCHECKED_NUMBER)
    ]


@pytest.mark.parametrize(
    "claim",
    [
        "Costs as a percentage of revenue rose.",
        "Results on a GAAP basis held.",
        "Sales in the first half of 2024 rose.",
        "It is one of the largest firms.",
        "Key points were made.",
    ],
)
def test_ordinary_words_are_not_quantities(
    session: Session, source: Source, apple: Entity, claim: str
) -> None:
    _, chunk = filed(session, source, apple, HALF)
    result = check(session, T0, apple, cite(chunk, HALF_QUOTE, claim))
    assert result.ok, result.failures


# --- more scale words ---
SCALES = (
    "The fee was $5 per share for the period. "
    "Cash held was $5 MM at year end. "
    "Debt was 5 mln at close. "
    "Loans were 7 bln overall. "
    "Sales were 3 crore in India. "
    "Fees were 2 lakh in total."
)


@pytest.mark.parametrize(
    ("claim", "quote", "ok"),
    [
        ("Cash was $5 MM.", "The fee was $5 per share for the period", False),
        ("Cash was $5 MM.", "Cash held was $5 MM at year end", True),
        ("Debt was 5 mln.", "Debt was 5 mln at close", True),
        ("Debt was 5 million.", "Debt was 5 mln at close", False),
        ("Debt was 5.", "Debt was 5 mln at close", False),
        ("Loans were 7 bln.", "Loans were 7 bln overall", True),
        ("Sales were 3 crore.", "Sales were 3 crore in India", True),
        ("Sales were 3.", "Sales were 3 crore in India", False),
        ("Fees were 2 lakh.", "Fees were 2 lakh in total", True),
    ],
)
def test_more_scale_words(
    session: Session, source: Source, apple: Entity, claim: str, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, SCALES)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


# --- a scale word does not attach across the end of a clause ---


def test_scale_word_after_a_full_stop_is_not_read(
    session: Session, source: Source, apple: Entity
) -> None:
    _, chunk = filed(session, source, apple, "Revenue rose 5. Percent of sales grew too.")
    quote = "Revenue rose 5. Percent of sales grew too"
    assert check(session, T0, apple, cite(chunk, quote, "Revenue rose 5.")).ok
    assert reasons(check(session, T0, apple, cite(chunk, quote, "Revenue rose 5%."))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


def test_scale_word_after_a_comma_is_not_read(
    session: Session, source: Source, apple: Entity
) -> None:
    _, chunk = filed(session, source, apple, "In 2024, billions were invested in plants.")
    quote = "In 2024, billions were invested in plants"
    assert check(session, T0, apple, cite(chunk, quote, "In 2024 capital was invested.")).ok
    claim = "In 2024 billions were invested."
    assert reasons(check(session, T0, apple, cite(chunk, quote, claim))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


# --- a quote that stops on "$" or "(" carries none of the following digits ---


@pytest.mark.parametrize(
    ("text", "quote", "claim"),
    [
        (
            "It returned approximately $26.0 billion to shareholders.",
            "It returned approximately $",
            "Returned $26 billion.",
        ),
        (
            "The reported net loss was (1,234) for the period.",
            "The reported net loss was (",
            "The net loss was (1,234).",
        ),
    ],
)
def test_quote_ending_before_the_digits_does_not_supply_them(
    session: Session, source: Source, apple: Entity, text: str, quote: str, claim: str
) -> None:
    _, chunk = filed(session, source, apple, text)
    assert len(quote) >= v.MIN_QUOTE_CHARS
    assert reasons(check(session, T0, apple, cite(chunk, quote, claim))) == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


# --- table joins only a percent sign alone in its cell ---
HEADER_TABLE = (
    "Year | 2024 | % change | 2023\n"
    "Gross margin of the segment | 19 | %\n"
    "Change in operating margin | (19 | %)\n"
    "Last row of the table | 7 | %"
)


@pytest.mark.parametrize(
    ("claim", "quote", "ok"),
    [
        # "% change" is a column header, not the percent sign of 2024. Measured 2026-10-22:
        # "| 20xx | % change" in 50 real filing texts; a row ending "| 19 | %" in 3,615 (e.g.
        # Broadcom 8-K exhibit avgo-01292023x8kxex99.htm).
        ("Results for 2024 and 2023.", "Year | 2024 | % change | 2023", True),
        ("Growth of 2024%.", "Year | 2024 | % change | 2023", False),
        # A percent sign alone at the end of a row joins the figure before it.
        ("Segment margin was 19%.", "Gross margin of the segment | 19 | %", True),
        ("Segment margin was 19.", "Gross margin of the segment | 19 | %", False),
        ("Operating margin changed (19%).", "Change in operating margin | (19 | %)", True),
        # ... also at the very end of the chunk.
        ("The last row shows 7%.", "Last row of the table | 7 | %", True),
    ],
)
def test_table_percent_join(
    session: Session, source: Source, apple: Entity, claim: str, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, HEADER_TABLE)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


# --- a currency next to the number is part of its shape ---
# Shape of the convenience translations in 20-F filings, e.g. PDD 20-F 0001104659-26-050727
# (chunk 491084): "RMB97,842.5 million (US$13,991.3 million)". The spaced "RMB 12.3" and
# "US$ 1.7" layout with a space after the code: measured 2026-10-22 in 20-F texts, e.g.
# d38065d20f.htm ("RMB ... million (US$ ...").
TRANSLATED = (
    "The group reported revenue of RMB 12.3 billion (US$ 1.7 billion) in total for the year."
)
TRANSLATED_QUOTE = "revenue of RMB 12.3 billion (US$ 1.7 billion) in total"


@pytest.mark.parametrize(
    ("claim", "ok"),
    [
        # The dollar code belongs to 1.7, not to 12.3.
        ("Revenue of US$ 12.3 billion.", False),
        ("Revenue of RMB 12.3 billion.", True),
        # No currency named: the bare shape is recorded too.
        ("Revenue of 12.3 billion.", True),
        ("Revenue of US$ 1.7 billion.", True),
        ("Revenue of RMB 1.7 billion.", False),
    ],
)
def test_currency_code_binds_to_its_own_number(
    session: Session, source: Source, apple: Entity, claim: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, TRANSLATED)
    result = check(session, T0, apple, cite(chunk, TRANSLATED_QUOTE, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


CODES = "Total revenues were RMB 5 billion for the year. Fees were 5 billion USD in total."


@pytest.mark.parametrize(
    ("quote", "ok"),
    [
        ("Total revenues were RMB 5 billion for the year", False),
        # Codes compare case-insensitively: "usd" is USD.
        ("Fees were 5 billion USD in total", True),
    ],
)
def test_currency_code_after_the_scale_word(
    session: Session, source: Source, apple: Entity, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, CODES)
    result = check(session, T0, apple, cite(chunk, quote, "Revenue was 5 billion usd."))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


# --- count words must be in the quote ---
COUNTS = (
    "NVIDIA announced three acquisitions in the year. NVIDIA is one of the largest chip makers."
)


@pytest.mark.parametrize(
    ("claim", "quote", "ok"),
    [
        ("NVIDIA announced three acquisitions.", "NVIDIA is one of the largest chip makers", False),
        ("NVIDIA announced three acquisitions.", "NVIDIA announced three acquisitions", True),
        # "one" is not a checked count word.
        ("NVIDIA is one of the largest.", "NVIDIA is one of the largest chip makers", True),
    ],
)
def test_count_words_must_be_in_the_quote(
    session: Session, source: Source, apple: Entity, claim: str, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, COUNTS)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


@pytest.mark.parametrize(
    "claim",
    [
        "Revenue more than doubles.",
        "Revenue is doubling.",
        "Revenue will double.",
        # Accepted false rejection: "double-digit" reads as a multiple word and cannot be
        # compared; the judge writes the actual figure instead.
        "It posted double-digit growth.",
        "Costs are halving.",
    ],
)
def test_multiple_word_forms_are_unchecked(
    session: Session, source: Source, apple: Entity, claim: str
) -> None:
    _, chunk = filed(session, source, apple, HALF)
    assert reasons(check(session, T0, apple, cite(chunk, HALF_QUOTE, claim))) == [
        (0, v.FailureReason.UNCHECKED_NUMBER)
    ]


def test_first_half_is_not_a_multiple(session: Session, source: Source, apple: Entity) -> None:
    _, chunk = filed(session, source, apple, HALF)
    result = check(session, T0, apple, cite(chunk, HALF_QUOTE, "In the first half of 2024."))
    assert result.ok, result.failures


# --- "half" is a count word: it must be in the quote ---
HALVES = (
    "Sales in the six months ended June 30, 2024 rose as planned. "
    "Travel costs were cut in half during the year."
)


@pytest.mark.parametrize(
    ("claim", "quote", "ok"),
    [
        ("Travel costs were cut in half.", "Travel costs were cut in half during the year", True),
        (
            "Travel costs were cut in half.",
            "Sales in the six months ended June 30, 2024 rose as planned",
            False,
        ),
        # Documented false rejection: "first half" means the six months, but the words differ.
        (
            "In the first half of 2024 sales rose.",
            "Sales in the six months ended June 30, 2024 rose as planned",
            False,
        ),
    ],
)
def test_half_must_be_in_the_quote(
    session: Session, source: Source, apple: Entity, claim: str, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, HALVES)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


def test_ordinary_lowercase_word_is_not_a_currency_code(
    session: Session, source: Source, apple: Entity
) -> None:
    # "try" is the Turkish lira code only in capitals; lowercase codes count only for the
    # listed ones ("usd", "rmb", "eur", ...), as test_currency_code_after_the_scale_word shows.
    _, chunk = filed(session, source, apple, HALF)
    result = check(session, T0, apple, cite(chunk, HALF_QUOTE, "We try to grow in 2024."))
    assert result.ok, result.failures


# --- footnote markers glued to a number or a scale word ---

# Real text: Quanta Services (PWR) 8-K EX-99.1 0001193125-26-324855/d56853dex991.htm, chunk
# 509620 in the store on 2026-10-22. "*" marks a record result ("* = Record quarterly or record
# second quarter result"); it sits after the scale word ("Billion*") and after a bare amount
# ("$2.96*").
QUANTA = (
    "Second Quarter GAAP Diluted EPS of $2.96* and Adjusted Diluted EPS of $4.24*\n\n"
    "Net Income Attributable to Common Stock of $451.4 Million*\n\n"
    "Remaining Performance Obligations (RPO) of $33.6 Billion* and Total Backlog of "
    "$53.4 Billion*\n\n"
    "Significantly Increasing 2026 Financial Expectations Across All Metrics\n\n"
    "* = Record quarterly or record second quarter result"
)
QUANTA_RPO = (
    "Remaining Performance Obligations (RPO) of $33.6 Billion* and Total Backlog of $53.4 Billion*"
)
QUANTA_EPS = "GAAP Diluted EPS of $2.96* and Adjusted Diluted EPS of $4.24*"


def test_footnote_mark_after_a_scale_word_is_ignored(
    session: Session, source: Source, apple: Entity
) -> None:
    # Failed with number_not_in_quote before: "Billion*" was not read as the scale "billion".
    _, chunk = filed(session, source, apple, QUANTA)
    claim = "Total backlog reached $53.4 Billion, alongside RPO of $33.6 Billion."
    result = check(session, T0, apple, cite(chunk, QUANTA_RPO, claim))
    assert result.ok, result.failures


@pytest.mark.parametrize(
    ("claim", "quote", "ok"),
    [
        ("GAAP diluted EPS was $2.96 and adjusted EPS $4.24.", QUANTA_EPS, True),
        ("GAAP diluted EPS was $2.96* and adjusted EPS $4.24*.", QUANTA_EPS, True),
        # The mark is dropped, the digits are still compared exactly.
        ("GAAP diluted EPS was $2.97.", QUANTA_EPS, False),
        ("GAAP diluted EPS was $29.6.", QUANTA_EPS, False),
        ("GAAP diluted EPS was 2.96.", QUANTA_EPS, False),
        # The same guards after a scale word.
        ("Total backlog reached $53.5 Billion.", QUANTA_RPO, False),
        ("Total backlog reached $53.4 Million.", QUANTA_RPO, False),
        ("Total backlog reached 53.4 Billion.", QUANTA_RPO, False),
        ("Total backlog reached $534 Billion.", QUANTA_RPO, False),
    ],
)
def test_footnote_mark_does_not_loosen_the_number(
    session: Session, source: Source, apple: Entity, claim: str, quote: str, ok: bool
) -> None:
    _, chunk = filed(session, source, apple, QUANTA)
    result = check(session, T0, apple, cite(chunk, quote, claim))
    if ok:
        assert result.ok, result.failures
    else:
        assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]


@pytest.mark.parametrize("mark", ["*", "†", "‡", "§"])
def test_each_footnote_mark_is_ignored_at_the_end(
    session: Session, source: Source, apple: Entity, mark: str
) -> None:
    # Made-up sentence in the real layout; only "*" is measured (Quanta above).
    text = f"Total Backlog of $53.4 Billion{mark} and EPS of $2.96{mark} at quarter end."
    _, chunk = filed(session, source, apple, text)
    quote = f"Total Backlog of $53.4 Billion{mark} and EPS of $2.96{mark}"
    result = check(session, T0, apple, cite(chunk, quote, "Backlog was $53.4 Billion; EPS $2.96."))
    assert result.ok, result.failures


def test_footnote_mark_before_punctuation_is_ignored(
    session: Session, source: Source, apple: Entity
) -> None:
    text = "For the year, total backlog was $53.4 Billion*. Margins were stable."
    _, chunk = filed(session, source, apple, text)
    quote = "total backlog was $53.4 Billion*."
    assert check(session, T0, apple, cite(chunk, quote, "Total backlog was $53.4 Billion.")).ok


def test_footnote_mark_in_the_claim_only_is_ignored(
    session: Session, source: Source, apple: Entity
) -> None:
    text = "Remaining performance obligations were $33.6 Billion at quarter end."
    _, chunk = filed(session, source, apple, text)
    quote = "Remaining performance obligations were $33.6 Billion"
    result = check(session, T0, apple, cite(chunk, quote, "RPO was $33.6 Billion*."))
    assert result.ok, result.failures


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        # A mark inside the digits is not at the end: the piece keeps its shape "$5*3.4".
        ("Total Backlog of $5*3.4 Billion at quarter end.", "Total Backlog of $5*3.4 Billion"),
        ("Total Backlog of $53*.4 Billion at quarter end.", "Total Backlog of $53*.4 Billion"),
    ],
)
def test_footnote_mark_inside_a_number_is_not_dropped(
    session: Session, source: Source, apple: Entity, text: str, quote: str
) -> None:
    _, chunk = filed(session, source, apple, text)
    result = check(session, T0, apple, cite(chunk, quote, "Total backlog was $53.4 Billion."))
    assert reasons(result) == [(0, v.FailureReason.NUMBER_NOT_IN_QUOTE)]
