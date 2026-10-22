"""Guards against citations that point at the wrong text: lost words, broken spans, wrong sections.

Every chunk is later cited as (document id, chunk), and the Citation Validator re-reads
``full_text[char_start:char_end]``. A chunk whose span drifts from its text, or a filing whose
words vanish between HTML and text, makes a citation unverifiable.

This module has no ``as_of``: html_to_text and chunk_text are pure functions over one file's
content, so there is no time information to leak. Look-ahead tests live in test_edgar.py and
tests/store/test_documents.py. Assumptions here that only real data can confirm (check with
``audit_documents`` over the real text store and by reading the chunks of the standard cases):
- Assumed: a source newline inside an HTML paragraph is plain whitespace, not a line break
  (HTML semantics; older EDGAR files wrap their source lines at ~80 columns).
- Assumed: 10-K / 10-Q / 20-F section headings are standalone lines starting "PART <roman>" or
  "ITEM <n>[letter]", shorter than 150 characters. Check on the real NVDA 10-Q 0001045810-26-000075
  and the NVDA 10-K 0001045810-26-000021: section paths must include "Part I > Item 2" and
  "Part I > Item 1A" respectively, and no chunk of the table of contents may carry an Item path.
- Assumed: inline-XBRL hidden facts sit inside <ix:header>; dropping it removes nothing a reader
  of the rendered filing would see. Check: the NVDA 10-Q text must not start with XBRL tags or
  context ids (e.g. "c-1"), and must contain the cover-page sentence "QUARTERLY REPORT".
- Measured (fixture): the real NVDA EX-99.1 (q2fy27pr.htm) builds paragraphs from <div>, not
  <p>, and splits table numbers and "%" signs across cells.
"""

from pathlib import Path

import pytest

from asof.ingest.filing_text import ChunkSpan, chunk_text, html_to_text, plain_to_text

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"
# Measured: the real EX-99.1 of NVDA 8-K 0001045810-26-000073 (Q2 FY27 results), as served by
# https://www.sec.gov/Archives/edgar/data/1045810/000104581026000073/q2fy27pr.htm; pure ASCII.
PRESS_RELEASE = FIXTURES / "nvda-q2fy27pr.htm"


@pytest.fixture(scope="module")
def press_release_text() -> str:
    return html_to_text(PRESS_RELEASE.read_text(encoding="utf-8"))


def assert_chunk_invariants(text: str, chunks: list[ChunkSpan], max_chars: int) -> None:
    """The contract every caller relies on, checked span by span."""
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    previous_end = 0
    covered: set[int] = set()
    for chunk in chunks:
        assert chunk.text == text[chunk.char_start : chunk.char_end]
        assert chunk.char_start >= previous_end, "chunks overlap or are out of order"
        assert 0 < len(chunk.text) <= max_chars
        assert chunk.text == chunk.text.strip(), "chunk has leading/trailing whitespace"
        covered.update(range(chunk.char_start, chunk.char_end))
        previous_end = chunk.char_end
    lost = [i for i, ch in enumerate(text) if not ch.isspace() and i not in covered]
    assert not lost, f"{len(lost)} non-whitespace chars in no chunk, first at {lost[0]}"


# --- html_to_text: normal path on the real press release -----------------------------------------


def test_real_press_release_keeps_the_revenue_sentence(press_release_text: str) -> None:
    assert (
        "NVIDIA (NASDAQ: NVDA) today reported revenue for the second quarter ended July 26, 2026, "
        "of $96.2 billion, up 18% from the previous quarter and up 106% from a year ago."
    ) in press_release_text


def test_real_press_release_renders_table_rows_with_pipes(press_release_text: str) -> None:
    # Measured cells of the GAAP summary row: Revenue, $96,221, $81,615, $46,743, 18, %, 106, %.
    lines = press_release_text.splitlines()
    revenue_rows = [line for line in lines if line.startswith("Revenue | $96,221")]
    assert revenue_rows, "GAAP revenue row missing"
    assert revenue_rows[0].startswith("Revenue | $96,221 | $81,615 | $46,743")


def test_real_press_release_has_no_markup_or_raw_entities(press_release_text: str) -> None:
    assert "<" not in press_release_text
    assert "&#" not in press_release_text
    assert "&nbsp;" not in press_release_text
    assert "It’s doing useful work" in press_release_text  # &#8217; decoded
    assert "Workiva" not in press_release_text  # an HTML comment, not content
    assert "\n\n\n" not in press_release_text
    assert not any("  " in line for line in press_release_text.splitlines())
    assert press_release_text == press_release_text.strip()


# --- html_to_text: small exact cases ------------------------------------------------------------


def test_paragraphs_are_separated_by_one_blank_line() -> None:
    assert html_to_text("<p>One</p><p>Two</p>") == "One\n\nTwo"


def test_br_ends_a_line_without_a_blank_line() -> None:
    assert html_to_text("<p>a<br>b</p>") == "a\nb"


def test_table_rows_are_pipe_joined_and_the_table_is_set_off_by_blank_lines() -> None:
    html = (
        "<p>Before</p>"
        "<table><tr><td>Revenue</td><td></td><td><font>$</font><font>96,221</font></td></tr>"
        "<tr><td>Net income</td><td>&nbsp;</td><td>$59,688</td></tr></table>"
        "<p>After</p>"
    )
    assert html_to_text(html) == "Before\n\nRevenue | $96,221\nNet income | $59,688\n\nAfter"


def test_hidden_and_non_content_elements_are_dropped() -> None:
    html = (
        "<html><head><title>Document</title><style>p {color: red}</style></head><body>"
        "<script>var x = 1;</script>"
        "<ix:header><ix:hidden><ix:nonNumeric name='dei:EntityCentralIndexKey'>0001045810"
        "</ix:nonNumeric></ix:hidden></ix:header>"
        "<p>Body</p></body></html>"
    )
    assert html_to_text(html) == "Body"


def test_entities_are_decoded() -> None:
    assert html_to_text("<p>It&#8217;s R&amp;D&nbsp;spend</p>") == "It’s R&D spend"


def test_runs_of_spaces_collapse_and_ends_are_stripped() -> None:
    assert html_to_text("  <p>  a &nbsp;&nbsp;   b  </p>  ") == "a b"


def test_source_newlines_inside_a_paragraph_are_whitespace() -> None:
    # Assumed (HTML semantics): see the module docstring.
    assert html_to_text("<p>word\nwrapped\n  here</p>") == "word wrapped here"


def test_empty_blocks_do_not_stack_blank_lines() -> None:
    assert html_to_text("<p>a</p><p></p><div><br></div><p> </p><p>b</p>") == "a\n\nb"


def test_empty_html_gives_empty_text() -> None:
    assert html_to_text("") == ""
    assert html_to_text("<html><body><div>&nbsp;</div></body></html>") == ""


# --- chunk_text: non-periodic forms -------------------------------------------------------------


def test_empty_or_whitespace_text_has_no_chunks() -> None:
    assert chunk_text("", form="8-K", exhibit="EX-99.1") == []
    assert chunk_text("  \n\n \n", form="10-Q", exhibit=None) == []


def test_exhibit_names_the_section_of_a_non_periodic_form() -> None:
    text = "Para one.\n\nPara two."
    chunks = chunk_text(text, form="8-K", exhibit="EX-99.1")
    assert chunks == [ChunkSpan(0, "EX-99.1", 0, len(text), text)]


def test_form_names_the_section_when_there_is_no_exhibit() -> None:
    chunks = chunk_text("Only paragraph.", form="6-K", exhibit=None)
    assert [c.section_path for c in chunks] == ["6-K"]


def test_headings_are_ignored_outside_periodic_forms() -> None:
    text = "PART I\n\nItem 2. Results\n\nRevenue grew."
    chunks = chunk_text(text, form="8-K", exhibit=None)
    assert {c.section_path for c in chunks} == {"8-K"}


def test_paragraphs_are_packed_greedily_up_to_max_chars() -> None:
    p1, p2, p3 = "a" * 900, "b" * 900, "c" * 900
    text = f"{p1}\n\n{p2}\n\n{p3}"
    chunks = chunk_text(text, form="8-K", exhibit="EX-99.1", max_chars=2000)
    assert [c.text for c in chunks] == [f"{p1}\n\n{p2}", p3]
    assert_chunk_invariants(text, chunks, 2000)


def test_block_longer_than_max_chars_splits_at_line_boundaries() -> None:
    lines = [str(i) * 500 for i in range(1, 6)]
    text = "\n".join(lines)  # one block, 2504 chars
    chunks = chunk_text(text, form="8-K", exhibit=None, max_chars=1200)
    assert [c.text for c in chunks] == [
        f"{lines[0]}\n{lines[1]}",
        f"{lines[2]}\n{lines[3]}",
        lines[4],
    ]
    assert_chunk_invariants(text, chunks, 1200)


def test_single_line_longer_than_max_chars_is_split_hard() -> None:
    text = "x" * 5000
    chunks = chunk_text(text, form="8-K", exhibit=None, max_chars=2000)
    assert [(c.char_start, c.char_end) for c in chunks] == [(0, 2000), (2000, 4000), (4000, 5000)]
    assert_chunk_invariants(text, chunks, 2000)


def test_leading_and_trailing_whitespace_is_outside_every_chunk() -> None:
    text = "\n\n  First.  \n\n\n   Second.\n"
    chunks = chunk_text(text, form="8-K", exhibit=None)
    # Both blocks fit in one chunk; the whitespace between them stays inside it.
    assert [c.text for c in chunks] == ["First.  \n\n\n   Second."]
    assert_chunk_invariants(text, chunks, 2000)


# --- chunk_text: periodic forms -----------------------------------------------------------------

TEN_Q = "\n".join(
    [
        "NVIDIA CORPORATION",
        "FORM 10-Q",
        "",
        "TABLE OF CONTENTS",
        "",
        "Item 1. | Financial Statements | 3",
        "Item 2. | Management's Discussion and Analysis | 25",
        "",
        "PART I. FINANCIAL INFORMATION",
        "",
        "Item 1. Financial Statements",
        "",
        "Revenue was $96.2 billion.",
        "",
        "Item 2. Management's Discussion and Analysis of Financial Condition and Results of "
        "Operations",
        "",
        "Data Center revenue grew.",
        "",
        "PART II. OTHER INFORMATION",
        "",
        "ITEM 1A. RISK FACTORS",
        "",
        "Competition is intense.",
    ]
)


@pytest.mark.parametrize("form", ["10-Q", "10-K", "20-F", "10-K/A", "10-Q/A", "20-F/A"])
def test_periodic_headings_build_section_paths(form: str) -> None:
    chunks = chunk_text(TEN_Q, form=form, exhibit=None)
    assert [c.section_path for c in chunks] == [
        "Cover",
        "Part I",
        "Part I > Item 1",
        "Part I > Item 2",
        "Part II",
        "Part II > Item 1A",
    ]
    assert_chunk_invariants(TEN_Q, chunks, 2000)


def test_a_section_chunk_starts_at_its_heading() -> None:
    chunks = chunk_text(TEN_Q, form="10-Q", exhibit=None)
    by_path = {c.section_path: c.text for c in chunks}
    assert (
        by_path["Part I > Item 1"] == "Item 1. Financial Statements\n\nRevenue was $96.2 billion."
    )
    assert by_path["Part II > Item 1A"] == "ITEM 1A. RISK FACTORS\n\nCompetition is intense."


def test_table_of_contents_rows_are_not_headings() -> None:
    cover = chunk_text(TEN_Q, form="10-Q", exhibit=None)[0]
    assert cover.section_path == "Cover"
    assert "Item 1. | Financial Statements | 3" in cover.text
    assert "Item 2. | Management's Discussion and Analysis | 25" in cover.text


def test_long_line_starting_with_item_is_not_a_heading() -> None:
    sentence = "Item 7 of the credit agreement requires the company to " + "maintain ratios " * 10
    assert len(sentence) >= 150
    text = f"PART I\n\nItem 1. Business\n\nWe make GPUs.\n\n{sentence.strip()}"
    chunks = chunk_text(text, form="10-K", exhibit=None)
    assert [c.section_path for c in chunks] == ["Part I", "Part I > Item 1"]
    assert chunks[-1].text.endswith(sentence.strip())


def test_item_before_any_part_has_no_part_prefix() -> None:
    # Chosen behaviour: an Item seen before any PART heading gets the bare "Item N" path.
    text = "Cover page.\n\nItem 1. Identity of Directors\n\nNot applicable."
    chunks = chunk_text(text, form="20-F", exhibit=None)
    assert [c.section_path for c in chunks] == ["Cover", "Item 1"]


def test_a_chunk_never_spans_two_sections_even_when_both_fit() -> None:
    text = "PART I\n\nItem 1. A\n\nx\n\nItem 2. B\n\ny"
    chunks = chunk_text(text, form="10-Q", exhibit=None, max_chars=2000)
    assert [c.text for c in chunks] == ["PART I", "Item 1. A\n\nx", "Item 2. B\n\ny"]


def test_long_section_is_split_but_keeps_its_path() -> None:
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    text = f"PART I\n\nItem 2. MD&A\n\n{body}"
    chunks = chunk_text(text, form="10-Q", exhibit=None, max_chars=1000)
    assert len(chunks) > 3
    assert {c.section_path for c in chunks[1:]} == {"Part I > Item 2"}
    assert_chunk_invariants(text, chunks, 1000)


# --- chunk_text on the real press release ------------------------------------------------------


@pytest.mark.parametrize("max_chars", [2000, 500])
def test_real_press_release_chunks_satisfy_the_invariants(
    press_release_text: str, max_chars: int
) -> None:
    chunks = chunk_text(press_release_text, form="8-K", exhibit="EX-99.1", max_chars=max_chars)
    assert len(chunks) > 1
    assert {c.section_path for c in chunks} == {"EX-99.1"}
    assert_chunk_invariants(press_release_text, chunks, max_chars)


# --- review follow-ups ---------------------------------------------------------------------------


def test_display_none_on_a_tag_that_may_be_left_open_does_not_swallow_the_rest() -> None:
    # A <p> or <td> may legally omit its end tag; skipping until </p> could drop the whole tail.
    html = "<p style='display:none'>hidden<p>Visible one.<p>Visible two."
    text = html_to_text(html)
    assert "Visible one." in text and "Visible two." in text


def test_hidden_div_is_dropped() -> None:
    assert html_to_text("<div style='display: none'><div>x</div></div><p>Body</p>") == "Body"


def test_an_element_skipped_to_the_end_of_the_file_is_an_error() -> None:
    # Unclosed <ix:header>: everything after it would vanish without a word.
    with pytest.raises(ValueError, match="ix:header"):
        html_to_text("<p>Kept</p><ix:header><p>Revenue was $96.2 billion.</p>")


def test_plain_text_keeps_its_lines() -> None:
    raw = "EXHIBIT 99.1\r\n\r\n\r\nRevenue   was up.   \r\nMargins held.\r\n"
    assert plain_to_text(raw) == "EXHIBIT 99.1\n\nRevenue was up.\nMargins held."


def test_transition_reports_get_section_paths() -> None:
    text = "Cover page\n\nPART I\n\nItem 1. Business\n\nWe make drones."
    for form in ("10-KT", "10-QT", "10-KT/A"):
        paths = [c.section_path for c in chunk_text(text, form=form, exhibit=None)]
        assert paths[-1] == "Part I > Item 1", form
