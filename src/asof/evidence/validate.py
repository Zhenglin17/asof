"""Citation Validator: a conclusion is output only if every claim is backed by real evidence.

Each claim cites one chunk and copies a passage from it word for word (`quote`). Whether the
passage supports the claim is decided by code, not by another model, so the same draft always
gets the same verdict:

1. the evidence kind is supported (only text chunks in v1);
2. the chunk was public at the draft's as_of: an unknown id, a chunk not yet published and one
   already replaced by a revision all fail alike, so a caller cannot learn that a future chunk
   exists;
3. the chunk belongs to the company the draft is about;
4. the quote is MIN_QUOTE_CHARS to MAX_QUOTE_CHARS long and appears in the chunk
   (whitespace-normalised, a table figure split over cells joined, case-sensitive) starting
   and ending on word boundaries, so "5.2 million" cannot be cut out of "$15.2 million";
5. every piece of the claim that holds a digit appears in the quote in the same shape: as
   written, up to thousands commas and trailing decimal zeros, with the scale word after it.
   "$26 billion" matches "$26.0 billion"; "$26bn", "C$26 billion", "26 billion" (when the
   quote has "$"), "12%" for "-12%" and "Q3" against a Q2 release do not. A quote is read in
   whole words, so starting it after a minus sign does not drop the sign; a number counts only
   if some of its digits are inside the quote. A currency next to a number is part of its
   shape ("RMB 12.3 billion" is not "US$ 12.3 billion"), and any currency or count written as
   a word ("USD", "dollars", "three") must be in the quote too. A quantity in words with no
   number ("twenty-seven billion", "fifty basis points", "doubling") is rejected, since it
   could not be compared.

Matching shapes rather than meanings rejects any notation it does not know instead of passing
it; the price is false rejections ("-281" for a table's "(281)"), which the judge fixes by
copying the filing's notation. Not caught: a claim that keeps the numbers but flips the words
around them ("fell 12%" for "grew 12%"); a bare table figure whose unit and currency are only
in the table header ("96,212" from a table "in millions"); a figure moved to another row of a
table inside the same quote (bounded by MAX_QUOTE_CHARS); a currency or multiple outside the
word lists below ("lira" is listed, "dirham" is not; "cut in half" and "two-thirds" pass only
if "two" / "half" is quoted).

One failing citation rejects the whole draft. Every failing citation is reported, each with the
first check it failed, so a caller can retry with precise feedback.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from sqlmodel import Session

from asof.store.models import EvidenceKind
from asof.store.queries import visible_chunk

MIN_QUOTE_CHARS = 20
# A quote is a passage, not the whole chunk; a long one would let a claim pick any number in it.
MAX_QUOTE_CHARS = 500

# Numbers are compared by shape, not by meaning. Every whitespace-separated piece of the claim
# that holds a digit must appear in the quote in the same shape, where a shape is the piece as
# written ("$26.0", "C$5", "(281)", "-12%", "RMB99.9", "Q2", "FY2027") with thousands commas
# and trailing decimal zeros dropped, plus the next word when it gives the scale ("billion",
# "percent", "B"). Reading meaning out of filings never ends (currencies, table layouts, signs,
# labels); comparing shapes rejects whatever it does not recognise instead of passing it.
_NUMERAL = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_SCALE = {
    "hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion",
    "billions", "trillion", "trillions", "bn", "mn", "mm", "mln", "bln", "tn", "b", "m", "k",
    "crore", "crores", "lakh", "lakhs", "percent", "percentage", "points", "pp", "bps", "basis",
    "cent", "cents", "dozen", "dozens",
}  # fmt: skip
# Quantity words with no number in front ("twenty-seven billion", "fifty basis points", "five
# cents", "doubled") state an amount nobody can compare, so the claim is rejected.
_ORPHAN_SCALE = {
    "hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion",
    "billions", "trillion", "trillions", "bn", "mn", "mm", "mln", "bln", "tn", "crore",
    "crores", "lakh", "lakhs", "percent", "bps", "cent", "cents", "dozen", "dozens",
}  # fmt: skip
_ORPHAN_BEFORE_POINTS = {"basis", "percentage"}
_MULTIPLE = re.compile(r"doubl\w*|tripl\w*|quadrupl\w*|halv\w*|\w+fold|twice|thrice")
# A count in words ("three acquisitions", "half of revenue") must be in the quote as a word.
_NUMBER_WORDS = {
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "half", "halves",
}  # fmt: skip
# A currency written as its own word is part of the amount: "USD 5 billion" must not stand on
# a quote that says "RMB 5 billion". Any of these in a claim must also be in the quote. A
# currency outside these lists goes unchecked.
_CURRENCY_CODES = {
    "USD", "US$", "$", "RMB", "CNY", "HKD", "HK$", "TWD", "NTD", "NT$", "EUR", "GBP", "JPY",
    "INR", "KRW", "CAD", "C$", "AUD", "A$", "NZD", "CHF", "SGD", "S$", "BRL", "R$", "MXN",
    "ZAR", "SEK", "NOK", "DKK", "ILS", "NIS", "RUB", "TRY", "IDR", "THB", "MYR", "PHP", "VND",
    "SAR", "AED", "PLN", "CZK", "HUF", "CLP", "COP", "ARS", "€", "£", "¥", "₩", "₹",
}  # fmt: skip
# Codes also read in lower case; not "try", "cop" or "php", which are English words or names.
_LOWERCASE_CODES = {
    "usd", "rmb", "cny", "hkd", "twd", "ntd", "eur", "gbp", "jpy", "inr", "krw", "cad", "aud",
    "nzd", "chf", "sgd", "brl", "mxn", "zar", "sek", "nok", "dkk", "ils", "nis", "rub", "idr",
    "thb", "myr", "vnd", "aed", "pln",
}  # fmt: skip
_CURRENCY_WORDS = {
    "dollar", "yuan", "renminbi", "euro", "yen", "won", "rupee", "shekel", "pound", "franc",
    "peso", "krona", "krone", "kronor", "ringgit", "baht", "rupiah", "rand", "ruble", "lira",
}  # fmt: skip
# Tables converted to text split one figure over cells: "(281 | )", "19 | %", "$ | 96,212".
_TABLE_JOINS = [
    (re.compile(r"(?<=\d)[ \t]*\|[ \t]*\)"), ")"),
    # Only a percent sign alone in its cell: "| 2024 | % change" stays as it is.
    (re.compile(r"(?<=[\d)])[ \t]*\|[ \t]*%(?=[ \t]*(?:\)|\||\n|$))"), "%"),
    (re.compile(r"\$[ \t]*\|[ \t]*(?=[\d(])"), "$"),
]
_TRAILING = ".,;:!?"
# Footnote marks glued to the end of a figure or its scale word ("$53.4 Billion*", "$2.96†") are
# dropped like punctuation, but unlike punctuation they do not end a clause. A mark inside the
# digits ("$5*3.4") is kept, so the figure keeps its shape and fails.
_FOOTNOTE = "*†‡§"


class FailureReason(StrEnum):
    NO_CITATIONS = "no_citations"
    UNSUPPORTED_KIND = "unsupported_kind"
    NOT_VISIBLE = "not_visible"
    ENTITY_MISMATCH = "entity_mismatch"
    MISSING_QUOTE = "missing_quote"
    QUOTE_TOO_SHORT = "quote_too_short"
    QUOTE_TOO_LONG = "quote_too_long"
    QUOTE_NOT_IN_CHUNK = "quote_not_in_chunk"
    UNCHECKED_NUMBER = "unchecked_number"
    NUMBER_NOT_IN_QUOTE = "number_not_in_quote"


@dataclass(frozen=True)
class ClaimCitation:
    claim: str
    evidence_kind: str
    chunk_id: int | None
    quote: str | None
    evidence_ref: str | None = None


@dataclass(frozen=True)
class DecisionDraft:
    as_of: datetime
    entity_id: int | None
    citations: list[ClaimCitation]


@dataclass(frozen=True)
class CitationFailure:
    # Position in DecisionDraft.citations; -1 for a failure of the draft as a whole.
    index: int
    reason: FailureReason
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    failures: list[CitationFailure]

    @property
    def ok(self) -> bool:
        return not self.failures


def normalise_space(s: str) -> str:
    """Collapse every run of whitespace (newline, tab, no-break space) to one space."""
    return " ".join(s.split())


def normalise(s: str) -> str:
    """Whitespace collapsed and a table figure split over cells joined: "(281 | )" -> "(281)"."""
    # Joined before whitespace is collapsed, while a row's end is still a line break.
    for pattern, joined in _TABLE_JOINS:
        s = pattern.sub(joined, s)
    return normalise_space(s)


def _bare(piece: str) -> str:
    """A piece without sentence punctuation, footnote marks or an unmatched parenthesis around
    it."""
    while True:
        before = piece
        piece = piece.rstrip(_TRAILING).rstrip(_FOOTNOTE)
        if piece.endswith(")") and "(" not in piece:
            piece = piece[:-1]
        if piece.startswith("(") and ")" not in piece:
            piece = piece[1:]
        if piece == before:
            return piece


def _numeral(m: re.Match[str]) -> str:
    value = m[0].replace(",", "")
    return value.rstrip("0").rstrip(".") if "." in value else value


def _currency(part: str) -> str | None:
    if part in _CURRENCY_CODES or part in _LOWERCASE_CODES:
        return part.upper()
    word = part.lower()
    singular = word[:-1] if word.endswith("s") else word
    return singular if singular in _CURRENCY_WORDS or word in _CURRENCY_WORDS else None


def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


@dataclass(frozen=True)
class _Reading:
    shapes: set[str]
    # Currencies and counts written as words; each must also be in the quote.
    terms: set[str]
    # Quantity words no number stands in front of.
    orphans: list[str]


def _read(pieces: list[str], indices: Iterable[int], *, quote: bool = False) -> _Reading:
    """Number shapes, worded terms and orphan quantity words of the pieces at indices.

    The scale word after a number, and a currency right before it or after its scale word, are
    read from the neighbouring pieces even when they lie outside indices, unless a clause ends
    in between ("rose 5. Percent of ..." is not 5%). A currency joins the shape: "RMB 12.3
    billion" is not "US$ 12.3 billion". In a quote the shape also counts without its currency,
    so a claim that names no currency still matches.
    """
    reading = _Reading(set(), set(), [])

    def neighbour(i: int) -> str:
        return _bare(pieces[i]) if 0 <= i < len(pieces) else ""

    for i in indices:
        piece = _bare(pieces[i])
        if _has_digit(piece):
            shape = _NUMERAL.sub(_numeral, piece)
            nxt = i + 1
            if pieces[i][-1] not in _TRAILING:
                following = neighbour(nxt).lower()
                if following == "percent":
                    shape += "%"
                    nxt += 1
                elif following in _SCALE:
                    shape += " " + following
                    nxt += 1
            prefix = None
            if i > 0 and pieces[i - 1][-1] not in _TRAILING:
                prefix = _currency(neighbour(i - 1))
            # Only a currency word follows its amount ("5 billion dollars"); a code after it, as
            # in "RMB 12.3 billion (US$ 1.7 billion)", belongs to the next amount.
            suffix = None
            after = pieces[nxt] if nxt < len(pieces) else ""
            if pieces[nxt - 1][-1] not in _TRAILING and not after.startswith("("):
                suffix = _currency(neighbour(nxt))
                suffix = suffix if suffix is not None and suffix.islower() else None
            full = " ".join(p for p in (prefix, shape, suffix) if p)
            reading.shapes.add(full)
            if quote:
                reading.shapes.add(shape)
            continue
        parts = piece.split("-")
        for k, part in enumerate(parts):
            previous = parts[k - 1] if k else neighbour(i - 1)
            after = parts[k + 1] if k + 1 < len(parts) else neighbour(i + 1)
            word = part.lower()
            number_before = _has_digit(previous)
            scale = word in _ORPHAN_SCALE
            scale = scale or (word in _ORPHAN_BEFORE_POINTS and after.lower() == "points")
            if (scale and not number_before) or part.endswith("%") or _MULTIPLE.fullmatch(word):
                reading.orphans.append(piece)
            currency = _currency(part)
            if currency is not None:
                reading.terms.add(currency)
            if word in _NUMBER_WORDS:
                reading.terms.add(word)
    return reading


def claim_reading(claim: str) -> _Reading:
    pieces = normalise(claim).split(" ")
    return _read(pieces, range(len(pieces)))


def quote_reading(text: str, start: int, end: int) -> _Reading:
    """What the quote text[start:end] states, read in whole pieces of the chunk: a quote that
    starts after a minus sign or stops before "billion" still carries them. A number counts
    only if some of its digits are inside the quote."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    pieces = [text[a:b] for a, b in spans]
    inside = [
        i
        for i, (a, b) in enumerate(spans)
        if a < end
        and b > start
        and (not _has_digit(pieces[i]) or _has_digit(text[max(a, start) : min(b, end)]))
    ]
    return _read(pieces, inside, quote=True)


def _glued(text: str, i: int, step: int) -> bool:
    """Whether text[i] continues the word or number that the quote's edge would cut."""
    if not 0 <= i < len(text):
        return False
    c = text[i]
    if c.isalnum():
        return True
    j = i + step
    return c in ".," and 0 <= j < len(text) and text[j].isdigit()


def quote_spans(quote: str, text: str) -> list[tuple[int, int]]:
    """Where quote occurs in text without cutting a word or number at either end."""
    spans = []
    start = text.find(quote)
    while start != -1:
        end = start + len(quote)
        cut_before = quote[0].isalnum() and _glued(text, start - 1, -1)
        cut_after = quote[-1].isalnum() and _glued(text, end, 1)
        if not cut_before and not cut_after:
            spans.append((start, end))
        start = text.find(quote, start + 1)
    return spans


def _check(
    session: Session, draft: DecisionDraft, citation: ClaimCitation
) -> tuple[FailureReason, str] | None:
    if citation.evidence_kind != EvidenceKind.CHUNK:
        return FailureReason.UNSUPPORTED_KIND, f"{citation.evidence_kind!r} is not checked in v1"
    found = None
    if citation.chunk_id is not None:
        found = visible_chunk(session, draft.as_of, citation.chunk_id)
    if found is None:
        return (
            FailureReason.NOT_VISIBLE,
            f"no chunk {citation.chunk_id} was public at {draft.as_of.isoformat()}",
        )
    chunk, document = found
    if draft.entity_id is None or document.entity_id != draft.entity_id:
        return (
            FailureReason.ENTITY_MISMATCH,
            f"chunk {chunk.id} belongs to entity {document.entity_id}, "
            f"the decision is about {draft.entity_id}",
        )
    if citation.quote is None:
        return FailureReason.MISSING_QUOTE, "a chunk citation needs a verbatim quote"
    quote = normalise(citation.quote)
    if len(quote) < MIN_QUOTE_CHARS:
        return (
            FailureReason.QUOTE_TOO_SHORT,
            f"quote has {len(quote)} characters, needs {MIN_QUOTE_CHARS}",
        )
    if len(quote) > MAX_QUOTE_CHARS:
        return (
            FailureReason.QUOTE_TOO_LONG,
            f"quote has {len(quote)} characters, at most {MAX_QUOTE_CHARS}",
        )
    text = normalise(chunk.text)
    spans = quote_spans(quote, text)
    if not spans:
        return (
            FailureReason.QUOTE_NOT_IN_CHUNK,
            f"chunk {chunk.id} does not contain {quote!r} as a whole passage",
        )
    claim = claim_reading(citation.claim)
    if claim.orphans:
        return (
            FailureReason.UNCHECKED_NUMBER,
            f"claim states an amount in words: {claim.orphans}; write it as the quote does",
        )

    def missing(span: tuple[int, int]) -> set[str]:
        quoted = quote_reading(text, *span)
        return (claim.shapes - quoted.shapes) | (claim.terms - quoted.terms)

    absent = min((missing(span) for span in spans), key=len)
    if absent:
        return (
            FailureReason.NUMBER_NOT_IN_QUOTE,
            f"claim states {sorted(absent)}, which the quote does not show in that form",
        )
    return None


def validate(session: Session, draft: DecisionDraft) -> ValidationResult:
    if draft.as_of.tzinfo is None or draft.as_of.utcoffset() is None:
        raise ValueError(f"as_of must carry a timezone, got naive {draft.as_of!r}")
    if not draft.citations:
        return ValidationResult(
            [CitationFailure(-1, FailureReason.NO_CITATIONS, "a decision needs evidence")]
        )
    failures = []
    for index, citation in enumerate(draft.citations):
        failed = _check(session, draft, citation)
        if failed is not None:
            failures.append(CitationFailure(index, *failed))
    return ValidationResult(failures)
