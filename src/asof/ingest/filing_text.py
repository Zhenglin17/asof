"""Filing HTML to plain text, and plain text to citable chunks.

Both functions are pure. The text they produce is what a citation points into: every chunk is
an exact slice ``text[char_start:char_end]``, so the Citation Validator can re-read it.
"""

import re
from dataclasses import dataclass
from html.parser import HTMLParser

# Elements that end a paragraph. EDGAR filings build paragraphs from <div> far more often than <p>
# (the NVDA Q2 FY27 press release has no <p> at all), so <div> counts as one.
_BLOCK = frozenset(
    {
        "address", "article", "blockquote", "body", "center", "dd", "div", "dl", "dt", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "hr", "html", "li", "ol", "p", "pre", "section", "ul",
    }
)  # fmt: skip
# Never content. <ix:header> holds the hidden inline-XBRL facts a reader of the filing never sees.
_SKIP = frozenset({"head", "script", "style", "title", "ix:header"})
_VOID = frozenset({"area", "base", "br", "col", "hr", "img", "input", "link", "meta", "wbr"})
_HIDDEN = re.compile(r"display\s*:\s*none", re.IGNORECASE)
# Hidden elements are skipped up to their end tag, so only tags whose end tag is never left out:
# a hidden <p>, <td> or <li> may close implicitly, and waiting for </p> would drop the rest.
_HIDEABLE = frozenset({"div", "span", "table", "section", "ix:hidden"})
_SPACES = re.compile(r"\s+")


def _clean(fragment: str) -> str:
    return _SPACES.sub(" ", fragment).strip()


class _TextBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._line: list[str] = []
        self._blank_pending = False
        # Skipping: the tag that started it and how deep its own nesting has gone.
        self._skip_tag: str | None = None
        self._skip_depth = 0
        # Only the outermost table's rows and cells shape the output; nested tables are flattened
        # into the cell that holds them.
        self._table_depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    # -- output ------------------------------------------------------------------------------

    def _emit(self, line: str) -> None:
        if not line:
            return
        if self._blank_pending and self.lines:
            self.lines.append("")
        self._blank_pending = False
        self.lines.append(line)

    def _end_line(self) -> None:
        self._emit(_clean("".join(self._line)))
        self._line = []

    def _end_paragraph(self) -> None:
        self._end_line()
        self._blank_pending = True

    def _end_cell(self) -> None:
        if self._cell is not None and self._row is not None:
            self._row.append(_clean("".join(self._cell)))
        self._cell = None

    def _end_row(self) -> None:
        self._end_cell()
        if self._row is not None:
            self._end_line()
            self._emit(" | ".join(cell for cell in self._row if cell))
        self._row = None

    # -- parser callbacks --------------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            elif self._skip_tag == "head" and tag == "body":
                # </head> may be left out; <body> ends the head all the same.
                self._skip_tag = None
                self._structure(tag, opening=True)
            return
        style = dict(attrs).get("style") or ""
        if tag in _SKIP or (tag in _HIDEABLE and _HIDDEN.search(style)):
            self._skip_tag, self._skip_depth = tag, 1
            return
        self._structure(tag, opening=True)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_tag is None:
            self._structure(tag, opening=True)

    def handle_endtag(self, tag: str) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        self._structure(tag, opening=False)

    def handle_data(self, data: str) -> None:
        if self._skip_tag is not None:
            return
        if self._table_depth:
            if self._cell is None:
                if not data.strip():
                    return
                # Text loose in a table (a cell tag left out): keep it as a cell of its own.
                if self._row is None:
                    self._row = []
                self._cell = []
            self._cell.append(data)
        else:
            self._line.append(data)

    def _structure(self, tag: str, *, opening: bool) -> None:
        if tag == "table":
            if opening:
                self._table_depth += 1
                if self._table_depth == 1:
                    self._end_paragraph()
                elif self._cell is not None:
                    self._cell.append(" ")
            else:
                if self._table_depth == 1:
                    self._end_row()
                    self._end_paragraph()
                self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth:
            outer = self._table_depth == 1
            if outer and tag == "tr":
                self._end_row()
                if opening:
                    self._row = []
            elif outer and tag in ("td", "th"):
                self._end_cell()
                if opening:
                    if self._row is None:
                        self._row = []
                    self._cell = []
            elif self._cell is not None and (tag in _BLOCK or tag in ("br", "td", "th", "tr")):
                self._cell.append(" ")
            return
        if tag == "br":
            self._end_line()
        elif tag in _BLOCK:
            self._end_paragraph()

    def finish(self) -> str:
        self.close()
        if self._skip_tag is not None:
            raise ValueError(f"<{self._skip_tag}> is never closed; everything after it was dropped")
        self._end_row()
        self._end_line()
        return "\n".join(self.lines)


def html_to_text(html: str) -> str:
    """Readable text of a filing: one paragraph per block, table rows as ``cell | cell``."""
    builder = _TextBuilder()
    builder.feed(html)
    return builder.finish()


def plain_to_text(raw: str) -> str:
    """A plain-text exhibit: its own line breaks kept, spaces collapsed, one blank line at most."""
    lines = [_clean(line) for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


@dataclass(frozen=True)
class ChunkSpan:
    ordinal: int
    section_path: str
    char_start: int
    char_end: int
    text: str


PERIODIC_FORMS = frozenset(
    {"10-K", "10-Q", "20-F", "10-KT", "10-QT"}
    | {f"{f}/A" for f in ("10-K", "10-Q", "20-F", "10-KT", "10-QT")}
)
_PART = re.compile(r"^PART\s+([IVX]+)\b", re.IGNORECASE)
_ITEM = re.compile(r"^ITEM\s+(\d+[A-Z]?)\b", re.IGNORECASE)
_MAX_HEADING = 150


def _heading(line: str) -> tuple[str, str] | None:
    """("part", "II") or ("item", "1A") when a line is a section heading."""
    if len(line) >= _MAX_HEADING or " | " in line:
        return None
    if m := _PART.match(line):
        return "part", m.group(1).upper()
    if m := _ITEM.match(line):
        return "item", m.group(1).upper()
    return None


def _lines(text: str) -> list[tuple[int, int]]:
    """Trimmed (start, end) of every non-blank line, or (-1, -1) for a blank line."""
    spans: list[tuple[int, int]] = []
    start = 0
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped:
            begin = start + (len(raw) - len(raw.lstrip()))
            spans.append((begin, begin + len(stripped)))
        else:
            spans.append((-1, -1))
        start += len(raw) + 1
    return spans


def _sections(
    text: str, periodic: bool, default: str
) -> list[tuple[str, list[list[tuple[int, int]]]]]:
    """Sections in order, each a list of blocks, each block a list of trimmed line spans."""
    sections: list[tuple[str, list[list[tuple[int, int]]]]] = []
    path = "Cover" if periodic else default
    part: str | None = None
    blocks: list[list[tuple[int, int]]] = []
    block: list[tuple[int, int]] = []

    def close_block() -> None:
        nonlocal block
        if block:
            blocks.append(block)
        block = []

    for span in _lines(text):
        if span[0] < 0:
            close_block()
            continue
        found = _heading(text[span[0] : span[1]]) if periodic else None
        if found is not None:
            close_block()
            if blocks:
                sections.append((path, blocks))
            blocks = []
            kind, label = found
            if kind == "part":
                part = f"Part {label}"
                path = part
            else:
                path = f"{part} > Item {label}" if part else f"Item {label}"
        block.append(span)
    close_block()
    if blocks:
        sections.append((path, blocks))
    return sections


def _atoms(text: str, block: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    """Pieces no longer than max_chars: the whole block, else its lines, else hard cuts."""
    if block[-1][1] - block[0][0] <= max_chars:
        return [(block[0][0], block[-1][1])]
    pieces: list[tuple[int, int]] = []
    for start, end in block:
        while end - start > max_chars:
            cut = start + max_chars
            piece_end = start + len(text[start:cut].rstrip())
            pieces.append((start, piece_end))
            start = cut + (len(text[cut:end]) - len(text[cut:end].lstrip()))
        if start < end:
            pieces.append((start, end))
    return pieces


def chunk_text(
    text: str, *, form: str, exhibit: str | None, max_chars: int = 2000
) -> list[ChunkSpan]:
    """Greedy packing of paragraphs into chunks of at most max_chars, never across sections."""
    periodic = form in PERIODIC_FORMS
    chunks: list[ChunkSpan] = []
    for path, blocks in _sections(text, periodic, exhibit or form):
        atoms = [atom for block in blocks for atom in _atoms(text, block, max_chars)]
        start, end = atoms[0]
        for atom_start, atom_end in atoms[1:]:
            if atom_end - start <= max_chars:
                end = atom_end
                continue
            chunks.append(ChunkSpan(len(chunks), path, start, end, text[start:end]))
            start, end = atom_start, atom_end
        chunks.append(ChunkSpan(len(chunks), path, start, end, text[start:end]))
    return chunks
