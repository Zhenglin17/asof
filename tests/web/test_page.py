"""Guards against a demo page that loads blank: a data file the page fetches that the build no
longer writes, an external script that is unpinned or comes from an unexpected host, and grid
keys whose number format differs from the one the page builds ("1.0", never "1")."""

import json
import re
from pathlib import Path

from asof.demo.build import CONVICTIONS, RISKS, TRAILS, grid, run_key

WEB = Path(__file__).resolve().parents[2] / "web"
PAGE = (WEB / "index.html").read_text()


def test_every_fetched_data_file_exists() -> None:
    listed = re.search(r"const names = \[([^\]]+)\]", PAGE)
    assert listed is not None
    files = [n.strip().strip('"') for n in listed.group(1).split(",")]
    assert files == ["meta", "runs", "decisions", "chunks", "timeline", "entities"]
    for name in files:
        assert (WEB / "data" / f"{name}.json").is_file(), name


def test_external_scripts_are_pinned_to_cdnjs() -> None:
    sources = re.findall(r'<script[^>]*\bsrc="([^"]+)"', PAGE)
    assert sources
    tags = re.findall(r"<script[^>]*\bsrc=[^>]*>", PAGE)
    # a changed file on the CDN must not run: every external script carries its hash
    assert all(re.search(r'integrity="sha(384|512)-[A-Za-z0-9+/=]+"', t) for t in tags), tags
    for src in sources:
        assert re.fullmatch(
            r"https://cdnjs\.cloudflare\.com/ajax/libs/[\w.-]+/\d+\.\d+\.\d+/[\w./-]+", src
        ), src


def test_grid_keys_keep_one_decimal() -> None:
    # The page builds keys like "llm|1.0|0.25|0.0"; Python's str(1.0) is "1.0", JavaScript's "1".
    assert run_key("take_all", 1.0, 0.25) == "take_all|1.0|0.25"
    assert run_key("llm", 2.0, 1.0, 0.0) == "llm|2.0|1.0|0.0"
    runs = json.loads((WEB / "data" / "runs.json").read_text())["runs"]
    expected = {run_key("take_all", t, r) for t in TRAILS for r in RISKS}
    expected |= {run_key("llm", t, r, c) for t in TRAILS for r in RISKS for c in CONVICTIONS}
    assert len(expected) == len(grid())
    assert set(runs) == expected | {"long"}
    decisions = json.loads((WEB / "data" / "decisions.json").read_text())
    assert set(decisions) == expected


def test_meta_window_ends_on_the_last_session_with_data() -> None:
    meta = json.loads((WEB / "data" / "meta.json").read_text())
    runs = json.loads((WEB / "data" / "runs.json").read_text())
    assert meta["window"]["end"] == runs["benchmark"][-1]["date"]
    assert meta["long_window"]["end"] == runs["long_benchmark"][-1]["date"]
    assert meta["initial_equity"] > 0
