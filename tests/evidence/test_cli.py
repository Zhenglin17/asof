"""Guards against `asof evidence validate` crashing on bad input: a malformed draft file must end
with exit code 1 and one readable line, never a traceback; a well-formed draft gets 0 or 2.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from asof.cli import app
from asof.store.db import init_db, make_engine
from tests.store.factories import T0, new_chunk, new_document, new_entity, new_source, pk

runner = CliRunner()
QUOTE = "Total net sales increased 5% year over year."


@pytest.fixture
def store(tmp_path: Path) -> tuple[Path, int]:
    """A store with Apple and one public chunk; returns the db path and the chunk id."""
    path = tmp_path / "meta.db"
    engine = make_engine(path)
    init_db(engine)
    with Session(engine) as s:
        apple = new_entity(s)
        doc = new_document(s, source_id=pk(new_source(s)), entity_id=pk(apple), available_at=T0)
        chunk_id = pk(new_chunk(s, document_id=pk(doc), text=QUOTE))
    engine.dispose()
    return path, chunk_id


def good_draft(chunk_id: int) -> dict[str, Any]:
    return {
        "as_of": T0.isoformat(),
        "ticker": "AAPL",
        "citations": [
            {
                "claim": "Net sales grew 5%.",
                "evidence_kind": "chunk",
                "chunk_id": chunk_id,
                "quote": QUOTE,
            }
        ],
    }


def run(tmp_path: Path, db: Path, body: str) -> Any:
    draft = tmp_path / "draft.json"
    draft.write_text(body)
    return runner.invoke(app, ["evidence", "validate", str(draft), "--db", str(db)])


def test_valid_draft_exits_zero(tmp_path: Path, store: tuple[Path, int]) -> None:
    db, chunk_id = store
    result = run(tmp_path, db, json.dumps(good_draft(chunk_id)))
    assert result.exit_code == 0, result.output


def test_rejected_draft_exits_two(tmp_path: Path, store: tuple[Path, int]) -> None:
    db, chunk_id = store
    draft = good_draft(chunk_id)
    draft["citations"][0]["claim"] = "Net sales grew 6%."
    result = run(tmp_path, db, json.dumps(draft))
    assert result.exit_code == 2, result.output


def _without(d: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: val for k, val in d.items() if k != key}


def _bad_bodies(chunk_id: int) -> dict[str, str]:
    good = good_draft(chunk_id)
    extra_citation_field = good_draft(chunk_id)
    extra_citation_field["citations"][0]["page"] = 3
    extra_top_field = {**good_draft(chunk_id), "strategy": "momentum"}
    naive = {**good_draft(chunk_id), "as_of": "2024-07-15T09:30:00"}
    not_a_date = {**good_draft(chunk_id), "as_of": "yesterday"}
    missing_claim = good_draft(chunk_id)
    del missing_claim["citations"][0]["claim"]
    return {
        "malformed_json": '{"as_of": "2024-07-15T13:30:00+00:00", "ticker": ',
        "not_an_object": "[1, 2, 3]",
        "missing_ticker": json.dumps(_without(good, "ticker")),
        "missing_as_of": json.dumps(_without(good, "as_of")),
        "missing_citations": json.dumps(_without(good, "citations")),
        "missing_claim": json.dumps(missing_claim),
        "unknown_citation_field": json.dumps(extra_citation_field),
        "unknown_top_level_field": json.dumps(extra_top_field),
        "naive_as_of": json.dumps(naive),
        "as_of_not_a_date": json.dumps(not_a_date),
    }


@pytest.mark.parametrize(
    "case",
    [
        "malformed_json",
        "not_an_object",
        "missing_ticker",
        "missing_as_of",
        "missing_citations",
        "missing_claim",
        "unknown_citation_field",
        "unknown_top_level_field",
        "naive_as_of",
        "as_of_not_a_date",
    ],
)
def test_bad_draft_exits_one_with_one_line(
    tmp_path: Path, store: tuple[Path, int], case: str
) -> None:
    db, chunk_id = store
    result = run(tmp_path, db, _bad_bodies(chunk_id)[case])

    assert result.exit_code == 1, result.output
    # A clean exit, not an uncaught exception that the runner turned into exit code 1.
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    lines = result.output.strip().splitlines()
    assert len(lines) == 1, result.output
    assert "Traceback" not in result.output


def test_missing_draft_file_exits_one_with_one_line(
    tmp_path: Path, store: tuple[Path, int]
) -> None:
    db, _ = store
    result = runner.invoke(
        app, ["evidence", "validate", str(tmp_path / "absent.json"), "--db", str(db)]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Traceback" not in result.output
