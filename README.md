# asof

A point-in-time correct market research agent.

Every day it scans the market, gathers evidence, forms conclusions that cite
their sources, places simulated orders, keeps a ledger, attributes results,
and proposes changes to its own rulebook. Rule changes must pass a historical
replay, a permutation-null significance gate, and human approval before they
take effect.

The core invariant: **every read of data is bound to an `as_of` timestamp and
may only see information that was available at that moment.** Any violation is
look-ahead leakage and is treated as a bug.

## Status

Early scaffolding. No pipeline stages are implemented yet.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned via
`.python-version` and installed automatically.

```bash
uv sync                 # create .venv and install all dependencies
uv run asof --help      # CLI entry point
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format --check .
uv run pyright          # type check
```

## Layout

```
src/asof/
  ingest/     pull raw data from Alpaca, SEC EDGAR, FRED
  store/      SQLite + SQLModel metadata, Parquet + DuckDB bars
  scan/       pure-code daily scanner
  evidence/   as-of document retrieval and chunking
  judge/      cited conclusions with citation validation
  execute/    pure-code paper-trading executor and ledger
  attribute/  P&L attribution
  evolve/     rule proposals, replay, permutation gate, approval
  api/        FastAPI surface
rules/        versioned rulebook (YAML)
tests/        mirrors src/asof
configs/
docs/         public documentation
```

## License

MIT
