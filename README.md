# asof

A point-in-time correct market research agent.

Every day it scans the market, gathers evidence, forms conclusions that cite
their sources, places simulated orders, keeps a ledger, attributes results,
and proposes changes to its own strategies. A proposed change must pass a
historical replay, a permutation-null significance gate corrected for the
number of attempts, and human approval before it takes effect.

The core invariant: **every read of data is bound to an `as_of` timestamp and
may only see information that was available at that moment.** Filtering uses
when a document actually became available (`available_at`), not when the
underlying event happened, and revisions are appended rather than overwritten.
Any violation is look-ahead leakage and is treated as a bug.

## Status

- [x] Project scaffold: uv-managed Python 3.12, package layout, CLI
- [x] CI: lint, format, type check and tests on every push and pull request
- [x] Metadata store: SQLite schema with integrity constraints; all reads go
      through `visible_*(session, as_of)`
- [x] Instrument table: companies, funds and coins under one key; a reviewed watchlist of
      about 300 instruments resolved against the SEC ticker tables
- [ ] Historical backfill: market-wide daily and minute bars, SEC EDGAR filings and XBRL
      facts, FRED
- [ ] Scanner, evidence retrieval, judge, executor, attribution
- [ ] Strategy evolution: replay, permutation gate, approval

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

Create the metadata database:

```bash
uv run asof db init                     # $ASOF_DATA_DIR/meta.db, default /data/asof/meta.db
uv run asof db init --path ./meta.db    # any other location
```

The command is safe to run more than once. If `/data` is not writable on your
machine, set `ASOF_DATA_DIR` or pass `--path`.

Load the watchlist into the instrument table:

```bash
export ASOF_SEC_USER_AGENT="your-project you@example.com"   # the SEC requires a contact
uv run asof entities sync --download    # fetch the SEC ticker tables, then sync
uv run asof entities sync               # later runs reuse the saved tables
```

Each ticker in `configs/watchlist.yaml` is matched to its SEC identifier. The command writes
every instrument or none: it stops if a company is not listed or a ticker belongs to more than
one registrant, and it never moves a stored instrument to a different registrant.

## Layout

Directories marked *(planned)* do not exist yet.

```
src/asof/
  ingest/     pull raw data from Alpaca, SEC EDGAR, FRED; watchlist and SEC ticker tables
  store/      SQLite + SQLModel metadata and the as_of read path
  features/   shared feature registry: technical, fundamental, macro, event   (planned)
  scan/       pure-code daily scanner
  evidence/   as-of document retrieval and chunking
  judge/      cited conclusions with citation validation
  execute/    pure-code paper-trading executor and ledger
  attribute/  P&L attribution
  evolve/     strategy proposals, replay, permutation gate, approval
  api/        FastAPI surface
strategies/<name>/   one strategy per directory (scan, rules, judge, execute,
                     attribution configs), versioned as a whole            (planned)
tests/               mirrors src/asof
configs/             watchlist.yaml: instruments that get filings and fundamentals
docs/                public documentation                                  (planned)
```

## License

[MIT](LICENSE)
