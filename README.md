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
- [x] Entity table: companies, funds and coins under one key; a reviewed watchlist of
      about 300 instruments resolved against the SEC ticker tables
- [x] Market-wide daily bars: every exchange-listed symbol since 2020, including delisted
      ones, stored as Parquet with the instant each bar became knowable
- [x] Security identity: corporate actions (splits, renames, mergers) from Alpaca, a
      security master that maps each symbol-day to a permanent security across renames and
      ticker reuse, and an instrument class (common, ETF, leveraged ETF, warrant, unit...)
      for every segment
- [x] Liquid tier: the securities that averaged more than $50M a day over the previous 20
      sessions, computed as of any instant from data visible then; about 1,100-2,000 a month,
      3,964 ever between 2020 and 2026
- [ ] Historical backfill: minute bars, SEC EDGAR filings and XBRL facts, FRED
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

Load the watchlist into the entity table:

```bash
export ASOF_SEC_USER_AGENT="your-project you@example.com"   # the SEC requires a contact
uv run asof entities sync --download    # fetch the SEC ticker tables, then sync
uv run asof entities sync               # later runs reuse the saved tables
```

Each ticker in `configs/watchlist.yaml` is matched to its SEC identifier. The command writes
every entity or none: it stops if a company is not listed or a ticker belongs to more than
one registrant, and it never moves a stored instrument to a different registrant.

Backfill market data from Alpaca (a free paper-trading account is enough for history):

```bash
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
uv run asof market universe              # Alpaca asset list + historical SEC ticker tables
uv run asof market backfill              # daily bars since 2020 into $ASOF_DATA_DIR/market
uv run asof market status                # rows, symbols and date range per year
uv run asof market actions               # corporate actions: splits, name changes, mergers...
uv run asof market identity              # security master: which security each symbol was, per day
uv run asof market instruments           # instrument class of every security-master segment
uv run asof market liquid --as-of 2021-03-01            # the liquid tier that morning
uv run asof market liquid --from 2020-01-01 --to 2026-10-01   # one tier per month start, and the union
```

The universe is the union of every symbol Alpaca lists and every ticker that appeared in an
SEC ticker table since 2019 (archived copies), so delisted companies are included. Bars are
stored unadjusted, one Parquet file per year and first letter, each row stamped with
`available_at`: the moment the bar was final (20:00 New York for a daily bar, since its volume
includes after-hours trades). `visible_bars(con, root, timeframe, as_of)` is the only read path
and returns nothing that was not knowable at `as_of`. Reruns skip files fetched after their
period ended and refetch the rest, so the command is safe to run daily.

Bars are stored exactly as the vendor returns them and never dropped at write time. Which
security a row belongs to is a separate question, answered at read time by the security
master: Alpaca files a security's whole history under its latest symbol (BK's 2020 bars also
appear under BNY, its name since 2026), tickers get reused by unrelated companies (BBBY), and
the SEC's own ticker table lags renames by months. `asof market identity` cuts every symbol's
timeline into segments from Alpaca's name-change records and archived SEC ticker tables,
links segments of the same security under one permanent id, checks the links against
same-day identical OHLCV, and lists the cases it could not settle for a human to decide in
`configs/security_overrides.yaml`. `asof market instruments` then tags each segment with an
instrument class from the name it carried in that era, using the ordered rules in
`configs/instrument_class_rules.yaml`; the scanner will only look at common stocks and
unleveraged ETFs. Names the rules cannot judge (an issuer whose SEC entry carries dozens of
notes, such as the ETN shelves of Credit Suisse or Bank of Montreal) stay `unknown` until a
human records the class in `configs/instrument_class_overrides.yaml`.

The scanner does not look at all 20,000 symbols. `asof market liquid` computes the liquid
tier as of an instant: securities whose dollar volume over the previous 20 sessions averaged
more than $50M a day, common stocks and unleveraged ETFs only, with no price floor. Everything
it uses was visible at that instant: the session calendar, the bars, and the security-master
segments that assign bars to securities (a rename published four days after it took effect is
not applied until then). The divisor is the number of sessions, so a listing in its first week
or a one-day SPAC spike cannot buy its way in. Recomputed at every month start, the tier is the
scope the identity audit must settle and the universe the minute-bar backfill pulls; a symbol
that qualifies while still `unknown` makes the command exit non-zero so the override file gets
a line.

## Layout

Directories marked *(planned)* do not exist yet.

```
src/asof/
  ingest/     pull raw data from Alpaca, SEC EDGAR, FRED; watchlist, SEC ticker tables,
              corporate actions, security master and instrument classes
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
configs/             watchlist.yaml (instruments that get filings and fundamentals),
                     security_overrides.yaml (human identity decisions),
                     instrument_class_rules.yaml (ordered class rules),
                     instrument_class_overrides.yaml (human class decisions)
docs/                public documentation                                  (planned)
```

## License

[MIT](LICENSE)
