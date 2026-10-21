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
      ones and both sides of every rename, stored as Parquet with the instant each bar became
      knowable; a refetch adds to the store and never removes what an earlier fetch saw; a
      stretch the vendor has lost is imported from a second source, kept in its own directory
      and registered with a reason
- [x] Security identity: corporate actions (splits, renames, mergers) from Alpaca, a
      security master that maps each symbol-day to a permanent security across renames and
      ticker reuse, and an instrument class (common, ETF, leveraged ETF, warrant, unit,
      when-issued...) for every segment
- [x] Liquid tier: the securities that averaged more than $50M a day over the previous 20
      sessions, computed as of any instant from data visible then; about 1,100-2,000 a month,
      3,972 ever between 2020 and 2026
- [x] Resolved read path: every bar read carries the permanent id of the security that owned
      its symbol that day, as the security master was known at the time; relabelled copies
      and the vendor's volume-0 filler bars are gone by default
- [x] Data audit: 33 checks over the real store with reviewed exceptions, clean in the
      liquid tier, and hand-checked reference histories asserted as tests against it
- [x] Truncation test: the security master rebuilt from only the inputs knowable at an
      instant and compared with what the full master shows then; every identity fact (a
      segment, its end, its CIK) carries the instant it became knowable; gated on what each
      leak changes in the price rows and CIKs a backtest reads
- [ ] Historical backfill: minute bars, SEC EDGAR filings and XBRL facts, FRED
- [ ] Scanner, evidence retrieval, judge, executor, attribution
- [ ] Strategy evolution: replay, permutation gate, approval

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned via
`.python-version` and installed automatically.

```bash
uv sync                 # create .venv and install all dependencies
uv run asof --help      # CLI entry point
uv run pytest           # tests (the real-data cases skip when /data/asof is absent)
uv run pytest -m realdata   # only the hand-checked cases against the real store
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
uv run asof market backfill --symbols-file stale.txt --force   # top up a list of symbols
uv run asof market import-bars           # registered second-source bars (configs/bar_imports.yaml)
uv run asof market status                # rows, symbols and date range per year
uv run asof market actions               # corporate actions: splits, name changes, mergers...
uv run asof market identity              # security master: which security each symbol was, per day
uv run asof market instruments           # instrument class of every security-master segment
uv run asof market liquid --as-of 2021-03-01            # the liquid tier that morning
uv run asof market liquid --from 2020-01-01 --to 2026-10-01   # one tier per month start, and the union
uv run asof market audit                 # count every kind of data error found so far
uv run asof market truncation            # rebuild the master as of past instants; gate on impact
uv run asof market truncation --jobs 2   # 2 processes (default: 4 or fewer cores; 4 take 23 GB)
```

The universe is the union of every symbol Alpaca lists, every ticker that appeared in an SEC
ticker table since 2019 (archived copies) and both names of every rename Alpaca recorded, so
delisted companies and the old names of renamed ETFs are included. A symbol Alpaca still lists
is requested for every period; the SEC tables only bound the requests for symbols it no longer
lists (the SEC company table carries ETFs erratically, and a window closed by such a gap once
left 197 listed symbols without years of bars). Bars are
stored unadjusted, one Parquet file per year and first letter, each row stamped with
`available_at`: the moment the bar was final (20:00 New York for a daily bar, since its volume
includes after-hours trades). `visible_bars(con, root, timeframe, as_of)` is the only read path
and returns nothing that was not knowable at `as_of`. Reruns skip files fetched after their
period ended and refetch the rest, so the command is safe to run daily. A refetch only adds
rows: the vendor deletes a delisted security's history once its symbol is reused and files the
new holder's bars under it (Voxeljet's VJET became copies of RMED's history in 2026), so a bar
once stored is kept as first seen, and a partition is discarded by deleting its file.

Bars are stored exactly as the vendor returns them and never dropped at write time. Which
security a row belongs to is a separate question, answered at read time by the security
master: Alpaca files a security's whole history under its latest symbol (BK's 2020 bars also
appear under BNY, its name since 2026), tickers get reused by unrelated companies (BBBY), and
the SEC's own ticker table lags renames by months. `asof market identity` cuts every symbol's
timeline into segments from Alpaca's name-change records and archived SEC ticker tables,
links segments of the same security under one permanent id, checks the links against
same-day identical OHLCV, and lists the cases it could not settle for a human to decide in
`configs/security_overrides.yaml`. The vendor's records are read against the bars rather
than taken at face value: a rename filed weeks after the new name started trading is dated
from the hand-over visible in the bars, a record whose old name never traded here (a
temporary OTC suffix, a rename older than the store) is no event, a label that goes quiet
and comes back after a gap is a new security from its first bar, and when-issued symbols
(`.WI`, Nasdaq's fifth-letter V) are never the security itself, so a rename routed through
one joins the names on either side. The same bars settle what the records leave open: two
names that traded side by side with different prices right up to a rename record did not
rename, the old one converted into the new one's existing shares (Carnival plc's ADR into
Carnival Corp) and the two stay separate securities; a name whose bars become identical to
another's and then stop while the other trades on was renamed into it even though nobody
filed it (Cyclerion into Kalaris); and a segment whose trailing bars, after a gap or after
its filer left the SEC table, are copies of another security's (Axonics after its buyout,
Voxeljet after its delisting) ends where the copies begin, as does a label whose whole history
under an earlier holder is the current holder's history rewritten. When a label is taken at a
continuous price by a different filer, the bars decide: if the taker traded nothing of its own
while the earlier holder's segment ran, that segment is the taker's history rewritten under
the label (Barrick's 2020-2025 prices sit under B, the ticker Barnes Group held until Barrick
took it in 2025) and joins the taker, knowable only from the takeover record, so a reader in
2023 sees no segment for those bars rather than Barrick under a ticker it did not yet have;
otherwise the two stay separate securities. A filer that renamed off a label takes its CIK
along, however long the SEC table keeps listing it there (Xperi Holding became Adeia the day
the spun-off Xperi Inc. took XPER). A segment takes a CIK from snapshots before it only
while the label is still listed in the last snapshot before it: ABAT trading again in 2023 is
American Battery Technology, not Advanced Battery Technologies, last listed under ABAT in
2020. When the SEC replaced its table on 2020-07-10, rows that vanished for good on a ticker
that kept trading across the change are dropped from the old table as well (the ICLN ETF had
been listed under InterCallNet); a ticker that only shows flat zero-volume bars until a new
company lists under it keeps them, so the two stay apart. Every conflict the build reports carries a `resolution`
(`new_security`, `bridged`, `ignored`, `relabelled`, `unresolved`); the unresolved ones wait
for a human, and so does each relabelled merge, once, since a coincidence of prices could
produce the same absence of bars. `asof market instruments` then tags each segment with an
instrument class from the name it carried in that era, using the ordered rules in
`configs/instrument_class_rules.yaml`; the scanner will only look at common stocks and
unleveraged ETFs. A segment nobody ever named (an ETF's old ticker) inherits the class of a
later segment of the same security. Names the rules cannot judge (an issuer whose SEC entry
carries dozens of notes, such as the ETN shelves of Credit Suisse or Bank of Montreal) stay
`unknown` until a human records the class in `configs/instrument_class_overrides.yaml`.

Reading applies all of this by default. `visible_bars` matches each bar to the segment that
owned its symbol on that session date and returns it with the segment's `security_id`; a bar
owned by no segment (BNY's copy of BK's 2022 history) is not returned, and the two names of a
renamed security come back as one timeline. The master itself is read as of the same instant,
and a segment's end is a fact with its own date: the file says BK ends on 2026-05-21, but a
reader in 2022 sees BK as an open segment, and a reader between a rename's effective day and
the day the vendor filed it keeps attributing the old symbol's bars to the old segment. Each
bar also carries `traded`; the vendor keeps writing a volume-0 bar a day under an old symbol
for years after a rename (1.18M such bars, 6.5% of the store), and these are dropped unless
asked for. `split_day` marks the ex-date bar of a split known at the time, so a return across
it is not mistaken for a move. `resolve=False` returns the raw store for audits.

The vendor files a symbol's whole history under its current holder, so when a ticker is reused
the old holder's history can disappear: after Banzai took PARA in 2026, Paramount Global's
class B bars from the 2022 rename to the 2025 merger were gone. Such a stretch is imported from
a second source (Yahoo's chart API, which keeps the lineage under the company's newest symbol)
into `$ASOF_DATA_DIR/market/bars_1d_ext/`, with `source` and `source_symbol` columns, and
registered in `configs/bar_imports.yaml` with the range, a reason and a date; the Alpaca files
are not touched. On read an imported row replaces the vendor's row of the same symbol and
instant, so the entry is the recorded decision that the vendor's rows there are wrong, and
every bar read says where it came from (`source`). The audit counts the imported rows and how
many of them override a vendor row.

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

Tests on fake data only show that the code does what its author imagined; they stay green when
the imagined vendor differs from the real one. `asof market audit` therefore reads the real
store as of an instant and counts each kind of accident met so far, 33 checks in all:
duplicate or impossible bars, segments of one security that overlap, two securities sharing
an identical bar (a relabelled history the master did not catch), an earlier holder's segment
partly made of the current holder's bars, a split record with no price
move, a one-day thin-volume print far from the previous close with no split to explain it, a
price jump across a rename, traded bars that no segment owns, a listed symbol whose bars
stopped although the vendor still serves it, and a company the SEC lists on an exchange that
owns no bar at all (the vendor dropped Inhibrx's history when its ticker was reused; read from
the SEC exchange tables archived since 2021-07, so earlier losses stay invisible). A split
record counts against the liquid tier only when its security was in the tier within three
months of the ex-date, its ex-date bar is looked up under every name of the security, and a
move that lands one or two sessions late is the move. Each finding is a row with a
short key (`AAA:2020-01-07`, `S000730`, `AAA/BBB`) written to
`$ASOF_DATA_DIR/market/audit/<check>.parquet`. Gated checks must be clean inside the liquid
tier, the rest are reported for a human to read. A reviewed finding goes into
`configs/audit_exceptions.yaml` with a reason and a date, and an exception that no longer
matches anything is reported as stale. The command exits 2 while a gated check has rows
outside the exceptions. Next to it, `tests/store/test_golden_cases.py` asserts about twenty
hand-checked histories through the same read path: SIVB's last day, BBBY's two companies,
BK in 2022, NKLA appearing only once its rename was filed, HCTI's three reverse splits
compounding to TradingView's price.

Several identity rules read data after the event they judge: whether a rename was really a
conversion depends on SEC snapshots up to six months later, whether a takeover was a
relabelled copy depends on the bars on both sides. A segment stamped knowable at the event
may carry what they found. `asof market truncation` rebuilds the master from only the bars,
snapshots, rename records and dated human decisions knowable at an instant and compares it
with the full master as a reader at that instant sees it. A segment, field or grouping the
full master shows but the rebuild cannot produce is a **leak**; something the rebuild guesses
on days the full master leaves open, or a CIK it still withholds, is a **delay**. The first run
found that a segment's CIK came from SEC snapshots months after it started trading (AAAU
trades from 2020-01-02, the SEC lists it from 2020-07-10), so the CIK now has its own
`cik_available_at` and stays hidden until then. The second found segments stamped knowable
at 00:00 of the event that created them although the build needs their own first bar, which
is knowable only at 20:00 (CXAIV -> CXAI on 2023-03-15): a segment is now never knowable
before its first bar, and a takeover's relabelled history not before the taker's. The third
found share classes whose CIK came through a spelling the SEC only adopted later (`BRKB`
until 2020-06-01, `BRK.B` from 2020-07-10): such a CIK is now knowable from the first
snapshot that writes the label's own spelling. Matching spellings by their letters alone
would link stale rows to other companies (the SEC's `LGCU` is Luna Gold, the store's `LGC.U`
Legacy Acquisition). The fourth was a bug in the build itself: pieces were numbered before
zero-bar pieces were filtered out and not renumbered after, so the 120 pieces appended next
shared their numbers with surviving ones (YEXT, YFFI, ...); one unrelated rename record could
then swap segments of other securities, and YPF, YQ and YOU went missing from some rebuilds.
The
default instants are fixed in `configs/truncation_instants.txt`: computed ones move whenever
the master changes, and a leak seen only at an instant that moved away would look fixed.
The store is read once and the 66 rebuilds run in forked worker processes (about 12
minutes with four, impact included, against 40 serially, with identical reports; six ran a
31 GB machine out of memory). The same comparison runs in the test suite on every hand-built
identity scenario at every instant at which its master or its inputs change, against a
baseline of its own.

Most leaks change nothing a backtest reads, so the real-data gate counts what they change.
Each rebuild is also compared on the only two things a backtest takes from the master, for
the securities ever in the liquid tier: the daily price rows each security shows on each day
(open, high, low, close, volume and the split recorded for the bar's symbol, whatever symbol
the row came through) and the CIK each symbol shows on each day. When II-VI took the ticker
COHR in 2022, the two masters group the segments differently but read the same prices for
II-VI, so that leak changes nothing. B's 2020-2025 bars are Barrick's prices (closes of $13-30,
moved there when Barrick took the ticker from Barnes Group in 2025): the full master files them
under Barrick's CIK, while a rebuild in mid-2025, not yet knowing that A-Mark's later move to
GOLD overwrote Barrick's old ticker, still files them under Barnes Group's, which differs on
1,345 days of filings. Every
differing (security, day) price cell and (symbol, day) CIK cell is attributed to the leak
keys of the same instant whose segments its rows name; `configs/truncation_impact_baseline.txt`
holds the cells each known key covers summed over the fixed instants (`KEY price=N cik=M`, so
one more wrong cell at any instant raises it; 44 of the real store's 248 leak keys changed
anything, 36 after the spelling fix and 29 after the key fix below) and the fingerprint of that instant list. The command exits 2 on a key with impact
not in the baseline, a number above it, or a cell no key names; on the default instants also
on a listed key without impact (`--prune-baseline` deletes those and lowers numbers that
shrank; lines are only added by hand). A run on instants outside the list judges only new
keys and unmatched cells; once the list itself changes, every default run fails until
`--rebase-baseline` re-sums the numbers over the new list, printing each change. Diffs go to
`$ASOF_DATA_DIR/market/audit/truncation/<instant>.parquet`, impact rows to
`$ASOF_DATA_DIR/market/audit/truncation_impact/<instant>.parquet`.

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
                     instrument_class_overrides.yaml (human class decisions),
                     audit_exceptions.yaml (reviewed audit findings),
                     bar_imports.yaml (bars imported from a second source),
                     truncation_instants.txt (instants the truncation test rebuilds at),
                     truncation_impact_baseline.txt (known leak impact, a ratchet)
docs/                public documentation                                  (planned)
```

## License

[MIT](LICENSE)
