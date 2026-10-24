# asof

A point-in-time correct market research agent: it may only read what was public at the
time, and every conclusion must quote it.

**[Live demo](https://zhenglin17.github.io/asof/)**: replays you can tune, every cited
judgement, and the guards against seeing the future.

[![The demo page](docs/demo.png)](https://zhenglin17.github.io/asof/)

Replayed evening by evening, it scans for post-earnings gaps, a language model reads the
filing behind each one and answers take or pass, and code checks that every claim quotes,
word for word, a passage that was public at that instant. A simulated executor trades the
takes, a ledger records every leg, and a report compares the result with a judge that takes
everything and with SPY.

The core invariant: **every read of data is bound to an `as_of` instant and may only see what
was available then.** Filings are filtered on when EDGAR made them public (`available_at`),
not on the period they describe; revisions are appended, never overwritten. Any violation is
look-ahead leakage and is treated as a bug.

The full manual (every command, what each stage stores, the source layout) is
[docs/commands.md](docs/commands.md).

## How it works

```mermaid
flowchart TB
    subgraph sources["Sources"]
        EDGAR[SEC EDGAR<br/>8-K, 10-Q, 10-K]
        ALP[Alpaca<br/>daily bars, corporate actions]
    end
    subgraph store["Point-in-time store"]
        DOCS[(filings and chunks<br/>stamped available_at)]
        BARS[(bars + security master<br/>stamped available_at)]
        VIS{{"visible_*(as_of)"}}
    end
    subgraph loop["Fast loop, deterministic, at 20:00 New York"]
        direction LR
        SCAN[Scanner<br/>pure code]
        JUDGE[Judge<br/>LLM fills a fixed schema]
        CV[Citation validator]
        EXE[Executor<br/>pure code]
        LED[(ledger, attribution)]
    end
    REP[Replay report<br/>vs take-all, vs SPY]
    EVO[Evolution, planned<br/>proposal, replay, gate, approval]

    EDGAR --> DOCS
    ALP --> BARS
    DOCS --> VIS
    BARS --> VIS
    SCAN --> JUDGE --> CV
    CV -- "fails: sent back once, then no-call" --> JUDGE
    CV -- "every claim quotes a visible passage" --> EXE --> LED
    VIS --> loop --> REP
    REP -.-> EVO -.-> loop
```

- **Store.** About 300 watchlist instruments, daily bars for every listed US symbol since 2020
  (delisted ones included), and 11,522 filings of 218 companies in 703,614 citable chunks. A
  security master maps each symbol-day to a permanent security across renames and ticker
  reuse. Every row carries the instant it became knowable; all reads go through
  `visible_*(session, as_of)`. [Details](docs/commands.md#commands-at-a-glance)
- **Scanner.** `earnings_gap`: the first session after an earnings release gapped up at least
  4% on at least twice its 20-day volume. [Details](docs/commands.md#replay)
- **Judge.** Either take-all (the baseline) or an LLM that sees the reaction-day numbers and
  the triggering filing as public that evening, and answers take or pass with a conviction
  and one to eight claims, each quoting a chunk. Code, not the model, decides what happens
  next. Answers are cached by request, so a replay runs offline and repeats exactly (the cache
  holds the filings and is not part of the repository).
  [Details](docs/commands.md#llm-judge)
- **Citation validator.** A claim must quote, word for word, a chunk of the right company
  that was public at that instant. [Details](docs/commands.md#evidence)
- **Executor.** Next-open entry, ATR zone stops with a trailing key and one re-entry, sizing
  from a fixed risk budget with position and exposure caps. Each closed leg is a ledger row.
- **Evolution** (planned). Every parameter set a replay runs under is registered, so the
  number of attempts is on record for the significance gate a change will have to pass.

## Results so far

One replay of the LLM judge, 2026-07-01 to 2026-10-16 (where the data ended), after the
model's training cutoff (2026-06-30), default settings, next to take-all over the same
candidates. No significance test has been run, and none of the differences below is
distinguishable from noise.

|                         | LLM judge | Take-all | SPY (price only) |
|-------------------------|----------:|---------:|-----------------:|
| Candidates              | 51: 32 take, 15 pass, 4 no-call | 51 | |
| Traded decisions        | 31        | 40       |                  |
| Winners                 | 12 of 31  | 13 of 40 |                  |
| Return, closed trades   | -1.04%    | -0.73%   | +2.45%           |
| Max drawdown, closed    | 3.0%      | 4.9%     |                  |
| Mean R per traded decision | -0.07  | -0.10    |                  |

- The LLM's lower drawdown mostly reflects fewer trades and less money at risk, not better
  selection; the win rates differ by one winning trade, and the mean R gap comes from four
  trades take-all had no capital left for.
- Take-all did not trade every candidate either: the exposure caps skipped some, and a pass
  frees room for another trade, so the comparison mixes judgement with capacity.
- The 11 passes that take-all traded earned it +0.23R on average, but -0.24R without the
  single best one (NOW, +4.95R).
- On the demo page's grid of 36 settings, LLM minus take-all changes sign with the stop and
  risk settings. The minimum-conviction 0.6 cells look best, but they are the same 5 trades.

**51 candidates in one market regime cannot tell whether the judge helps**; what this shows is
the pipeline running end to end with every decision traceable to the passage it rests on.

Take-all over 2023-01-03 to 2026-10-16: 643 candidates, 547 traded, **+62.4% against SPY's
+98.6%**, max drawdown 10.4% on closed trades, and the ten best trades made 77% of the profit.
The strategy trailed buy-and-hold, and even that figure is an upper bound: the watchlist was
picked with hindsight and SPY doubled over the period. SPY is fully invested while the
strategy holds cash most of the time, so the returns are not risk-matched.

The grid ran on throwaway copies of the database, so it adds nothing to the recorded attempt
count. It is still a search to anyone who picks its best cell, and is shown as such.

## Leakage guards and the tests behind them

| Way to see the future | Guard | Tests |
|---|---|---|
| Reading a filing before it was public | Filter on `available_at`; revisions are new rows | `tests/store/test_queries.py`, `test_documents.py` |
| Using today's bar during today | A daily bar is visible only after the post-market close | `tests/store/test_market.py`, `tests/replay/test_replay.py` |
| Knowing today what a ticker meant then | Security master as known at each instant; a truncation test rebuilds it from only the inputs knowable then and compares | `tests/ingest/test_master_truncation.py`, `test_truncation_gate.py` |
| Future statistics in search | Ranking uses term statistics of the visible documents only | `tests/evidence/test_retrieve.py`, `tests/store/test_chunk_search.py` |
| Citing a passage that was not public yet | The validator checks the instant; the prompt holds only visible chunks | `tests/judge/test_llm.py`, `tests/judge/test_window.py` |
| A model that already knows the outcome | Results split at the training cutoff; the prompt names the ticker, not today's company name | `tests/replay/test_compare.py`, `tests/judge/test_llm.py` |
| Outcomes visible before they happened | A trade is visible from its exit-day close, an outcome after its re-entry window | `tests/attribute/test_trades.py`, `tests/replay/test_replay.py` |
| A result that depends on later days | Replaying to day N gives the same decisions as replaying further | `tests/replay/test_replay.py` |
| Trying settings until one works | Every parameter set a research replay runs under is registered and counted | `tests/strategy/test_config.py`, `tests/replay/test_report.py` |

On the real store, `asof market audit` runs 33 data checks with reviewed exceptions, and
hand-checked reference histories (renames, ticker reuse, splits) are asserted as tests.

## Known limitations

- **One short window.** The LLM judge has been replayed over 51 candidates in about four
  months; no significance test has been run.
- **Hindsight universe.** The watchlist was picked today, so companies that disappeared are
  under-represented.
- **Daily bars, simple fills.** Within a day the worst order of high and low is assumed;
  fills pay a flat 5 bps of slippage and no commission.
- **Closed trades only.** Return and drawdown move when a trade closes; open positions are not
  marked to market, which understates drawdown.
- **Price-only benchmark.** SPY without dividends (about 1-1.5% a year), and no interest on the
  strategy's idle cash.
- **Vendor history.** A few identity facts the data vendor rewrote after the fact cannot be
  dated; they are listed in a baseline that may only shrink.
- **One model, one prompt.** Answers are cached; another model or prompt may decide
  differently. A training cutoff hides outcomes, not what the model knew about each company
  before it.
- **Long only, gap-ups only.** Never tested in a falling or very volatile market.
- **Not built yet:** minute bars, macro data, the evolution loop and its significance gate.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is installed automatically.

```bash
git clone https://github.com/Zhenglin17/asof && cd asof
uv sync
uv run pytest                          # cases that need the real data store skip without it
python3 -m http.server -d web 8000      # the demo page, from the committed data
```

Building the store yourself needs free Alpaca keys and an SEC contact string, and takes hours:

```bash
export ASOF_DATA_DIR=~/asof-data      # default /data/asof
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=... ASOF_SEC_USER_AGENT="name you@example.com"
uv run asof db init && uv run asof entities sync --download
uv run asof market universe && uv run asof market backfill && uv run asof market import-bars
uv run asof market actions && uv run asof market identity && uv run asof market instruments
uv run asof market audit
uv run asof docs sync && uv run asof docs audit && uv run asof evidence index
uv run asof replay run --strategy earnings_gap --start 2023-01-03 --end 2026-10-21
uv run asof replay report --run 1
```

The full pipeline, in order, is in [Commands at a glance](docs/commands.md#commands-at-a-glance).

## Status

- [x] Point-in-time store: bars, security master, filings and chunks, all read through `as_of`
- [x] Data audit and truncation test over the real store
- [x] Evidence search and citation validator
- [x] Fast loop: scanner, take-all and LLM judges, executor, ledger, attribution, replay report
- [x] Demo page on GitHub Pages
- [ ] Minute bars, XBRL facts, FRED
- [ ] Strategy evolution: permutation gate, approval

Feature by feature: [Status in detail](docs/commands.md#status-in-detail).

## License

[MIT](LICENSE)
