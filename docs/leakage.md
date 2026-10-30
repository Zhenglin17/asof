# Leakage guards and the tests behind them

Each way a backtest can see the future, the mechanism that stops it, and the tests that fail if
it comes back.

| Way to see the future | Guard | Tests |
|---|---|---|
| Reading a filing before it was public | Filter on `available_at`; revisions are new rows | `tests/store/test_queries.py`, `test_documents.py` |
| Using today's bar during today | A daily bar is visible only after the post-market close | `tests/store/test_market.py`, `tests/replay/test_replay.py` |
| Knowing today what a ticker meant then | Security master as known at each instant; a truncation test rebuilds it from only the inputs knowable then and compares | `tests/ingest/test_master_truncation.py`, `test_truncation_gate.py` |
| Future statistics in search | Ranking uses term statistics of the visible documents only | `tests/evidence/test_retrieve.py`, `tests/store/test_chunk_search.py` |
| Citing a passage that was not public yet | The validator checks the instant; the prompt holds only visible chunks | `tests/judge/test_llm.py`, `tests/judge/test_window.py` |
| A model that already knows the outcome | Results split at the training cutoff; the prompt names the ticker, not today's company name | `tests/replay/test_compare.py`, `tests/judge/test_llm.py` |
| Outcomes visible before they happened | A trade is visible from its exit-day close, an outcome after its re-entry window | `tests/attribute/test_trades.py`, `tests/replay/test_replay.py` |
| Rescaling history with a price not yet known | A spin-off's factor needs the child's first close: it is used only from the day every price it rests on is visible, and sessions in between have no value | `tests/store/test_spinoffs.py`, `tests/features/test_bx.py` |
| Moving a stop with a spin-off price not yet known | A position held into a spin-off's ex-date is frozen from that open (the record is on file by then) until the factor's close; its children are sold at their next open, and an event still unpriced after `spinoff_max_wait` sessions resumes on raw prices | `tests/execute/test_simulate_spinoffs.py`, `tests/replay/test_replay_spinoffs.py` |
| A result that depends on later days | Replaying to day N gives the same decisions as replaying further | `tests/replay/test_replay.py` |
| A market filter read from tomorrow's index bar | The SPY trend filter reads only the index closes final at that night's close, adjusted by the splits on file then | `tests/scan/test_bx_filters.py` |
| One pass over a window reading the window's end | The B-Xtrender scan computes each series once at the window's last close and is held equal to a scan run each night; a security whose symbol (or spin-off child) was recorded in the security master late is recomputed as known that night | `tests/scan/test_bx_trender.py` |
| Sizing on prices rescaled later | A B-Xtrender entry takes its close and ATR from the bars visible at the signal night, adjusted only by the splits on file then whose ex-date has come | `tests/replay/test_replay_bx.py` |
| Trying settings until one works | Every parameter set a research replay runs under is registered and counted | `tests/strategy/test_config.py`, `tests/replay/test_report.py` |
| Classing an old listing by today's fund table | The SEC fund table is today's and tickers are reused (FB is a fund today, Facebook until 2022): a ticker counts as a fund only for a segment with no CIK or the fund's own; the audit lists fund-class segments of company filers | `tests/ingest/test_instrument_class.py`, `tests/store/test_audit_m9.py` |
| A share count before its filing was public | A market cap uses the SEC share count from the day after its filing date, of the company the security belonged to that night, with the splits on file then | `tests/store/test_shares.py` |
| Peeking at the test period | Replays and judge runs that touch the sealed holdout are refused; a strategy may open it once, and the opening is recorded | `tests/evolve/test_holdout.py`, `tests/replay/test_replay_holdout.py` |

On the real store, `asof market audit` runs 40 data checks with reviewed exceptions (the session
calendar among them, against the NYSE schedule), and
hand-checked reference histories (renames, ticker reuse, splits, spin-offs) are asserted as tests.
