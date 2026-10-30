# Results in detail

The README keeps the headline numbers; this page keeps every replay's full account, in the order
they were run. Nothing here is out of sample unless it says so.

## Does the timing rule add anything (exploratory)

The rule: buy when the last completed weekly B-Xtrender bar rose, the daily one rose and is at or
above zero, and the weekly THT trend is green; sell on its exit signals. It came out of at least
18 exploratory versions tried on the seven largest tech stocks, which the attempt registry does
not cover, and was frozen on 2026-10-29 with eight variants (exit A or B, idle cash kept or
spread over the holdings, equal or risk weights). It was then run on three lists from 2020-02-03
to 2025-06-30, 5 bps a trade. These runs used exploratory scripts outside the replay pipeline,
and no pass mark was set in advance. Shown is A-spread-equal, the best of the eight on every
list; the other seven earn less.

"Held, same days" is the same list held throughout and re-weighted on the days the rule trades,
so that the only difference is when the rule is in or out of a stock. "Held, yearly" buys the
list each January and leaves it. SPY is price only.

| List | Rule | Held, same days | Held, yearly | Max drawdown, rule | Held, same days |
|---|--:|--:|--:|--:|--:|
| Top 20 US common stocks by market cap, as known each January | +117% | +118% | +135% | 29.0% | 30.2% |
| The same top 20 without the seven | +37% | +56% | +55% | 32.4% | 31.4% |
| The seven big tech stocks (in sample: the rule was shaped here) | +436% | +433% | +494% | 41.9% | 49.8% |
| SPY, price only | +91% | | | 34.2% | |

The top-20 list is picked each January from SEC share counts public by that night times that
night's close; Visa and Berkshire Hathaway report no all-class share count and are left out. About
a third of it is the seven. When a company has two share classes the more traded one is kept, so
Alphabet switched from GOOG (2021) to GOOGL (2022), which forced one sale the held list did not
make.

Deepest fall inside each sell-off (rule against held, same days):

| | 2020 crash | 2022 bear | 2025 tariffs | From 2022-04-18 |
|---|--:|--:|--:|--:|
| Top 20 | 29% / 30% | 21% / 29% | 16% / 14% | 23% / 22% |
| Without the seven | 32% / 31% | 16% / 22% | 16% / 12% | 21% / 18% |
| The seven | 42% / 33% | 30% / 50% | 24% / 19% | 35% / 40% |

The smaller falls in 2022 come mostly from the warm-up: the weekly THT colour needs history, and
Microsoft had none until 2022-04-18 (NVIDIA 2021-04-16, Apple 2020-05-15), so the rule could not
hold it while the held list did. From 2022-04-18, with every name tradable, the rule fell about
as far as holding on the top 20 and further without the seven, and trailed holding on return on
all three lists (+64% against +69%, +29% against +42%, +112% against +120%). Only on the seven,
where it was shaped, does a smaller fall remain.

The total depends on the start date. An earlier exploratory run of the same rule on the seven,
begun on 2020-01-02, shows +687% to 2025-06-30: it was up 17% by the end of January and 28% by
the close of 2020-02-03, and it carried those positions on, where a run begun on 2020-02-03 starts
in cash and waits for new entries.

No figure here uses a date after 2025-06-30. The rule's exploration on the seven did look at
2025-07 to 2025-12, inside the sealed holdout year, so the rule has no clean holdout left there.

## Earlier replays

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

Take-all over 2023-01-03 to 2026-10-16: 643 candidates, 547 traded, and the ten best trades
made 77% of the profit. Marked to market at every close:

|                       | Take-all | SPY     | QQQ     | Top-10 basket |
|-----------------------|---------:|--------:|--------:|--------------:|
| Total return          | +63.2%   | +107.7% | +169.6% | +417.7%       |
| Max drawdown          | 11.0%    | 18.8%   | 22.8%   | 29.5%         |
| Sharpe (daily, rf 0)  | 1.08     | 1.38    | 1.41    | 1.59          |

Benchmarks reinvest dividends. The strategy had 40% of its equity invested on average, so its
smaller drawdown is the cash, not an edge; a Sharpe ratio does not change with leverage, and on
it the strategy trails too, though with under four years of daily returns its standard error
is about 0.5 and the gap to SPY is within it. By period the Sharpe ratio was 1.50 on development
dates (SPY 1.35), 0.37 on the holdout's dates (SPY 1.68) and -0.07 after the model's cutoff
(SPY 0.80). This strategy was replayed over the holdout before the protocol existed, so none of
this is out of sample, and the watchlist was picked with hindsight. Against random
entries from the same candidate pool on the same days, its 632 closed legs averaged +1.73%
where random picks averaged +1.54%: an edge of +0.19% a leg, p = 0.33. Most of the per-trade
return came from the market and the watchlist, not from the gap. On development dates (where
filings start in 2023) the gate report shows +0.69% a leg over random entries, p = 0.11: no
evidence of an edge, from 383 legs that could only detect one above +1.9%. All of it comes
from three trades (QBTS, SMCI, DUOL); without them it is 0.00%. This strategy was replayed over
these dates before the protocol existed, so none of it is clean evidence.

The grid ran on throwaway copies of the database, so it adds nothing to the recorded attempt
count. It is still a search to anyone who picks its best cell, and is shown as such.

**B-Xtrender, development dates (2020-01 to 2025-06), three pre-registered variants.** None
beats SPY, so the holdout stays sealed:

| Variant | Window | Account | SPY | Max drawdown | SPY's |
|---|---|--:|--:|--:|--:|
| S1: weekly and daily histograms turn up | 2020-01 .. 2025-06 | -4.5% | +107.3% | 36.0% | 33.8% |
| S3: S1, new entries only while SPY is above its 200-day mean | 2020-01 .. 2025-06 | -10.1% | +107.3% | 35.8% | 33.8% |
| S2: S1, and the monthly bar rose (in cash until 2023-01) | 2023-01 .. 2025-06 | -5.3% | +66.3% | 25.9% | 18.8% |

S2 needs 36 completed monthly bars, so it can trade only from 2023-01 and is compared over that
window. The three variants count as three attempts. S1's 3,392 half-position legs averaged
-0.56% where random picks from the same pool on the same dates averaged +0.35%, p = 1.00 after
the family correction; positive in 2 of 7 walk-forward windows. The random legs keep the real
legs' exit dates, and those come from each stock's own path (a sale the morning after the
daily histogram turns red), so this comparison does not tell a bad entry from a bad exit. The
replay books no dividends (about a point a year at 63% invested), and about 3% of entries were
sized on an ATR inflated by an extreme or bad high or low (NVDA 2024-06-10 stores a high of
195.95 against a close of 121.79); neither is near the gap. [Details](commands.md#b-xtrender-on-development-dates)
