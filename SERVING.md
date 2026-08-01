# Serving RX Logi

Read sections 1 and 2 before you write any integration code. Everything else
here helps you size a machine. Getting either section wrong does not raise an
error - it returns plans that are quietly wrong.

## 1. Feed bars GROUPED BY CLOSE INSTANT, never one at a time

Gather every bar that closed at the same moment, across every timeframe, and
hand them over as one group:

```python
from inference import LiveBook

book = LiveBook(("1H", "30m", "15m"), retain="live")
book.warm(history_by_tf)          # once, at startup

# then, per close instant:
plans = book.on_closed_bars([
    ("30m", (ts_30m, o, h, l, c)),
    ("15m", (ts_15m, o, h, l, c)),
    ("5m",  (ts_5m,  o, h, l, c)),
    ("3m",  (ts_3m,  o, h, l, c)),
])
# plans == {} on ~99% of instants. Otherwise {cell: plan} for the cells that fired.
```

Groups must arrive in strictly increasing close time, and every bar in a group
must close at the same instant. Both are checked and raise `RuntimeError`.

### Why a group and not a bar

A plan made at time T is allowed to see everything that closed AT T. The zone
enumerator admits a zone whose confirm bar closes exactly at T, and the context
lookup is right-inclusive for the same reason.

But T is also the close of the bar that *triggers* the plan. At 08:30 the 3m,
5m, 15m and 30m bars all close together, and the path timeframe sits in the
middle of any ordering you could choose:

```
coarse first:  15m arrives before 3m  -> the 3m zone that closed at 08:30 is missing
fine first  :  3m arrives before 30m  -> the 30m context row is missing
```

There is no per-bar order that is correct. The engine has to take the whole
group, then decide. This is not a style preference: during development, feeding
bar by bar produced 49 candidate zones where the batch reference produced 50,
with no error raised.

### What "the same instant" means

A bar stamped `ts` on timeframe `tf` closes at `ts + tf`. A `D` bar stamped
`2026-01-01 00:00` closes at `2026-01-02 00:00`, which is the same instant a
`15m` bar stamped `2026-01-01 23:45` closes. Those two belong in one group.

Do not put the same timeframe in a group twice. It is not checked, and it will
feed that timeframe's store twice.

## 2. What every bar must look like

Every timeframe in `bars_by_tf` must meet all of the rules below:

- **Index:** Each bar is indexed by its **OPEN** timestamp, not its close time.
- **Timezone:** Use UTC. Pass tz-naive UTC or tz-aware UTC. The model converts
  to America/New_York **INTERNALLY** for killzone features - do not pre-shift.
  Passing New York time because the model uses New York internally is exactly
  the mistake this rule warns against.
- **Bar convention:** pass only CLOSED bars. A bar stamped `t` on timeframe `tf`
  is usable once `t + tf <= now`. Passing the bar that is still forming is the
  same class of mistake as feeding one at a time - it will not raise.
- **Shape:** `bars_by_tf` is a dict of timeframe to a pandas DataFrame with
  columns `open`, `high`, `low`, `close`, indexed as above. `LiveBook.warm()`
  takes that dict; `on_closed_bars()` takes tuples, as in section 1.
- **Which timeframes:** a cell needs all five of its own, and `warm()` raises if
  any is missing.

  | cell | timeframes |
  |---|---|
  | `1H` | 15m, 30m, 1H, 4H, D |
  | `30m` | 5m, 15m, 30m, 1H, 4H |
  | `15m` | 3m, 5m, 15m, 30m, 1H |

  A `LiveBook` holding several cells needs the union of their rows.
- **Ordering:** Within each timeframe, bars must be ascending by time,
  contiguous, and have no gaps or duplicates.
- **Minimum lookback:** Provide **>= 12 months** of continuous history per
  timeframe so the structure state (ERL, swings, zones) is warm; more history
  is fine and preferred.

## 3. Which entry point

| | |
|---|---|
| `LiveBook` | serving. Holds state, costs the bar rather than the decade. |
| `plan(bars_by_tf, cell)` | a one-off answer, or a reference. Recomputes everything from the start of the tape on every call. |

`LivePlanner(cell)` is a `LiveBook` holding one cell, for when you serve a
single cell.

## 4. Memory

Measured on XAUUSD over 2015-2025, all three cells sharing one book:

| | |
|---|---|
| per symbol | **763 MB** |
| five symbols, ONE process | **~4.4 GB** |
| five symbols, five processes | ~7.2 GB |

**Run one process.** Each process pays about 560 MB for Python, torch and the
model weights before any market data; five processes pay it five times.

## 5. Warm start

**331 s per symbol** for the full 2015-2025 history (129.6 microseconds per
bar). There is no snapshot: a restart re-walks the history.

That number is from an 8-core i7. On a slower box it is a floor, not an
estimate - measure it on yours.

## 6. Memory grows with the bars you feed, without a ceiling

Zones are reaped as they die: after eleven years only 8,982 are still live, and
that number is stable. But the bar-indexed side - the OHLC arrays and the
timestamp list - is never reaped, and it costs about **299 bytes per bar fed**.

For a 3m stream that is roughly 30 MB per year, per symbol. It will not
explode, but it does not level off either.

**Have a restart policy.** There is no history-trimming call in this version.

## 7. Known issue: OB re-freeze skew

The shipped stage-1 was trained on order-block features as they were before a
2026-07-21 amendment, and serves with the amended ones. Measured impact:
direction flips on 0.15-0.50% of plans, and the confidence grade moves a tier
on about 2.3%. Stage-2 is unaffected.

This is recorded rather than fixed: 2.0.1 ships the same weights as 2.0, byte
for byte, on purpose.

## 8. Check your install

```
pip install -r requirements.txt
python verify.py
```

It reproduces three frozen samples through both `plan()` and `LiveBook`.

Expect the whole script to take about four to five minutes on the reference
machine, against roughly thirty seconds for the batch check alone in earlier
versions. Most of that is the `LiveBook` replay - about 140 s for 15m, 51 s for
30m and 19 s for 1H - because it computes a plan at every structural break the
way a live feed would. Each cell announces itself before it starts, so the
script is not hung when it goes quiet. Do not kill it.

If it passes with only this folder present, the package is wired correctly.
