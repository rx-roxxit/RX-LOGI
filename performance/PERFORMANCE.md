# RX Logi V2.0 — Performance (end-to-end, blind 2025)

Everything here is the **whole planner run end-to-end** on **2025**, a year neither
model was trained on: stage-1 makes its own continue/reverse call, and only where it
says continue does stage-2 rank the zones. This is the honest product number — not an
"if the direction were known" figure. Pooled across the five instruments. Raw numbers
in `performance.json`.

Two things it does, measured separately, because they have different strengths:

## 1. The read (direction) — strong and calibrated

When the planner says "continue", how often structure actually continues, by
**setup grade** (from `dir_confidence`):

| grade | 1H dir-acc | 30m | 15m | ~setups/month (all cells) |
|---|---|---|---|---|
| **S** ≥90% | 0.94 | 0.80* | 1.00* | rare (a few) |
| **A** ≥80% | 0.88 | 0.88 | 0.86 | ~85 |
| **B** ≥70% | 0.74 | 0.80 | 0.77 | ~165 |
| **C** ≥60% | 0.65 | 0.67 | 0.65 | ~130 |
| **D** ≥50% | 0.51 | 0.60 | 0.55 | ~85 |

\* tiny sample. The grade **means what it says** — an A-grade read is right ~86–88%.
Acting on grade ≥A, direction is right **~86–89%**; on ≥B, **~80–83%**.

## 2. The zone — narrows to 3, the launch is often among them

Given the read, stage-2 shows up to 3 zones. On the plans where the planner says
continue (grade ≥D):

| cell | zones/plan | launch in top-3 (hit@3) | top-1 is the launch (hit@1) |
|---|---|---|---|
| 1H | ~8 → top 3 | **0.43** | 0.30 |
| 30m | ~10 → top 3 | **0.43** | 0.27 |
| 15m | ~11 → top 3 | **0.42** | 0.26 |

Each zone carries its own confidence = P(this zone launches, given the move continues),
and it is calibrated: 1H zones shown at ≥70% launch ~69% of the time (given continue),
≥80% ~100% (small n).

## 3. Fully correct (read AND zone) — filter both confidences

"hit@1" above is the whole thing right at once: direction continued **and** the #1 zone
was the launch. Cold it is ~0.26–0.30. It climbs when you act only where **both** the
read and the zone are confident:

| filter | 1H | 30m | 15m |
|---|---|---|---|
| read >50% & zone-conf ≥50% | 0.45 (≈14/mo) | 0.42 (≈25/mo) | 0.37 (≈50/mo) |
| read >60% & zone-conf ≥50% | 0.48 (≈10/mo) | 0.44 (≈17/mo) | 0.38 (≈36/mo) |
| read >70% & zone-conf ≥50% | **0.52** (≈6/mo) | **0.51** (≈9/mo) | 0.41 (≈19/mo) |

So a selective user, taking only high-conviction setups, gets the read-and-zone both
right ~41–52% of the time, at a workable frequency.

## Per-instrument (grade ≥B) — transfer holds

`dir-acc / top-1 / top-3`:

| instrument | 1H | 30m | 15m |
|---|---|---|---|
| **XAUUSD** (gold) | .79 / **.37** / .45 | .82 / .30 / .46 | .77 / .27 / .45 |
| EURUSD | .82 / .30 / .44 | .79 / .32 / .46 | .82 / .26 / .49 |
| GBPUSD | .81 / .33 / .54 | .85 / .28 / .47 | .83 / .27 / .41 |
| USA30 | .76 / .28 / .43 | .85 / .29 / .47 | .79 / .28 / .47 |
| USATECH | .83 / .30 / .46 | .83 / .25 / .41 | .80 / .27 / .44 |

No instrument collapses; gold is among the strongest on 1H.

## How to read this as a product

- The **direction read is the strong layer** — trust the grade; it is well-calibrated.
- The **zone layer narrows** the field to 3 candidates and is right (launch in the 3)
  ~43% of the time when the read is taken; the per-zone % tells you which to weight.
- It is a **decision-support planner**: it hands the user a graded read and a short
  list of zones with confidence. The user applies their own confirmation and decides
  the entry. It is **not** a win-rate and **not** an entry signal.

*(All figures: blind 2025, models trained ≤2024, pooled five instruments; the shipped
prod weights add 2025 and are estimated by these figures. 2026 is held out.)*
