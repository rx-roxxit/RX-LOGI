# RX Logi V2.0.1 - Model Card

**What it is.** A trade **planner**. On every confirmed ERL break it reads whether
market structure will **continue** or **reverse** (stage-1), and ranks the
**1-3 zones** price is most likely to launch from (stage-2).
Every output carries a confidence. **It is not an auto-trader and not an entry
signal** - the human confirms the setup in their own way and decides the entry.

Two models run in sequence (like two people handing off work): stage-2 only runs
when stage-1 says continue is the more likely read.

---

## How to call

```python
from inference import plan
out = plan(bars_by_tf, cell)     # cell in {"1H", "30m", "15m"}
```

Call it after every **closed** path-timeframe bar. It returns `None` unless that bar
just confirmed a new ERL break; otherwise it returns one plan.

### Input - `bars_by_tf`
A dict `{timeframe: DataFrame}`, each DataFrame with columns `open, high, low, close`
indexed by the bar's **OPEN** timestamp. Requirements (this is a hard model
requirement - the engine must satisfy it):

| item | requirement |
|---|---|
| timeframes per cell | **1H**: `D,4H,1H,30m,15m` - **30m**: `4H,1H,30m,15m,5m` - **15m**: `1H,30m,15m,5m,3m` |
| timezone | **UTC** (tz-naive or tz-aware UTC). The model converts to America/New_York **internally** for killzones - do **not** pre-shift. |
| bar convention | index = bar OPEN; pass only **closed** bars (open + timeframe_duration <= now) |
| ordering | ascending by time, contiguous, no gaps or duplicates per timeframe |
| lookback | **>= 12 months** of continuous history per timeframe so structure is warm (more is fine) |

### Output
`None`, or a dict:

| field | meaning |
|---|---|
| `instant` | ISO time of the break this plan is for |
| `direction` | `"up"` (structure made a high) / `"down"` (made a low) |
| `state` | `HH` / `LL` |
| `continue` | `true` if P(continue) > 0.5 |
| `dir_confidence` | **P(continue)** - how sure the read is |
| `erl_target` | the level a continuation makes new (the objective) |
| `invalidation` | break this and the read is wrong |
| `zones` | up to 3, each: `rank, lo, hi, touch, confidence, n_zones, n_tf, has_ob, has_fvg, has_line` |

`zones[].confidence` = **P(this zone is the launch, given the move continues)**.

**Recommended display.** Show a **setup grade** from `dir_confidence`
(S >= 90% - A >= 80% - B >= 70% - C >= 60% - D >= 50%) and each zone's own `confidence`
as a percent. **Do not** multiply the two into one number - the product is
mathematically compressed and does not tier. See `performance/`.

---

## Constraints & scope

- **Python runtime only.** The feature/structure pipeline is vendored byte-identical
  under `lib/` and must run natively. **Never re-implement it in another language** -
  that is where model quality dies. If the web is Node, run inference in a Python
  sidecar.
- **`scikit-learn==1.9.0` is required exactly** - the stage-2 `.joblib` models are
  version-sensitive. See `requirements.txt`.
- **Instruments.** Trained pooled on **XAUUSD, EURUSD, GBPUSD, US30(USA30),
  NAS100(USATECH)** with no instrument identifier - it is instrument-agnostic and
  transfers across the five. Use on those; other FX/index symbols are plausible but
  unvalidated. It is **not** for a symbol it has never seen the character of.
- **It is a planner, not P&L.** Numbers in `performance/` measure "did the read/zone
  come true", not win-rate or R. Entry, stop, target sizing, and the trade decision
  are the user's.
- **Confidence is calibrated** (especially the direction read) - a grade means what it
  says. See `performance/`.

### Known issue: OB re-freeze skew (recorded 2026-07-30, not fixed)

Stage-1's shipped weights were trained on order-block features as they stood
before the 2026-07-21 amendment, and are served with the amended features.
Measured over the 2015-2025 tape: the direction call flips on 0.15-0.50% of
plans and the confidence grade moves a tier on about 2.3%. Stage-2 is clean.

Left in place deliberately - 2.0.1 is a serving change and ships the same
weights as 2.0, byte for byte. Retraining would make the two versions
incomparable for no measured gain.

## Provenance

- stage-1: Transformer (ChainNet), 3-seed ensemble - stage-2: GBM
  (HistGradientBoosting), 3-seed ensemble.
- Trained 2016-2024, validated **blind on 2025**. The shipped `prod_*` weights are the
  same recipe retrained **including 2025**; 2026 is held out, so their exact accuracy
  is estimated by the 2025 figures.
- Feature freeze verified at load (`lib/.../frozen.py`): the serving features are
  checked byte-for-byte against the USER-verified archive on every import.

## Files

```
inference.py            the entry point - plan(bars_by_tf, cell)
config.json             the input contract + architecture
verify.py               re-run the bundled samples and check they reproduce
requirements.txt        pinned dependency versions
weights/                stage1/*.pt (+ ctx_mu/sd) - stage2/*.joblib
lib/                    the vendored feature/structure/model pipeline (byte-identical)
sample_io/              per-cell real bars + expected output (sanity on move)
performance/            what it can do, measured on 2025 (PERFORMANCE.md + PDF)
model_card.md           this file
```
