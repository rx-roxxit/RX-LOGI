# RX Logi V2.0

A trade **planner** (not an auto-trader, not an entry signal): on every ERL break it
reads continue/reverse and ranks the 1-3 launch zones, each with confidence. The human
confirms and decides.

**Serving this in production? Read [SERVING.md](SERVING.md) first.** It carries
the feed contract, the memory and warm-start budgets, and the known issues.
The feed contract in particular does not fail loudly when you get it wrong.

Two entry points: `LiveBook` holds state and costs the bar rather than the
decade - use it to serve. `plan(bars_by_tf, cell)` recomputes from the start of
the tape on every call - use it for a one-off answer or as a reference.

## Run
```bash
pip install -r requirements.txt          # Python 3.11; scikit-learn pin is exact
python verify.py                         # sanity: re-run the bundled samples
```
```python
from inference import plan
out = plan(bars_by_tf, "1H")             # see model_card.md for the input contract
```

## Read first
- **`model_card.md`** - input/output contract, timeframes, timezone, constraints.
- **`config.json`** - the same contract, machine-readable.
- **`performance/`** - what it does on blind 2025 (`PERFORMANCE.md` + PDF).

## Layout
`inference.py` entry - `lib/` vendored pipeline (run natively, never re-port) -
`weights/` stage-1 `.pt` + stage-2 `.joblib` - `sample_io/` re-runnable samples -
`_build/` how the package was assembled (not needed to run).
