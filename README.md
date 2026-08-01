# RX Logi V2.0

A trade **planner** (not an auto-trader, not an entry signal): on every ERL break it
reads continue/reverse and ranks the 1–3 launch zones, each with confidence. The human
confirms and decides.

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
- **`model_card.md`** — input/output contract, timeframes, timezone, constraints.
- **`config.json`** — the same contract, machine-readable.
- **`performance/`** — what it does on blind 2025 (`PERFORMANCE.md` + PDF).

## Layout
`inference.py` entry · `lib/` vendored pipeline (run natively, never re-port) ·
`weights/` stage-1 `.pt` + stage-2 `.joblib` · `sample_io/` re-runnable samples ·
`_build/` how the package was assembled (not needed to run).
