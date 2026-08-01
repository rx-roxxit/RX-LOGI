# -*- coding: utf-8 -*-
"""Sanity check after moving the package to a new machine.

Re-runs every bundled sample and confirms plan() reproduces the frozen expected
output. If this passes with only this folder present (no repo, no training env),
the package is wired correctly and the models load and run.

Two paths are checked for every sample: plan(), which recomputes from the start
of the tape, and LiveBook, which holds the market as state and takes one group
of closed bars at a time. The LiveBook replay is the slow part - it computes a
plan at every structural break the way a live feed would, so expect roughly
four to five minutes in total rather than the thirty seconds the batch check
alone took.

    python verify.py
"""
import json
import os
import time

import pandas as pd

from inference import CELLS_ALL, LiveBook, _to_tape, plan
from src.core.s19.replay import replay_feed

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "sample_io")


def _canon(x):
    return json.dumps(x, sort_keys=True)


def main():
    cells = sorted(d for d in os.listdir(SAMPLES)
                   if os.path.isdir(os.path.join(SAMPLES, d)))
    missing = sorted(set(CELLS_ALL) - set(cells))
    if missing:
        raise SystemExit(
            f"sample_io is missing {missing} - this package is incomplete and "
            f"cannot be verified. Expected a directory per cell in {CELLS_ALL}.")
    for cell in cells:
        d = os.path.join(SAMPLES, cell)
        exp = json.load(open(os.path.join(d, "expected.json")))['expected']
        bars = {}
        for f in os.listdir(d):
            if f.startswith("bars_") and f.endswith(".parquet"):
                tf = f[len("bars_"):-len(".parquet")]
                bars[tf] = pd.read_parquet(os.path.join(d, f))
        out = plan(bars, cell)
        assert _canon(out) == _canon(exp), (
            f"[{cell}] output changed:\n got {out}\n exp {exp}")
        print(f"[{cell}] OK  {out['instant']}  {out['direction']} "
              f"cont={out['dir_confidence']} zones={len(out['zones'])}")

        # Gate 6 - the incremental path, on exactly the same bars.
        #
        # expected.json IS plan()'s frozen output, so reproducing it proves
        # LiveBook == plan() as well; there is no need to compare the two
        # directly.
        #
        # Look the plan up by instant rather than taking the last one: LiveBook
        # emits a plan at every ERL break where plan() returns only the latest,
        # and an assumption about the tail of a sequence is a thing this
        # project has already been bitten by once.
        # _to_tape is exactly what plan() runs the DataFrames through
        # (inference.py) - sorted index, tz-naive UTC. Calling it rather than
        # imitating it is what makes "the same bars" true by construction.
        replay_bars = {tf: _to_tape(df) for tf, df in bars.items()}
        print(f"[{cell}] replaying {sum(len(v[0]) for v in replay_bars.values())} "
              "bars through LiveBook (this is the slow part, ~1-2 min per cell)",
              flush=True)
        lb = LiveBook((cell,), retain="live")
        seen = {}
        started = time.perf_counter()
        for group in replay_feed(replay_bars):
            for _c, p in lb.on_closed_bars(group).items():
                seen[p["instant"]] = p
        elapsed = time.perf_counter() - started
        inst = exp["instant"]
        if inst not in seen:
            raise AssertionError(
                f"[{cell}] LiveBook produced no plan at {inst}\n"
                f"  got     : None ({len(seen)} plans replayed, "
                f"instants {sorted(seen)[:3]}"
                f"{' ...' if len(seen) > 3 else ''})\n"
                f"  expected: {exp}")
        if _canon(seen[inst]) != _canon(exp):
            raise AssertionError(
                f"[{cell}] LiveBook plan at {inst} differs from expected:\n"
                f" got {seen[inst]}\n exp {exp}")
        print(f"[{cell}] OK  LiveBook reproduces {inst} "
              f"({len(seen)} plans replayed, {elapsed:.0f}s)", flush=True)
    print(f"\nVERIFY PASSED - {len(cells)} samples reproduce via batch and LiveBook. "
          "Package is self-contained.")


if __name__ == "__main__":
    main()
