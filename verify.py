# -*- coding: utf-8 -*-
"""Sanity check after moving the package to a new machine.

Re-runs every bundled sample and confirms plan() reproduces the frozen expected
output. If this passes with only this folder present (no repo, no training env),
the package is wired correctly and the models load and run.

    python verify.py
"""
import json
import os

import pandas as pd

from inference import plan

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "sample_io")


def _canon(x):
    return json.dumps(x, sort_keys=True)


def main():
    cells = sorted(d for d in os.listdir(SAMPLES)
                   if os.path.isdir(os.path.join(SAMPLES, d)))
    for cell in cells:
        d = os.path.join(SAMPLES, cell)
        exp = json.load(open(os.path.join(d, "expected.json")))
        bars = {}
        for f in os.listdir(d):
            if f.startswith("bars_") and f.endswith(".parquet"):
                tf = f[len("bars_"):-len(".parquet")]
                bars[tf] = pd.read_parquet(os.path.join(d, f))
        out = plan(bars, cell)
        assert _canon(out) == _canon(exp["expected"]), (
            f"[{cell}] output changed:\n got {out}\n exp {exp['expected']}")
        print(f"[{cell}] OK  {out['instant']}  {out['direction']} "
              f"cont={out['dir_confidence']} zones={len(out['zones'])}")
    print(f"\nVERIFY PASSED — {len(cells)} samples reproduce. Package is self-contained.")


if __name__ == "__main__":
    main()
