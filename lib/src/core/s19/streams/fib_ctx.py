"""FibCtxStream == context.fib_state, built one bar at a time.

Everything here is a running quantity already: the current expansion leg, how
far price has run since it started, and how deep the previous three legs
retraced. The only subtlety is that the run-extremes are seeded from the CLOSE
of the bar the leg was confirmed on and then immediately widened by that same
bar's low and high - the legacy code does exactly that, in that order.

When more than one leg is confirmed on a bar, the legacy code keeps the LAST one
(`born[k][-1]`).
"""
from __future__ import annotations

import numpy as np

from ..context import FIB_FEATS


class FibCtxStream:
    def __init__(self) -> None:
        self._rows: list[list[float]] = []
        self._cur: dict | None = None
        self._cur_from = 0
        self._depths: list[float] = []
        self._lo_run = 0.0
        self._hi_run = 0.0

    def feed(self, i: int, h, l, c, born: list[dict]) -> None:
        price = float(c[i])
        if born:
            if self._cur is not None:              # close the previous leg
                span = self._cur["start_price"] - self._cur["end_price"]
                if abs(span) > 1e-12:
                    if self._cur["dir"] == "down":
                        ext = (self._hi_run - self._cur["end_price"]) / span
                    elif span < 0:
                        ext = (self._cur["end_price"] - self._lo_run) / -span
                    else:
                        ext = 0.0
                    self._depths.append(float(np.clip(ext, -1.0, 3.0)))
            self._cur = born[-1]
            self._cur_from = i
            self._lo_run = self._hi_run = price
        if self._cur is None:
            self._rows.append([0.0] * len(FIB_FEATS))
            return
        self._lo_run = min(self._lo_run, float(l[i]))
        self._hi_run = max(self._hi_run, float(h[i]))
        span = self._cur["start_price"] - self._cur["end_price"]
        pos = (price - self._cur["end_price"]) / span if abs(span) > 1e-12 else 0.0
        prev = (self._depths[-3:] + [0.0, 0.0, 0.0])[:3][::-1]
        self._rows.append([float(np.clip(pos, -1.0, 3.0)),
                           1.0 if self._cur["dir"] == "up" else 0.0,
                           min((i - self._cur_from) / 100.0, 3.0), *prev])

    def cap(self) -> None:
        """Drop rows no future plan can see: live only ever reads the newest."""
        if len(self._rows) > 2:
            self._rows = self._rows[-1:]

    def last_row(self) -> np.ndarray:
        out = np.zeros((1, len(FIB_FEATS)), np.float32)
        out[0] = self._rows[-1]
        return out

    def arrays(self) -> tuple[np.ndarray, list[str]]:
        out = np.zeros((len(self._rows), len(FIB_FEATS)), np.float32)
        for k, row in enumerate(self._rows):
            out[k] = row
        return out, FIB_FEATS
