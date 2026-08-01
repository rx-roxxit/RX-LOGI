"""Incremental mirror of app/features/fvg.py.

Two things the frozen module does in two passes happen here in one:
  * birth  - the 3-bar test at bar i (lookback 2 bars, nothing older)
  * mit_at - frozen scans forward from cb+1 per box, which is O(boxes x bars).
             Here the unmitigated boxes sit in two heaps keyed by the edge that
             would be touched, so a bar only pops the boxes it actually hits.
Order matters: mitigation is checked BEFORE the birth at bar i, because the
frozen scan starts at cb+1 - a box is never mitigated by its own confirm bar.
"""
from __future__ import annotations

import heapq


class IncrFvg:
    def __init__(self, dt, o, h, l, c) -> None:
        self.dt, self.h, self.l = dt, h, l
        self.boxes: list[dict] = []
        self._bull: list[tuple[float, int]] = []   # (-hi, idx)  max-heap on hi
        self._bear: list[tuple[float, int]] = []   # (lo,  idx)  min-heap on lo
        self._died: list[int] = []

    def feed(self, i: int) -> list[dict]:
        h, l, dt = self.h, self.l, self.dt
        self._died = []
        li, hi_ = float(l[i]), float(h[i])
        while self._bull and -self._bull[0][0] >= li:      # bull: low <= zone top
            k = heapq.heappop(self._bull)[1]
            self.boxes[k]["mit_at"] = dt[i]
            self._died.append(k)
        while self._bear and self._bear[0][0] <= hi_:      # bear: high >= zone bottom
            k = heapq.heappop(self._bear)[1]
            self.boxes[k]["mit_at"] = dt[i]
            self._died.append(k)
        self._died.sort()      # heaps pop by level; consumers need box order

        if i < 2:
            return []
        bull = h[i - 2] < l[i]
        bear = l[i - 2] > h[i]
        if not (bull or bear):
            return []
        lo = float(h[i - 2]) if bull else float(h[i])
        hi = float(l[i]) if bull else float(l[i - 2])
        box = {"at": dt[i - 1], "confirm_at": dt[i],
               "side": "bull" if bull else "bear",
               "lo": lo, "hi": hi, "mit_at": None}
        self.boxes.append(box)
        k = len(self.boxes) - 1
        heapq.heappush(self._bull if bull else self._bear,
                       ((-hi, k) if bull else (lo, k)))
        return [box]

    def died_at_bar(self) -> list[int]:
        """Indices into self.boxes whose mit_at was set by the last fed bar."""
        return self._died

    def result(self) -> dict:
        return {"boxes": self.boxes}
