"""Incremental mirror of app/features/ob.py.

The frozen loop runs `for b in range(0, n - 3)` and confirms at b+3, so at bar i
the block under test starts at b = i - 3. The bullish branch `continue`s on a
hit, so bearish is only tested when bullish did not match - the if/else below
reproduces that exactly. Mitigation uses the same heaps as IncrFvg and, as
there, is evaluated BEFORE the birth at bar i.
"""
from __future__ import annotations

import heapq


class IncrOb:
    def __init__(self, dt, o, h, l, c) -> None:
        self.dt, self.o, self.h, self.l, self.c = dt, o, h, l, c
        self.boxes: list[dict] = []
        self._bull: list[tuple[float, int]] = []   # (-hi, idx)
        self._bear: list[tuple[float, int]] = []   # (lo,  idx)
        self._died: list[int] = []

    def feed(self, i: int) -> list[dict]:
        dt, o, h, l, c = self.dt, self.o, self.h, self.l, self.c
        self._died = []
        li, hi_ = float(l[i]), float(h[i])
        while self._bull and -self._bull[0][0] >= li:
            k = heapq.heappop(self._bull)[1]
            self.boxes[k]["mit_at"] = dt[i]
            self._died.append(k)
        while self._bear and self._bear[0][0] <= hi_:
            k = heapq.heappop(self._bear)[1]
            self.boxes[k]["mit_at"] = dt[i]
            self._died.append(k)
        self._died.sort()      # heaps pop by level; consumers need box order

        b = i - 3
        if b < 0:
            return []
        box = None
        fvg_a = l[b + 2] - h[b]
        fvg_b = l[b + 3] - h[b + 1]
        if (c[b] < o[b]
                and c[b + 1] > o[b + 1] and c[b + 1] > h[b]
                and c[b + 2] > o[b + 2]
                and c[b + 3] > o[b + 3]
                and (fvg_a > 0 or fvg_b > 0)):
            box = {"at": dt[b], "confirm_at": dt[b + 3], "side": "bull",
                   "lo": float(l[b]), "hi": float(h[b]),
                   "gap": round(float(max(fvg_a, fvg_b)), 6), "mit_at": None}
        else:
            gvf_a = l[b] - h[b + 2]
            gvf_b = l[b + 1] - h[b + 3]
            if (c[b] > o[b]
                    and c[b + 1] < o[b + 1] and c[b + 1] < l[b]
                    and c[b + 2] < o[b + 2]
                    and c[b + 3] < o[b + 3]
                    and (gvf_a > 0 or gvf_b > 0)):
                box = {"at": dt[b], "confirm_at": dt[b + 3], "side": "bear",
                       "lo": float(l[b]), "hi": float(h[b]),
                       "gap": round(float(max(gvf_a, gvf_b)), 6), "mit_at": None}
        if box is None:
            return []
        self.boxes.append(box)
        k = len(self.boxes) - 1
        if box["side"] == "bull":
            heapq.heappush(self._bull, (-box["hi"], k))
        else:
            heapq.heappush(self._bear, (box["lo"], k))
        return [box]

    def died_at_bar(self) -> list[int]:
        """Indices into self.boxes whose mit_at was set by the last fed bar."""
        return self._died

    def result(self) -> dict:
        return {"boxes": self.boxes}
