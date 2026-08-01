"""Incremental mirror of app/features/liquidity.py.

The frozen module walks EVERY active level on EVERY bar - that inner loop is the
single most expensive thing in the whole pipeline (133.5s of the 278.5s that
zones_on_tf costs on a 3m tape). Levels are ordered here instead: SSL sits above
price, so the LOWEST unswept SSL is the first one a high can take; BSL mirrors
it. A bar therefore pops only the levels it actually sweeps.

Order matters: the frozen loop births the bar's new level BEFORE the sweep scan,
so a level can be born and swept on the same bar. Same order here.
"""
from __future__ import annotations

import heapq

from .swing import IncrSwing


class IncrLiquidity:
    def __init__(self, dt, o, h, l, c, *, swings: IncrSwing | None = None) -> None:
        self.dt, self.h, self.l = dt, h, l
        self.sw = swings if swings is not None else IncrSwing(dt, o, h, l, c)
        self._own_swings = swings is None
        self.lines: list[dict] = []
        self._ssl: list[tuple[float, int]] = []    # (level, idx)   min-heap
        self._bsl: list[tuple[float, int]] = []    # (-level, idx)  max-heap
        self._died: list[int] = []

    def feed(self, i: int) -> list[dict]:
        self._died = []
        if self._own_swings:
            self.sw.feed(i)
        dt = self.dt
        born: list[dict] = []
        ev = self.sw.last
        if ev is not None and ev["confirm"] == i:
            side = "SSL" if ev["side"] == "high" else "BSL"
            line = {"at": dt[ev["bar"]], "confirm_at": dt[i],
                    "level": ev["price"], "side": side, "swept_at": None}
            self.lines.append(line)
            k = len(self.lines) - 1
            if side == "SSL":
                heapq.heappush(self._ssl, (line["level"], k))
            else:
                heapq.heappush(self._bsl, (-line["level"], k))
            born.append(line)

        hi_, lo_ = float(self.h[i]), float(self.l[i])
        while self._ssl and self._ssl[0][0] < hi_:          # high > level
            k = heapq.heappop(self._ssl)[1]
            self.lines[k]["swept_at"] = dt[i]
            self._died.append(k)
        while self._bsl and -self._bsl[0][0] > lo_:         # low < level
            k = heapq.heappop(self._bsl)[1]
            self.lines[k]["swept_at"] = dt[i]
            self._died.append(k)
        self._died.sort()      # heaps pop by level; consumers need line order
        return born

    def died_at_bar(self) -> list[int]:
        """Indices into self.lines whose swept_at was set by the last fed bar."""
        return self._died

    def result(self) -> dict:
        return {"lines": self.lines}
