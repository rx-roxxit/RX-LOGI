"""Incremental mirror of app/features/sweep.py.

Deliberately a straight transcription of the frozen per-bar body: the pools are
capped at MAX_SWINGS per side and carry a pending-reclaim bar, so there is no
ordering trick to exploit and the reverse iteration must be preserved verbatim -
frozen deletes while iterating, and `markers` order is part of the contract.

(stage2.sweep_markers already mirrors this module with the reclaim window as a
parameter; this class mirrors the module's own output, marker fields included.)
"""
from __future__ import annotations

from ..frozen import sweep as _fsweep
from .swing import IncrSwing


class IncrSweep:
    def __init__(self, dt, o, h, l, c, *, swings: IncrSwing | None = None) -> None:
        self.dt, self.h, self.l, self.c = dt, h, l, c
        self.sw = swings if swings is not None else IncrSwing(dt, o, h, l, c)
        self._own_swings = swings is None
        self.markers: list[dict] = []
        self._ssl: list[dict] = []
        self._bsl: list[dict] = []

    def _mark(self, i: int, side: str, price: float, q: dict) -> dict:
        m = {"at": self.dt[i], "confirm_at": self.dt[i], "side": side,
             "price": price, "level": q["lvl"], "level_at": self.dt[q["bar"]]}
        self.markers.append(m)
        return m

    def feed(self, i: int) -> list[dict]:
        if self._own_swings:
            self.sw.feed(i)
        h, l, c = self.h, self.l, self.c
        ev = self.sw.last
        if ev is not None and ev["confirm"] == i:
            pool = self._ssl if ev["side"] == "high" else self._bsl
            pool.append({"lvl": ev["price"], "bar": ev["bar"], "brk": -1})
            if len(pool) > _fsweep.MAX_SWINGS:
                del pool[0: len(pool) - _fsweep.MAX_SWINGS]

        born: list[dict] = []
        for j in range(len(self._ssl) - 1, -1, -1):
            q = self._ssl[j]
            if q["brk"] < 0:
                if h[i] > q["lvl"]:
                    if c[i] <= q["lvl"]:
                        born.append(self._mark(i, "SSL", float(h[i]), q))
                        del self._ssl[j]
                    else:
                        q["brk"] = i
            else:
                if c[i] <= q["lvl"]:
                    born.append(self._mark(i, "SSL", float(h[i]), q))
                    del self._ssl[j]
                elif i - q["brk"] >= _fsweep.RECLAIM_BARS:
                    del self._ssl[j]

        for j in range(len(self._bsl) - 1, -1, -1):
            q = self._bsl[j]
            if q["brk"] < 0:
                if l[i] < q["lvl"]:
                    if c[i] >= q["lvl"]:
                        born.append(self._mark(i, "BSL", float(l[i]), q))
                        del self._bsl[j]
                    else:
                        q["brk"] = i
            else:
                if c[i] >= q["lvl"]:
                    born.append(self._mark(i, "BSL", float(l[i]), q))
                    del self._bsl[j]
                elif i - q["brk"] >= _fsweep.RECLAIM_BARS:
                    del self._bsl[j]
        return born

    def result(self) -> dict:
        return {"markers": self.markers}
