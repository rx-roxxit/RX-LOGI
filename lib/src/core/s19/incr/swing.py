"""Incremental mirror of app/features/swing.py.

The state machine itself is NOT mirrored - `frozen.swing.SwingSM` already
exposes `update(i)`, so it is reused as-is. Only the HH/HL/LH/LL classification
loop of `swing_events()` is carried here (two variables).
"""
from __future__ import annotations

from ..frozen import swing as _fswing


class IncrSwing:
    def __init__(self, dt, o, h, l, c) -> None:
        self.dt = dt
        self.sm = _fswing.SwingSM(h, l, c)
        self.events: list[dict] = []      # == swing.swing_events()
        self.markers: list[dict] = []     # == swing.compute()["markers"]
        self.last: dict | None = None     # event confirmed at the last fed bar
        self._prev_hi: float | None = None
        self._prev_lo: float | None = None

    def feed(self, i: int) -> list[dict]:
        self.last = None
        ev = self.sm.update(i)
        if ev is None:
            return []
        if ev["side"] == "high":
            ev["cls"] = "H" if self._prev_hi is None else (
                "HH" if ev["price"] > self._prev_hi else "LH")
            self._prev_hi = ev["price"]
        else:
            ev["cls"] = "L" if self._prev_lo is None else (
                "HL" if ev["price"] > self._prev_lo else "LL")
            self._prev_lo = ev["price"]
        self.events.append(ev)
        self.markers.append({"at": self.dt[ev["bar"]], "confirm_at": self.dt[i],
                             "side": ev["side"], "price": ev["price"],
                             "cls": ev["cls"]})
        self.last = ev
        return [self.markers[-1]]

    def result(self) -> dict:
        return {"markers": self.markers}
