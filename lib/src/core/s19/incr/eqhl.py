"""Incremental mirror of app/features/eqhl.py.

atr_series() keeps two names (`run` during the seed phase, `atr[i-1]` after) but
they are the same number at every step, so ONE running value reproduces it. The
constants come from the frozen module rather than being retyped, so a change
there cannot silently diverge here (and W2 would catch the change anyway).

Order matters: the ATR is advanced for bar i BEFORE the swing test, because the
frozen code reads atr[ev["confirm"]] - the confirm bar's own value.
"""
from __future__ import annotations

from ..frozen import eqhl as _feqhl
from .swing import IncrSwing


class IncrEqhl:
    def __init__(self, dt, o, h, l, c, *, swings: IncrSwing | None = None) -> None:
        self.dt, self.h, self.l, self.c = dt, h, l, c
        self.sw = swings if swings is not None else IncrSwing(dt, o, h, l, c)
        self._own_swings = swings is None
        self.events: list[dict] = []
        self.atr: float = 0.0
        self._prev_hi: dict | None = None
        self._prev_lo: dict | None = None

    def feed(self, i: int) -> list[dict]:
        h, l, c, dt = self.h, self.l, self.c, self.dt
        if i == 0:
            self.atr = h[0] - l[0]
        else:
            tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            if i < _feqhl.ATR_LEN:
                self.atr = (self.atr * i + tr) / (i + 1)
            else:
                self.atr = (self.atr * (_feqhl.ATR_LEN - 1) + tr) / _feqhl.ATR_LEN

        if self._own_swings:
            self.sw.feed(i)
        ev = self.sw.last
        if ev is None or ev["confirm"] != i:
            return []
        cur = {"bar": ev["bar"], "price": ev["price"]}
        born: list[dict] = []
        if ev["side"] == "high":
            if self._prev_hi is not None and \
                    abs(cur["price"] - self._prev_hi["price"]) < _feqhl.THRESH * self.atr:
                e = {"at": dt[self._prev_hi["bar"]], "at2": dt[cur["bar"]],
                     "p1": self._prev_hi["price"], "p2": cur["price"],
                     "kind": "EQH", "confirm_at": dt[i]}
                self.events.append(e)
                born.append(e)
            self._prev_hi = cur
        else:
            if self._prev_lo is not None and \
                    abs(cur["price"] - self._prev_lo["price"]) < _feqhl.THRESH * self.atr:
                e = {"at": dt[self._prev_lo["bar"]], "at2": dt[cur["bar"]],
                     "p1": self._prev_lo["price"], "p2": cur["price"],
                     "kind": "EQL", "confirm_at": dt[i]}
                self.events.append(e)
                born.append(e)
            self._prev_lo = cur
        return born

    def result(self) -> dict:
        return {"events": self.events}
