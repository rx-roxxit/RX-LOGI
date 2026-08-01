"""Incremental mirror of app/features/erl_irl.py.

`hi_irl`/`lo_irl` are indices into the output list assigned BEFORE the marker is
appended, so they point at the marker about to be added - reproduced here by
appending after the branch, exactly as the frozen code does.

Promotion is retroactive: a marker written long ago gains `promote` when a later
break upgrades it. That is why `result()` is built at call time from `raw`.

`items_at_bar()` exists for fib.py, which orders the ERL/promote stream by
(knowability, bar). Everything generated at bar i shares the same knowability,
so sorting the bar's items by `bar` reproduces the frozen sort key exactly.
"""
from __future__ import annotations

from .swing import IncrSwing


class IncrErlIrl:
    def __init__(self, dt, o, h, l, c, *, swings: IncrSwing | None = None) -> None:
        self.dt = dt
        self.sw = swings if swings is not None else IncrSwing(dt, o, h, l, c)
        self._own_swings = swings is None
        self.raw: list[dict] = []
        self._trend = 0
        self._has_hi = self._has_lo = False
        self._erl_hi = self._erl_lo = 0.0
        self._lo_irl: int | None = None
        self._hi_irl: int | None = None
        self._items: list[dict] = []

    def feed(self, i: int) -> list[dict]:
        if self._own_swings:
            self.sw.feed(i)
        self._items = []
        ev = self.sw.last
        if ev is None or ev["confirm"] != i:
            return []
        m = {"bar": ev["bar"], "confirm": ev["confirm"], "side": ev["side"],
             "price": ev["price"], "cls": ev["cls"], "role": "IRL", "promote": None}
        v = ev["price"]
        if ev["side"] == "high":
            if not self._has_hi:
                m["role"] = "ERL"; self._erl_hi = v; self._has_hi = True; self._trend = 1
            elif v > self._erl_hi:
                m["role"] = "ERL"; self._erl_hi = v
                if self._trend == 1 and self._lo_irl is not None:
                    promoted = self.raw[self._lo_irl]
                    promoted["promote"] = ev["confirm"]
                    self._items.append(promoted)
                    self._erl_lo = promoted["price"]; self._has_lo = True
                self._trend = 1; self._lo_irl = self._hi_irl = None
            else:
                if self._hi_irl is None or v > self.raw[self._hi_irl]["price"]:
                    self._hi_irl = len(self.raw)
        else:
            if not self._has_lo:
                m["role"] = "ERL"; self._erl_lo = v; self._has_lo = True; self._trend = -1
            elif v < self._erl_lo:
                m["role"] = "ERL"; self._erl_lo = v
                if self._trend == -1 and self._hi_irl is not None:
                    promoted = self.raw[self._hi_irl]
                    promoted["promote"] = ev["confirm"]
                    self._items.append(promoted)
                    self._erl_hi = promoted["price"]; self._has_hi = True
                self._trend = -1; self._lo_irl = self._hi_irl = None
            else:
                if self._lo_irl is None or v < self.raw[self._lo_irl]["price"]:
                    self._lo_irl = len(self.raw)
        self.raw.append(m)
        if m["role"] == "ERL":
            self._items.append(m)
        self._items.sort(key=lambda q: q["bar"])
        return [m]

    def items_at_bar(self) -> list[dict]:
        return self._items

    def result(self) -> dict:
        dt = self.dt
        return {"markers": [
            {"at": dt[m["bar"]], "confirm_at": dt[m["confirm"]], "side": m["side"],
             "price": m["price"], "cls": m["cls"], "role": m["role"],
             "promote_at": dt[m["promote"]] if m["promote"] is not None else None}
            for m in self.raw]}
