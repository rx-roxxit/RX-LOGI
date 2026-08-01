"""Incremental mirror of app/features/fib.py.

Frozen builds the whole ERL/promote stream, sorts it by (knowability, bar), then
walks it keeping every seen swing and picking, for each expansion, the most
recent opposite-side swing that sits STRICTLY EARLIER on the chart. Two details
are load-bearing:

  * everything generated at bar i shares one knowability, so ordering that bar's
    items by `bar` reproduces the frozen sort exactly (IncrErlIrl already does);
  * `max(bar) with bar < current` is not a running maximum - a promoted swing
    enters the stream late but carries an OLD bar. A sorted list + bisect gives
    the exact answer in O(log n) where the frozen list comprehension is O(n).

`replaced_at` is the next leg's confirm_at, so it is stamped on the previous leg
when a new one is born - the same pairing the frozen zip() does at the end.
"""
from __future__ import annotations

from bisect import bisect_left, insort

from .erl_irl import IncrErlIrl


class IncrFib:
    def __init__(self, dt, o, h, l, c, *, erl: IncrErlIrl | None = None) -> None:
        self.dt = dt
        self.erl = erl if erl is not None else IncrErlIrl(dt, o, h, l, c)
        self._own_erl = erl is None
        self.fibs: list[dict] = []
        self._bars: dict[str, list[int]] = {"high": [], "low": []}
        self._price: dict[tuple[str, int], float] = {}

    def feed(self, i: int) -> list[dict]:
        if self._own_erl:
            self.erl.feed(i)
        dt = self.dt
        born: list[dict] = []
        for m in self.erl.items_at_bar():
            bar = m["bar"]
            expands = m["role"] == "ERL" and (
                (m["side"] == "high" and m["cls"] == "HH") or
                (m["side"] == "low" and m["cls"] == "LL"))
            if expands:
                opp = "low" if m["side"] == "high" else "high"
                bars = self._bars[opp]
                j = bisect_left(bars, bar) - 1        # newest opposite swing before `bar`
                if j >= 0:
                    sbar = bars[j]
                    f = {"start_at": dt[sbar], "start_price": self._price[(opp, sbar)],
                         "end_at": dt[bar], "end_price": m["price"],
                         "dir": "up" if m["side"] == "high" else "down",
                         "confirm_at": dt[i], "replaced_at": None}
                    if self.fibs:
                        self.fibs[-1]["replaced_at"] = f["confirm_at"]
                    self.fibs.append(f)
                    born.append(f)
            insort(self._bars[m["side"]], bar)
            self._price[(m["side"], bar)] = m["price"]
        return born

    def result(self) -> dict:
        return {"fibs": self.fibs}
