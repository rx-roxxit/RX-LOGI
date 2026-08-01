"""Swing High/Low — candle-closure (mq5/RX_01_Swing.mq5, TTrades framing).

State machine (causal, no repaint):
  mode +1 (hunting a Swing High): track the highest-high bar; a bar that
    CLOSES below that bar's low confirms the Swing High -> flip to mode -1.
  mode -1 (hunting a Swing Low): track the lowest-low bar; a bar that
    CLOSES above that bar's high confirms the Swing Low -> flip to mode +1.

Display: red down-arrow = swing high (text above), blue up-arrow = swing low
(text below), classed HH/LH vs the previous swing high and HL/LL vs the
previous swing low (first swing of a side has no basis -> "H"/"L").
"""
from __future__ import annotations


class SwingSM:
    """The ONE swing state machine. Every other feature imports this —
    a second implementation of "swing" anywhere is a bug by definition."""

    def __init__(self, high, low, close):
        self.h, self.l, self.c = high, low, close
        self.mode = 1
        self.hi_bar = 0
        self.lo_bar = 0

    def update(self, i: int) -> dict | None:
        """Feed CLOSED bar i (in order). Returns a confirmed swing or None."""
        if i < 1:
            return None
        if self.mode == 1:
            if self.h[i] >= self.h[self.hi_bar]:
                self.hi_bar = i
            if self.c[i] < self.l[self.hi_bar] and self.hi_bar < i:
                ev = {"bar": self.hi_bar, "confirm": i, "side": "high",
                      "price": float(self.h[self.hi_bar])}
                self.mode = -1
                self.lo_bar = i
                return ev
        else:
            if self.l[i] <= self.l[self.lo_bar]:
                self.lo_bar = i
            if self.c[i] > self.h[self.lo_bar] and self.lo_bar < i:
                ev = {"bar": self.lo_bar, "confirm": i, "side": "low",
                      "price": float(self.l[self.lo_bar])}
                self.mode = 1
                self.hi_bar = i
                return ev
        return None


def swing_events(h, l, c) -> list[dict]:
    """All confirmed swings, classified HH/LH (highs) and HL/LL (lows)."""
    sm = SwingSM(h, l, c)
    out: list[dict] = []
    prev_hi: float | None = None
    prev_lo: float | None = None
    for i in range(len(c)):
        ev = sm.update(i)
        if ev is None:
            continue
        if ev["side"] == "high":
            ev["cls"] = "H" if prev_hi is None else ("HH" if ev["price"] > prev_hi else "LH")
            prev_hi = ev["price"]
        else:
            ev["cls"] = "L" if prev_lo is None else ("HL" if ev["price"] > prev_lo else "LL")
            prev_lo = ev["price"]
        out.append(ev)
    return out


def compute(dt, o, h, l, c) -> dict:
    markers = [{"at": dt[e["bar"]], "confirm_at": dt[e["confirm"]],
                "side": e["side"], "price": e["price"], "cls": e["cls"]}
               for e in swing_events(h, l, c)]
    return {"markers": markers}
