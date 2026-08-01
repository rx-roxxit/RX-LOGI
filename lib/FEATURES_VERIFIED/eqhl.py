"""EQH / EQL — equal highs & lows (USER hybrid spec 2026-07-20).

Swings: OUR candle-closure SwingSM (not LuxAlgo's rolling pivots).
Equality tolerance: LuxAlgo SMC's formula — |new - previous| < 0.1 x ATR(200),
evaluated with the ATR value at the new swing's CONFIRM bar.

★ DECISION (USER 2026-07-20): ATR is the ONLY indicator permitted in this
project, and only for this equality tolerance. ATR(200) implemented exactly as
pine's ta.atr: Wilder RMA of True Range (progressive running-mean seed while
fewer than 200 bars exist — causal, prefix-stable).

Each event connects two consecutive same-side swings (they may chain A≈B≈C).
Display: dotted line between the two swing points + EQH/EQL label.
"""
from __future__ import annotations

import numpy as np

from .swing import swing_events

ATR_LEN = 200
THRESH = 0.1


def atr_series(h, l, c) -> np.ndarray:
    h = np.asarray(h, np.float64); l = np.asarray(l, np.float64)
    c = np.asarray(c, np.float64)
    n = len(c)
    atr = np.empty(n)
    if n == 0:
        return atr
    run = h[0] - l[0]
    atr[0] = run
    for i in range(1, n):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        if i < ATR_LEN:
            run = (run * i + tr) / (i + 1)          # progressive seed (running mean)
            atr[i] = run
        else:
            atr[i] = (atr[i - 1] * (ATR_LEN - 1) + tr) / ATR_LEN
    return atr


def compute(dt, o, h, l, c) -> dict:
    atr = atr_series(h, l, c)
    out: list[dict] = []
    prev_hi: dict | None = None
    prev_lo: dict | None = None
    for ev in swing_events(h, l, c):
        i = ev["confirm"]
        cur = {"bar": ev["bar"], "price": ev["price"]}
        if ev["side"] == "high":
            if prev_hi is not None and abs(cur["price"] - prev_hi["price"]) < THRESH * atr[i]:
                out.append({"at": dt[prev_hi["bar"]], "at2": dt[cur["bar"]],
                            "p1": prev_hi["price"], "p2": cur["price"],
                            "kind": "EQH", "confirm_at": dt[i]})
            prev_hi = cur
        else:
            if prev_lo is not None and abs(cur["price"] - prev_lo["price"]) < THRESH * atr[i]:
                out.append({"at": dt[prev_lo["bar"]], "at2": dt[cur["bar"]],
                            "p1": prev_lo["price"], "p2": cur["price"],
                            "kind": "EQL", "confirm_at": dt[i]})
            prev_lo = cur
    return {"events": out}
