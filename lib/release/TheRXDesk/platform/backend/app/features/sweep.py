"""Liquidity Sweep — stop hunt (mq5/RX_09_Sweep.mq5) with USER naming:

  sweep SSL (above swing highs): high > level AND close <= level
      = grabbed the sell-holders' stops, closed back under -> mark v (red)
  sweep BSL (below swing lows):  low < level AND close >= level -> mark ^ (blue)
  Delayed reclaim: a close THROUGH the level (looks like a break) still
      counts as a sweep if price closes back within RECLAIM_BARS bars;
      otherwise it was a real break/BOS -> level removed, no mark.
  Level pool capped at the most recent MAX_SWINGS per side (ref behavior).

Display: red down-arrow above the sweep bar (SSL) / blue up-arrow below (BSL).
The mark sits on the bar that PROVED the sweep (the reclaim close), as the ref.
"""
from __future__ import annotations

from .swing import SwingSM

RECLAIM_BARS = 3        # ref InpReclaimBars
MAX_SWINGS = 80         # ref InpMaxSwings


def compute(dt, o, h, l, c) -> dict:
    sm = SwingSM(h, l, c)
    # per-side level pools: dicts with lvl, bar, brk (-1 = active, else close-through bar)
    ssl: list[dict] = []       # from swing highs (levels above)
    bsl: list[dict] = []       # from swing lows (levels below)
    markers: list[dict] = []
    n = len(c)
    for i in range(n):
        ev = sm.update(i)
        if ev is not None:
            pool = ssl if ev["side"] == "high" else bsl
            pool.append({"lvl": ev["price"], "bar": ev["bar"], "brk": -1})
            if len(pool) > MAX_SWINGS:
                del pool[0: len(pool) - MAX_SWINGS]

        for j in range(len(ssl) - 1, -1, -1):
            q = ssl[j]
            if q["brk"] < 0:
                if h[i] > q["lvl"]:
                    if c[i] <= q["lvl"]:
                        markers.append({"at": dt[i], "confirm_at": dt[i], "side": "SSL",
                                        "price": float(h[i]), "level": q["lvl"],
                                        "level_at": dt[q["bar"]]})
                        del ssl[j]
                    else:
                        q["brk"] = i                       # pending reclaim
            else:
                if c[i] <= q["lvl"]:
                    markers.append({"at": dt[i], "confirm_at": dt[i], "side": "SSL",
                                    "price": float(h[i]), "level": q["lvl"],
                                    "level_at": dt[q["bar"]]})
                    del ssl[j]
                elif i - q["brk"] >= RECLAIM_BARS:         # real break
                    del ssl[j]

        for j in range(len(bsl) - 1, -1, -1):
            q = bsl[j]
            if q["brk"] < 0:
                if l[i] < q["lvl"]:
                    if c[i] >= q["lvl"]:
                        markers.append({"at": dt[i], "confirm_at": dt[i], "side": "BSL",
                                        "price": float(l[i]), "level": q["lvl"],
                                        "level_at": dt[q["bar"]]})
                        del bsl[j]
                    else:
                        q["brk"] = i
            else:
                if c[i] >= q["lvl"]:
                    markers.append({"at": dt[i], "confirm_at": dt[i], "side": "BSL",
                                    "price": float(l[i]), "level": q["lvl"],
                                    "level_at": dt[q["bar"]]})
                    del bsl[j]
                elif i - q["brk"] >= RECLAIM_BARS:
                    del bsl[j]
    return {"markers": markers}
