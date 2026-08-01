"""Liquidity levels (mq5/RX_07_Liquidity.mq5) with USER naming (2026-07-20):

  BSL = stops of BUY holders  -> rest BELOW swing lows  -> green dashed line
  SSL = stops of SELL holders -> rest ABOVE swing highs -> red dashed line
  (the ref file's labels used the opposite naming; USER's definition wins)

Every confirmed swing spawns a level extending right until swept by a WICK
(SSL: high > level / BSL: low < level) -> line cut at the sweep bar + dimmed.
Active lines = liquidity not yet taken (DOL targets).

Display: dashed green/red line from the swing bar, type text at the right end.
"""
from __future__ import annotations

from .swing import swing_events


def compute(dt, o, h, l, c) -> dict:
    lines: list[dict] = []
    active: list[int] = []                 # indices into lines[]
    events = swing_events(h, l, c)
    by_confirm: dict[int, list[dict]] = {}
    for ev in events:
        by_confirm.setdefault(ev["confirm"], []).append(ev)

    n = len(c)
    for i in range(n):
        for ev in by_confirm.get(i, ()):   # level exists once its swing confirms
            lines.append({"at": dt[ev["bar"]], "confirm_at": dt[i],
                          "level": ev["price"],
                          "side": "SSL" if ev["side"] == "high" else "BSL",
                          "swept_at": None})
            active.append(len(lines) - 1)
        for k in range(len(active) - 1, -1, -1):
            q = lines[active[k]]
            swept = (h[i] > q["level"]) if q["side"] == "SSL" else (l[i] < q["level"])
            if swept:
                q["swept_at"] = dt[i]
                active.pop(k)
    return {"lines": lines}
