"""ERL / IRL — External vs Internal Range Liquidity (mq5/RX_02_ERL_IRL.mq5).

Rules (exactly the ref):
  trend = direction of the LAST ERL break (+1 broke up, -1 broke down).
  Swing high breaking erlHi:
    trend==+1 -> BOS (continuation): mark ERL High AND promote the LOWEST
                 IRL low since the previous ERL to ERL Low.
    trend==-1 -> CHoCH (counter): mark ERL High only, no promotion.
  Swing low breaking erlLo: mirror (BOS down promotes the HIGHEST IRL high).
  Swings inside the range = IRL.

Promotion is retroactive-but-honest: the marker keeps its origin bar and
gains promote_at = the break bar that upgraded it; replay shows it as IRL
until that bar, ERL after.

Display: red/blue arrows like `swing`, text "ERL-<cls>" / "IRL-<cls>".
"""
from __future__ import annotations

from .swing import swing_events


def events(h, l, c) -> list[dict]:
    """Raw index-based ERL/IRL stream (bar/confirm/promote as indices) —
    consumed by compute() for display and by fib.py for leg pairing."""
    trend = 0
    has_hi = has_lo = False
    erl_hi = erl_lo = 0.0
    lo_irl = hi_irl = None          # index into out[] of the extreme IRL so far
    out: list[dict] = []

    for ev in swing_events(h, l, c):
        m = {"bar": ev["bar"], "confirm": ev["confirm"], "side": ev["side"],
             "price": ev["price"], "cls": ev["cls"], "role": "IRL", "promote": None}
        v = ev["price"]
        if ev["side"] == "high":
            if not has_hi:
                m["role"] = "ERL"; erl_hi = v; has_hi = True; trend = 1
            elif v > erl_hi:                       # broke the upper edge
                m["role"] = "ERL"; erl_hi = v
                if trend == 1 and lo_irl is not None:      # BOS: promote lowest IRL low
                    out[lo_irl]["promote"] = ev["confirm"]
                    erl_lo = out[lo_irl]["price"]; has_lo = True
                trend = 1; lo_irl = hi_irl = None
            else:                                  # inside the range = IRL
                if hi_irl is None or v > out[hi_irl]["price"]:
                    hi_irl = len(out)
        else:
            if not has_lo:
                m["role"] = "ERL"; erl_lo = v; has_lo = True; trend = -1
            elif v < erl_lo:                       # broke the lower edge
                m["role"] = "ERL"; erl_lo = v
                if trend == -1 and hi_irl is not None:     # BOS: promote highest IRL high
                    out[hi_irl]["promote"] = ev["confirm"]
                    erl_hi = out[hi_irl]["price"]; has_hi = True
                trend = -1; lo_irl = hi_irl = None
            else:
                if lo_irl is None or v < out[lo_irl]["price"]:
                    lo_irl = len(out)
        out.append(m)
    return out


def compute(dt, o, h, l, c) -> dict:
    markers = [{"at": dt[m["bar"]], "confirm_at": dt[m["confirm"]],
                "side": m["side"], "price": m["price"], "cls": m["cls"],
                "role": m["role"],
                "promote_at": dt[m["promote"]] if m["promote"] is not None else None}
               for m in events(h, l, c)]
    return {"markers": markers}
