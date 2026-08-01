"""Fibonacci on ERL expansion legs (USER spec 2026-07-20).

A leg = two consecutive OPPOSITE-side ERL swings where the NEW swing extends
the range: it ends in an ERL-HH (up leg) or ERL-LL (down leg) — exactly the
USER's pairs HL>HH · LH>LL · LL>HH · HH>LL. Legs ending LH/HL are not drawn.

Measured 1 -> 0, always unfolding left to right:
  1 = the leg's START (older swing) · 0 = the leg's END (newer swing)
  price(L) = end + L * (start - end)   -> 1.618/2.618 project beyond the start
  (matches the USER's reference image: on a down leg they sit above the high)

Promoted swings count as ERL from their promote bar (same knowability rule) —
but only as leg STARTS: promotion marks a range edge retroactively, it is not a
boundary break, so a promoted swing can never END an expansion leg (adversary-
confirmed 2026-07-20: its HH/LL class is a local comparison, not an ERL break —
allowing it produced phantom legs born already-replaced).
Every leg is kept with confirm_at + replaced_at so TRAINING sees the current
AND the previous legs (how deep price retraced before = context the USER wants);
the chart draws the current leg solid + the previous one dimmed.
"""
from __future__ import annotations

from .erl_irl import events

LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618)


def compute(dt, o, h, l, c) -> dict:
    stream = []                       # (known_idx, bar, marker) — ERL role only
    for m in events(h, l, c):
        if m["role"] == "ERL":
            stream.append((m["confirm"], m["bar"], m))
        elif m["promote"] is not None:
            stream.append((m["promote"], m["bar"], m))
    stream.sort(key=lambda t: (t[0], t[1]))

    fibs: list[dict] = []
    seen: list[dict] = []             # ERL swings, in knowability order
    for known, bar, m in stream:
        expands = m["role"] == "ERL" and (
            (m["side"] == "high" and m["cls"] == "HH") or
            (m["side"] == "low" and m["cls"] == "LL"))
        if expands:
            opp = "low" if m["side"] == "high" else "high"
            cand = [s for s in seen if s["side"] == opp and s["bar"] < bar]
            if cand:
                s = max(cand, key=lambda q: q["bar"])
                fibs.append({"start_at": dt[s["bar"]], "start_price": s["price"],
                             "end_at": dt[bar], "end_price": m["price"],
                             "dir": "up" if m["side"] == "high" else "down",
                             "confirm_at": dt[known], "replaced_at": None})
        seen.append({"bar": bar, "side": m["side"], "price": m["price"]})

    for cur, nxt in zip(fibs, fibs[1:]):
        cur["replaced_at"] = nxt["confirm_at"]
    return {"fibs": fibs}
