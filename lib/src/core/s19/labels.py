"""S19 stage-1 label construction — PREREG_S19_ERL_CHAIN §2.

Everything here is derived from the frozen ERL/IRL feature; no new market
definition is invented.

  break stream   every ERL break (role=='ERL'). Structurally these are always
                 HH on the high side and LL on the low side: a swing that did
                 not exceed every recent same-side swing could not have crossed
                 the range edge in the first place.
  range edges    erl_hi / erl_lo INCLUDING promotions — a promoted IRL becomes
                 an edge at its promote bar, not at its own confirm bar. Missing
                 this makes every stage-2 band wrong (caught while writing the
                 pre-registration; see §2.5 worked examples).
  plan           made when a break confirms; concerns the next break (+ the one
                 after, when the next one reverses).
  walk           the USER's trigger rule ("plan only when the next event is not
                 already planned"): plans do not overlap — a plan owns the next
                 `depth` breaks, and the following plan starts where it ends.
                 Every break still gets a row (at serving time a wrong plan is
                 re-planned immediately, so the model must be able to plan at any
                 break) but `walk=True` marks the non-overlapping serving cadence,
                 which is what display and evaluation use.
                   A  continue        1 break   HL > HH
                   B  sweep+continue  2 breaks  LL > HH
                   C  reversal (MSS)  2 breaks  LL > LH > LL
"""
from __future__ import annotations

import numpy as np

from .frozen import erl_events

TF_MIN = {"15m": 15, "30m": 30, "1H": 60, "4H": 240, "D": 1440}
CLASSES = ("A", "B", "C")
CLASS_NAME = {"A": "BOS-continue", "B": "sweep-continue", "C": "MSS-reversal"}


def edge_stream(h, l, c) -> list[tuple]:
    """[(known_bar, side, price, origin_bar, kind)] — when each range edge takes
    effect. 'break' edges take effect at their confirm bar, 'promote' edges at
    the BOS bar that promoted them."""
    out = []
    for m in erl_events(h, l, c):
        if m["role"] == "ERL":
            out.append((m["confirm"], m["side"], m["price"], m["bar"], "break"))
        elif m["promote"] is not None:
            out.append((m["promote"], m["side"], m["price"], m["bar"], "promote"))
    out.sort(key=lambda t: (t[0], t[3]))
    return out


def breaks(h, l, c) -> list[dict]:
    """ERL breaks in confirmation order (the plan instants)."""
    return [{"dir": "U" if m["side"] == "high" else "D", "price": float(m["price"]),
             "bar": m["bar"], "confirm": m["confirm"]}
            for m in erl_events(h, l, c) if m["role"] == "ERL"]


def _edges_upto(stream, k, cursor=0):
    """Latest (hi, lo) edges known at bar k, scanning forward from `cursor`."""
    hi = lo = None
    i = cursor
    while i < len(stream) and stream[i][0] <= k:
        _, side, price, origin, kind = stream[i]
        if side == "high":
            hi = (price, origin, kind)
        else:
            lo = (price, origin, kind)
        i += 1
    return hi, lo, i


def build_plans(dt, o, h, l, c) -> list[dict]:
    """One row per plan instant. dt = array-like of timestamps (index-aligned)."""
    brk = breaks(h, l, c)
    stream = edge_stream(h, l, c)
    plans: list[dict] = []
    hi = lo = None
    cur = 0
    for i in range(len(brk) - 2):
        b0, b1, b2 = brk[i], brk[i + 1], brk[i + 2]
        k = b0["confirm"]
        nhi, nlo, cur = _edges_upto(stream, k, cur)
        hi = nhi or hi
        lo = nlo or lo
        if hi is None or lo is None:
            continue
        if b1["dir"] == b0["dir"]:
            cls, depth = "A", 1
        elif b2["dir"] == b0["dir"]:
            cls, depth = "B", 2
        else:
            cls, depth = "C", 2
        up = b0["dir"] == "U"
        # leg-1 admissible band for stage-2 (PREREG §3.1)
        if cls == "A":
            band_lo, band_hi = lo[0], hi[0]           # inside the range
        elif up:
            band_lo, band_hi = float("-inf"), lo[0]   # sweep below the low edge
        else:
            band_lo, band_hi = hi[0], float("inf")    # sweep above the high edge
        plans.append({
            "i": i,
            "known_at": dt[k], "known_bar": k,
            "state": "HH" if up else "LL",
            "state_price": b0["price"], "state_bar": b0["bar"],
            "cls": cls, "depth": depth,
            "chain": ("HL > HH" if cls == "A" and up else "LH > LL" if cls == "A" else
                      "LL > HH" if cls == "B" and up else "HH > LL" if cls == "B" else
                      "LL > LH > LL" if up else "HH > HL > HH"),
            "erl_hi": hi[0], "erl_hi_bar": hi[1], "erl_hi_kind": hi[2],
            "erl_lo": lo[0], "erl_lo_bar": lo[1], "erl_lo_kind": lo[2],
            "band_lo": band_lo, "band_hi": band_hi,
            # realized path (the label's evidence — display + scoring only)
            "b1_dir": b1["dir"], "b1_price": b1["price"], "b1_at": dt[b1["bar"]],
            "b1_known_at": dt[b1["confirm"]],
            "b2_dir": b2["dir"], "b2_price": b2["price"], "b2_at": dt[b2["bar"]],
            "b2_known_at": dt[b2["confirm"]],
            "done_known_at": dt[b1["confirm"] if depth == 1 else b2["confirm"]],
            # the two decision levels, split for honest metrics (PREREG §2.3)
            "lvl1": "continue" if cls == "A" else "reverse",
            "lvl2": None if cls == "A" else cls,
            # boundary whose violation would kill the plan on leg 1
            "invalidation": lo[0] if (cls == "A") == up else hi[0],
        })
    _mark_walk(plans)
    return plans


def _mark_walk(plans: list[dict]) -> None:
    """Non-overlapping serving cadence: a plan owns the next `depth` breaks."""
    by_i = {p["i"]: p for p in plans}
    for p in plans:
        p["walk"] = False
    nxt = min(by_i) if by_i else 0
    while nxt in by_i:
        p = by_i[nxt]
        p["walk"] = True
        nxt += p["depth"]


def class_mix(plans) -> dict:
    n = len(plans) or 1
    return {c: sum(p["cls"] == c for p in plans) / n for c in CLASSES}
