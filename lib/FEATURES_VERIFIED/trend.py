"""Market-structure trend, EXTERNAL vs INTERNAL (mq5/RX_03_Trend.mq5)
+ USER 2026-07-20: every BOS/MSS carries a horizontal LINE at the broken level,
drawn from the bar that made the old high/low to the bar that broke it.

Two independent 3-state machines:
  EXTERNAL — breaks of ERL (range edges): E-BOS / E-MSS
  INTERNAL — CLOSES through IRL inside the range: i-BOS / i-MSS
States: BULL / BEAR / WAITING (USER renamed RANGE -> WAITING).
  MSS = counter-trend break -> WAITING (needs a BOS to confirm)
  BOS = with-trend or from-WAITING break -> confirms BULL/BEAR.
Internal break only counts while price is still inside the ERL range.
An external break starts a new leg -> internal machine resets to WAITING.

Label fields: kind, dir, level = the BROKEN level, from_at = origin bar of the
broken high/low, at = the breaking bar (external: the new swing bar whose high/
low pierced the ERL · internal: the bar whose close pierced the IRL),
confirm_at = when it became knowable.
"""
from __future__ import annotations

from .swing import SwingSM

_ST = {1: "BULL", -1: "BEAR", 0: "WAITING"}


def compute(dt, o, h, l, c) -> dict:
    sm = SwingSM(h, l, c)
    # ERL/IRL promotion state (RX_02, embedded exactly as the ref does) + origin bars
    erl_tr = 0
    has_hi = has_lo = False
    erl_hi = erl_lo = 0.0
    erl_hi_bar = erl_lo_bar = 0
    has_lo_irl = has_hi_irl = False
    lo_irl_v = hi_irl_v = 0.0
    lo_irl_bar = hi_irl_bar = 0
    # external / internal trend machines
    ext = 0
    int_ = 0
    has_int_hi = has_int_lo = False
    int_hi_broken = int_lo_broken = False
    int_hi = int_lo = 0.0
    int_hi_bar = int_lo_bar = 0

    labels: list[dict] = []
    states: list[dict] = []      # change log for the corner chip

    def note_state(i: int) -> None:
        cur = {"confirm_at": dt[i], "ext": _ST[ext], "int": _ST[int_]}
        if not states or states[-1]["ext"] != cur["ext"] or states[-1]["int"] != cur["int"]:
            states.append(cur)

    def reset_internal() -> None:
        nonlocal has_int_hi, has_int_lo, int_hi_broken, int_lo_broken, int_
        has_int_hi = has_int_lo = False
        int_hi_broken = int_lo_broken = False
        int_ = 0

    n = len(c)
    for i in range(n):
        ev = sm.update(i)
        if ev is not None and ev["side"] == "high":
            hb, v = ev["bar"], ev["price"]
            if not has_hi:
                erl_hi = v; erl_hi_bar = hb; has_hi = True; erl_tr = 1
            elif v > erl_hi:                                   # EXTERNAL up-break
                mss = ext == -1
                labels.append({"at": dt[hb], "confirm_at": dt[i],
                               "level": float(erl_hi), "from_at": dt[erl_hi_bar],
                               "kind": "E-MSS" if mss else "E-BOS", "dir": "up"})
                ext = 0 if mss else 1
                erl_hi = v; erl_hi_bar = hb
                if erl_tr == 1 and has_lo_irl:
                    erl_lo = lo_irl_v; erl_lo_bar = lo_irl_bar; has_lo = True
                erl_tr = 1; has_lo_irl = has_hi_irl = False
                reset_internal()
            else:                                              # IRL high
                if not has_hi_irl or v > hi_irl_v:
                    hi_irl_v = v; hi_irl_bar = hb; has_hi_irl = True
                int_hi = v; int_hi_bar = hb; has_int_hi = True; int_hi_broken = False
        elif ev is not None:
            lb, v = ev["bar"], ev["price"]
            if not has_lo:
                erl_lo = v; erl_lo_bar = lb; has_lo = True; erl_tr = -1
            elif v < erl_lo:                                   # EXTERNAL down-break
                mss = ext == 1
                labels.append({"at": dt[lb], "confirm_at": dt[i],
                               "level": float(erl_lo), "from_at": dt[erl_lo_bar],
                               "kind": "E-MSS" if mss else "E-BOS", "dir": "down"})
                ext = 0 if mss else -1
                erl_lo = v; erl_lo_bar = lb
                if erl_tr == -1 and has_hi_irl:
                    erl_hi = hi_irl_v; erl_hi_bar = hi_irl_bar; has_hi = True
                erl_tr = -1; has_lo_irl = has_hi_irl = False
                reset_internal()
            else:                                              # IRL low
                if not has_lo_irl or v < lo_irl_v:
                    lo_irl_v = v; lo_irl_bar = lb; has_lo_irl = True
                int_lo = v; int_lo_bar = lb; has_int_lo = True; int_lo_broken = False

        # INTERNAL BOS/MSS — close through IRL, still inside the ERL range
        if has_int_hi and not int_hi_broken and c[i] > int_hi and (not has_hi or c[i] < erl_hi):
            mss = int_ == -1
            labels.append({"at": dt[i], "confirm_at": dt[i],
                           "level": float(int_hi), "from_at": dt[int_hi_bar],
                           "kind": "i-MSS" if mss else "i-BOS", "dir": "up"})
            int_ = 0 if mss else 1
            int_hi_broken = True
        if has_int_lo and not int_lo_broken and c[i] < int_lo and (not has_lo or c[i] > erl_lo):
            mss = int_ == 1
            labels.append({"at": dt[i], "confirm_at": dt[i],
                           "level": float(int_lo), "from_at": dt[int_lo_bar],
                           "kind": "i-MSS" if mss else "i-BOS", "dir": "down"})
            int_ = 0 if mss else -1
            int_lo_broken = True

        note_state(i)

    return {"labels": labels, "states": states}
