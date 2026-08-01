"""Per-timeframe CONTEXT groups that mtf.py did not already cover — zone and fib
(PREREG §4B.2). Both are computed on EVERY timeframe of a cell, because a trader
reads zones and retracement depth on every frame they plan from (USER 2026-07-20).

Causality: every array here is a running per-bar state built forward in time, so
value[k] depends on bars <= k only. Zone/fib objects enter the state at their own
`confirm_at` bar and leave it at their `mit_at`/`swept_at` bar — the same
lifecycle rule the chart draws with, read from the frozen features.
"""
from __future__ import annotations

import numpy as np

from .frozen import eqhl, fib, fvg, liquidity, ob

ZONE_FEATS = ["n_virgin_above", "n_virgin_below", "dist_virgin_above", "dist_virgin_below",
              "n_pool_above", "n_pool_below", "dist_pool_above", "dist_pool_below",
              "has_eqh", "has_eql", "dist_eqh", "dist_eql",
              "virgin_age_above", "virgin_age_below"]
FIB_FEATS = ["fib_pos", "fib_dir_up", "fib_age", "retr_prev1", "retr_prev2", "retr_prev3"]


def _idx(dt) -> dict:
    """timestamp -> bar index. Feature modules echo back whatever `dt` we passed
    in, so normalise both sides through datetime64[ns] rather than str()."""
    return {np.datetime64(t, "ns"): i for i, t in enumerate(np.asarray(dt, dtype="datetime64[ns]"))}


def _events(dt, o, h, l, c):
    """(born_bar, gone_bar|None, level, is_above_at_birth_side, kind) per object."""
    ix = _idx(dt)
    out = []
    for b in fvg.compute(dt, o, h, l, c)["boxes"]:
        out.append(("poi", ix[np.datetime64(b["confirm_at"], "ns")],
                    ix.get(np.datetime64(b["mit_at"], "ns")) if b["mit_at"] else None,
                    (b["lo"] + b["hi"]) / 2.0))
    for b in ob.compute(dt, o, h, l, c)["boxes"]:
        out.append(("poi", ix[np.datetime64(b["confirm_at"], "ns")],
                    ix.get(np.datetime64(b["mit_at"], "ns")) if b["mit_at"] else None,
                    (b["lo"] + b["hi"]) / 2.0))
    for q in liquidity.compute(dt, o, h, l, c)["lines"]:
        out.append(("pool", ix[np.datetime64(q["confirm_at"], "ns")],
                    ix.get(np.datetime64(q["swept_at"], "ns")) if q["swept_at"] else None,
                    q["level"]))
    for e in eqhl.compute(dt, o, h, l, c)["events"]:
        out.append(("eqh" if e["kind"] == "EQH" else "eql",
                    ix[np.datetime64(e["confirm_at"], "ns")], None, (e["p1"] + e["p2"]) / 2.0))
    return out


def zone_state(dt, o, h, l, c) -> tuple[np.ndarray, list[str]]:
    """(n_bars, 14) running zone picture — only objects alive at that bar."""
    n = len(c)
    ev = _events(dt, o, h, l, c)
    born: dict[int, list] = {}
    gone: dict[int, list] = {}
    for kind, b, g, lvl in ev:
        born.setdefault(b, []).append((kind, lvl, b))
        if g is not None:
            gone.setdefault(g, []).append((kind, lvl, b))
    live: list[tuple] = []
    out = np.zeros((n, len(ZONE_FEATS)), np.float32)
    price = np.asarray(c, float)
    for k in range(n):
        for x in born.get(k, ()):
            live.append(x)
        for x in gone.get(k, ()):
            if x in live:
                live.remove(x)
        p = price[k]
        scale = max(p, 1e-9)
        above = [(lvl, kind, b) for kind, lvl, b in live if lvl > p]
        below = [(lvl, kind, b) for kind, lvl, b in live if lvl <= p]
        poi_a = [x for x in above if x[1] == "poi"]
        poi_b = [x for x in below if x[1] == "poi"]
        pool_a = [x for x in above if x[1] == "pool"]
        pool_b = [x for x in below if x[1] == "pool"]
        eqh = [x for x in above if x[1] == "eqh"] + [x for x in below if x[1] == "eqh"]
        eql = [x for x in above if x[1] == "eql"] + [x for x in below if x[1] == "eql"]
        near = lambda xs, up: (min(x[0] for x in xs) - p) / scale if (xs and up) else \
                              ((p - max(x[0] for x in xs)) / scale if xs else 0.0)
        age = lambda xs: (k - max(x[2] for x in xs)) / 100.0 if xs else 0.0
        out[k] = [len(poi_a) / 10.0, len(poi_b) / 10.0, near(poi_a, True), near(poi_b, False),
                  len(pool_a) / 10.0, len(pool_b) / 10.0, near(pool_a, True), near(pool_b, False),
                  1.0 if eqh else 0.0, 1.0 if eql else 0.0,
                  near(eqh, True) if eqh else 0.0, near(eql, False) if eql else 0.0,
                  age(poi_a), age(poi_b)]
    return out, ZONE_FEATS


def fib_state(dt, o, h, l, c) -> tuple[np.ndarray, list[str]]:
    """(n_bars, 6) — where price sits on the live leg, plus how deep the previous
    three legs actually retraced (the context the USER asked for explicitly)."""
    n = len(c)
    ix = _idx(dt)
    legs = fib.compute(dt, o, h, l, c)["fibs"]
    born: dict[int, list] = {}
    for f in legs:
        born.setdefault(ix[np.datetime64(f["confirm_at"], "ns")], []).append(f)
    out = np.zeros((n, len(FIB_FEATS)), np.float32)
    price = np.asarray(c, float)
    cur = None
    cur_from = 0
    depths: list[float] = []           # realized retracement of each finished leg
    lo_run = hi_run = None
    for k in range(n):
        if k in born:
            if cur is not None:        # close the previous leg: how deep did it pull back?
                span = cur["start_price"] - cur["end_price"]
                if abs(span) > 1e-12:
                    ext = (hi_run - cur["end_price"]) / span if cur["dir"] == "down" \
                        else (cur["end_price"] - lo_run) / -span if span < 0 else 0.0
                    depths.append(float(np.clip(ext, -1.0, 3.0)))
            cur = born[k][-1]
            cur_from = k
            lo_run = hi_run = price[k]
        if cur is not None:
            lo_run = min(lo_run, float(l[k]))
            hi_run = max(hi_run, float(h[k]))
            span = cur["start_price"] - cur["end_price"]
            pos = (price[k] - cur["end_price"]) / span if abs(span) > 1e-12 else 0.0
            prev = (depths[-3:] + [0.0, 0.0, 0.0])[:3][::-1]
            out[k] = [float(np.clip(pos, -1.0, 3.0)),
                      1.0 if cur["dir"] == "up" else 0.0,
                      min((k - cur_from) / 100.0, 3.0), *prev]
    return out, FIB_FEATS
