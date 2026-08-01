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
    """(n_bars, 14) running zone picture — only objects alive at that bar.

    The per-bar loop this used to run now lives in streams/zone_ctx.py, where the
    live set is kept level-sorted instead of rebuilt; the answer is unchanged
    element for element (walled in tests/test_s19_store.py against _legacy).
    """
    from .store import FeatureStore
    st = FeatureStore("15m", {"zone_ctx"})       # tf only labels tokens; unused here
    st.extend(dt, o, h, l, c)
    return st.zone_ctx()


def fib_state(dt, o, h, l, c) -> tuple[np.ndarray, list[str]]:
    """(n_bars, 6) — where price sits on the live leg, plus how deep the previous
    three legs actually retraced (the context the USER asked for explicitly).

    The running leg and its extremes now live in streams/fib_ctx.py; the answer
    is unchanged element for element (walled against _legacy).
    """
    from .store import FeatureStore
    st = FeatureStore("15m", {"fib_ctx"})        # tf only labels tokens; unused here
    st.extend(dt, o, h, l, c)
    return st.fib_ctx()
