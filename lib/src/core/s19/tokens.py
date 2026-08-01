"""Event-sequence encoder — PREREG §4B.1.

One stream per cell: every event the ten frozen features emit, on ALL THREE
timeframes, ordered by WHEN IT BECAME KNOWN (not when it happened). A plan sees
the last SEQ_CAP tokens whose knowability instant is at or before its own.

Static per token (precomputed once): type one-hot, timeframe, size/quality,
clock+killzone of that token's own bar on its own timeframe.
Dynamic per plan (filled at slice time): distance from the plan's price
(range-normalised) and how long ago the token became known, in its own bars.

A token's knowability instant is `confirm_at + TF duration` — the bar has to
close before its event exists, the same rule the multi-TF wall enforces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .frozen import eqhl, erl_irl, fib, fvg, liquidity, ob, sweep, swing, trend
from .killzone import clock_features
from .labels import TF_MIN

TYPES = ["sw_HH", "sw_HL", "sw_LH", "sw_LL",
         "erl_up", "erl_dn", "promo_up", "promo_dn",
         "ebos_up", "ebos_dn", "emss_up", "emss_dn",
         "ibos_up", "ibos_dn", "imss_up", "imss_dn",
         "sweep_ssl", "sweep_bsl", "eqh", "eql",
         "fvg_bull", "fvg_bear", "fvg_used",
         "ob_bull", "ob_bear", "ob_used",
         "liq_swept_ssl", "liq_swept_bsl", "fib_up", "fib_dn"]
TID = {t: i for i, t in enumerate(TYPES)}
SEQ_CAP = 96
N_STATIC = len(TYPES) + 3 + 1 + 8          # type, tf one-hot, size, clock
N_TOKEN = N_STATIC + 2                      # + dist, age


def _rows(dt, o, h, l, c) -> list[tuple]:
    """(known_at, type, level, size) — size is a type-specific magnitude, 0 when
    the event has none (kept in one slot so the vocabulary stays flat)."""
    out: list[tuple] = []
    add = lambda k, t, lv, sz=0.0: out.append((np.datetime64(k, "ns"), TID[t], float(lv), float(sz)))

    for m in swing.compute(dt, o, h, l, c)["markers"]:
        if m["cls"] in ("HH", "HL", "LH", "LL"):
            add(m["confirm_at"], f"sw_{m['cls']}", m["price"])
    for m in erl_irl.compute(dt, o, h, l, c)["markers"]:
        if m["role"] == "ERL":
            add(m["confirm_at"], "erl_up" if m["side"] == "high" else "erl_dn", m["price"])
        if m["promote_at"]:
            add(m["promote_at"], "promo_up" if m["side"] == "high" else "promo_dn", m["price"])
    for lab in trend.compute(dt, o, h, l, c)["labels"]:
        key = {"E-BOS": "ebos", "E-MSS": "emss", "i-BOS": "ibos", "i-MSS": "imss"}[lab["kind"]]
        add(lab["confirm_at"], f"{key}_{'up' if lab['dir'] == 'up' else 'dn'}", lab["level"])
    for m in sweep.compute(dt, o, h, l, c)["markers"]:
        add(m["confirm_at"], "sweep_ssl" if m["side"] == "SSL" else "sweep_bsl", m["price"])
    for e in eqhl.compute(dt, o, h, l, c)["events"]:
        add(e["confirm_at"], "eqh" if e["kind"] == "EQH" else "eql", (e["p1"] + e["p2"]) / 2)
    for b in fvg.compute(dt, o, h, l, c)["boxes"]:
        add(b["confirm_at"], "fvg_bull" if b["side"] == "bull" else "fvg_bear",
            (b["lo"] + b["hi"]) / 2, abs(b["hi"] - b["lo"]))
        if b["mit_at"]:
            add(b["mit_at"], "fvg_used", (b["lo"] + b["hi"]) / 2)
    for b in ob.compute(dt, o, h, l, c)["boxes"]:
        add(b["confirm_at"], "ob_bull" if b["side"] == "bull" else "ob_bear",
            (b["lo"] + b["hi"]) / 2, abs(b.get("gap", 0.0)))
        if b["mit_at"]:
            add(b["mit_at"], "ob_used", (b["lo"] + b["hi"]) / 2)
    for q in liquidity.compute(dt, o, h, l, c)["lines"]:
        if q["swept_at"]:
            add(q["swept_at"], "liq_swept_ssl" if q["side"] == "SSL" else "liq_swept_bsl",
                q["level"])
    for f in fib.compute(dt, o, h, l, c)["fibs"]:
        add(f["confirm_at"], "fib_up" if f["dir"] == "up" else "fib_dn", f["end_price"],
            abs(f["start_price"] - f["end_price"]))
    return out


def tf_tokens(dt, o, h, l, c, tf: str, tf_slot: int) -> dict:
    """Static token matrix for one timeframe, sorted by knowability.

    The nine features are polled per bar in streams/tokens.py rather than each
    being run over the whole tape; the tie-break order inside a bar is the same
    _rows order, so the array is identical (walled against _legacy).
    """
    from .store import FeatureStore
    st = FeatureStore(tf, {"tokens"})
    st.extend(dt, o, h, l, c)
    return st.tokens(tf_slot)


def merged_stream(cell_tfs: tuple[str, str, str], tapes: dict) -> dict:
    """All three timeframes in one knowability-ordered stream."""
    parts = [tf_tokens(*tapes[tf], tf, i) for i, tf in enumerate(cell_tfs)]
    known = np.concatenate([p["known"] for p in parts])
    level = np.concatenate([p["level"] for p in parts])
    static = np.concatenate([p["static"] for p in parts], axis=0)
    tfmin = np.concatenate([np.full(len(p["known"]), p["tf_min"]) for p in parts])
    order = np.argsort(known, kind="stable")
    return {"known": known[order], "level": level[order],
            "static": static[order], "tf_min": tfmin[order]}


def sequences(stream: dict, plan_known_ts: pd.DatetimeIndex,
              ref_price: np.ndarray, range_width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_plans, SEQ_CAP, N_TOKEN) + valid-length. Oldest-first, padded at the front.

    Only tokens with known <= the plan's own knowability instant are visible —
    the whole causal contract of the sequence lives in this searchsorted.
    """
    kts = pd.DatetimeIndex(plan_known_ts).to_numpy().astype("datetime64[ns]")
    hi = np.searchsorted(stream["known"], kts, side="right")
    n = len(kts)
    seq = np.zeros((n, SEQ_CAP, N_TOKEN), np.float32)
    lens = np.zeros(n, np.int32)
    width = np.where(np.isfinite(range_width) & (range_width > 0), range_width, np.nan)
    for i in range(n):
        a = max(0, hi[i] - SEQ_CAP)
        b = hi[i]
        m = b - a
        if m <= 0:
            continue
        lens[i] = m
        sl = slice(SEQ_CAP - m, SEQ_CAP)
        seq[i, sl, :N_STATIC] = stream["static"][a:b]
        dist = (stream["level"][a:b] - ref_price[i]) / (width[i] if width[i] == width[i] else 1.0)
        age = (kts[i] - stream["known"][a:b]) / np.timedelta64(1, "m") / stream["tf_min"][a:b]
        seq[i, sl, N_STATIC] = np.clip(dist, -20, 20)
        seq[i, sl, N_STATIC + 1] = np.log1p(np.clip(age, 0, None)) / 5.0
    seq[~np.isfinite(seq)] = 0.0
    return seq, lens
