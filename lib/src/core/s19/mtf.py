"""Multi-timeframe assembly — PREREG_S19 §4 (cells) + §4A (time) + §8.3 (wall).

The whole risk of multi-TF lives in ONE line: which higher-timeframe bar is a
path-TF instant allowed to see. The rule here is the project's hard-won one:

    a bar is knowable only after it CLOSES
    known_ts(plan)  = open(confirm bar on path TF) + path TF duration
    HTF bar j usable  <=>  open(j) + HTF duration <= known_ts(plan)

Everything the model sees from a higher timeframe is therefore computed from
bars[0 .. j] only, which is what makes the truncation wall (test_s19_walls.py)
able to go red instead of a comment promising it is fine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .context import fib_state, zone_state
from .frozen import erl_events
from .killzone import clock_features
from .labels import TF_MIN, edge_stream

# cell -> (htf, mtf, path); the path TF is where plans are made (PREREG §4)
CELLS = {"1H": ("D", "4H", "1H"), "30m": ("4H", "1H", "30m"), "15m": ("1H", "30m", "15m")}

CTX_FEATS = ["last_break_up", "last_break_dn", "bars_since_break",
             "range_pos", "range_width_rel", "dist_hi_rel", "dist_lo_rel",
             "hi_from_promote", "lo_from_promote", "n_breaks_recent"]


def closed_index(bar_dt: pd.DatetimeIndex, tf_minutes: float,
                 known_ts: pd.DatetimeIndex) -> np.ndarray:
    """Index of the last bar of this TF that had CLOSED at each known_ts (-1 = none)."""
    close_ts = pd.DatetimeIndex(bar_dt) + pd.Timedelta(minutes=tf_minutes)
    return np.searchsorted(close_ts.to_numpy(), pd.DatetimeIndex(known_ts).to_numpy(),
                           side="right") - 1


def tf_state(dt, o, h, l, c, tf: str) -> dict:
    """Per-bar running state of one timeframe, from the frozen features only."""
    brk = [(m["confirm"], "U" if m["side"] == "high" else "D", float(m["price"]))
           for m in erl_events(h, l, c) if m["role"] == "ERL"]
    stream = edge_stream(h, l, c)
    n = len(c)
    last_dir = np.zeros(n, np.int8)
    since = np.full(n, -1.0)
    n_recent = np.zeros(n, np.float32)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi_pro = np.zeros(n, np.float32)
    lo_pro = np.zeros(n, np.float32)
    bi = 0
    si = 0
    cur_dir, cur_bar = 0, -1
    cur_hi = cur_lo = np.nan
    cur_hip = cur_lop = 0.0
    stamps: list[int] = []
    for k in range(n):
        while bi < len(brk) and brk[bi][0] <= k:
            cur_dir = 1 if brk[bi][1] == "U" else -1
            cur_bar = brk[bi][0]
            stamps.append(cur_bar)
            bi += 1
        while si < len(stream) and stream[si][0] <= k:
            _, side, price, _, kind = stream[si]
            if side == "high":
                cur_hi, cur_hip = price, 1.0 if kind == "promote" else 0.0
            else:
                cur_lo, cur_lop = price, 1.0 if kind == "promote" else 0.0
            si += 1
        last_dir[k] = cur_dir
        since[k] = (k - cur_bar) if cur_bar >= 0 else -1.0
        while stamps and stamps[0] < k - 200:
            stamps.pop(0)
        n_recent[k] = len(stamps)
        hi[k], lo[k], hi_pro[k], lo_pro[k] = cur_hi, cur_lo, cur_hip, cur_lop
    zs, zn = zone_state(dt, o, h, l, c)          # PREREG 4B.2 — zones on EVERY tf
    fs, fn = fib_state(dt, o, h, l, c)
    return {"last_dir": last_dir, "since": since, "n_recent": n_recent,
            "hi": hi, "lo": lo, "hi_pro": hi_pro, "lo_pro": lo_pro,
            "zone": zs, "zone_names": zn, "fib": fs, "fib_names": fn,
            "close": np.asarray(c, float), "dt": pd.DatetimeIndex(dt), "tf": tf}


def block(state: dict, idx: np.ndarray, ref_price: np.ndarray) -> np.ndarray:
    """(n_plans, len(CTX_FEATS) + clock) block for one timeframe.

    `idx` must come from closed_index(); rows with idx < 0 (no closed bar yet)
    are all-zero rather than silently reaching back to bar 0.
    """
    ok = idx >= 0
    j = np.where(ok, idx, 0)
    hi, lo = state["hi"][j], state["lo"][j]
    width = hi - lo
    safe = np.where(np.isfinite(width) & (width > 0), width, np.nan)
    price = np.where(ref_price > 0, ref_price, np.nan)
    feats = np.stack([
        (state["last_dir"][j] > 0).astype(np.float32),
        (state["last_dir"][j] < 0).astype(np.float32),
        np.log1p(np.clip(state["since"][j], 0, None)),
        (price - lo) / safe,                       # where price sits in the range
        safe / price,                              # range width, price-relative
        (hi - price) / price,
        (price - lo) / price,
        state["hi_pro"][j], state["lo_pro"][j],
        state["n_recent"][j] / 50.0,
    ], axis=1).astype(np.float32)
    clock, _ = clock_features(state["dt"][j], TF_MIN[state["tf"]])
    out = np.concatenate([feats, clock, state["zone"][j], state["fib"][j]], axis=1)
    out[~np.isfinite(out)] = 0.0
    out[~ok] = 0.0
    return out


def assemble(cell: str, tapes: dict, plans: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Context matrix for one cell. `tapes[tf] = (dt, o, h, l, c)`.

    known_ts = plan's confirm-bar open + path TF duration (the bar must close
    before its own information exists), then every TF is read at its own last
    CLOSED bar at that instant.
    """
    htf, mtf, path = CELLS[cell]
    known_at = pd.DatetimeIndex([p["known_at"] for p in plans])
    known_ts = known_at + pd.Timedelta(minutes=TF_MIN[path])
    pdt = pd.DatetimeIndex(tapes[path][0])
    pidx = np.searchsorted(pdt.to_numpy(), known_at.to_numpy(), side="left")
    ref = np.asarray(tapes[path][4], float)[np.clip(pidx, 0, len(pdt) - 1)]

    blocks, names = [], []
    for tf in (htf, mtf, path):
        dt, o_, h, l, c = tapes[tf]
        st = tf_state(dt, o_, h, l, c, tf)
        idx = closed_index(pd.DatetimeIndex(dt), TF_MIN[tf], known_ts)
        blocks.append(block(st, idx, ref))
        _, clock_names = clock_features(pd.DatetimeIndex(dt[:1]), TF_MIN[tf])
        names += [f"{tf}:{n}" for n in CTX_FEATS + clock_names
                  + st["zone_names"] + st["fib_names"]]
    kz = __import__("src.core.s19.killzone", fromlist=["kz_onehot"]).kz_onehot(known_ts)
    blocks.append(kz)
    names += [f"plan:kz_{n}" for n in ("london", "newyork", "asia")]
    return np.concatenate(blocks, axis=1).astype(np.float32), names
