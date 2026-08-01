"""Killzones — the USER's table (2026-07-20) is the ONLY source. No external
ICT time concepts are added; nothing is looked up anywhere else.

    London    02:00 - 05:00
    New York  08:00 - 11:00
    Asia      19:00 - 22:00

Clock: America/New_York, DST-aware — the convention this project already uses
for sessions. A fixed UTC-5 reading would slide the windows one hour off the
real New York trading clock for ~8 months a year (PREREG S19 §4A.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KZ = (("london", 2.0, 5.0), ("newyork", 8.0, 11.0), ("asia", 19.0, 22.0))
KZ_NAMES = [k[0] for k in KZ]
TZ = "America/New_York"


def _ny(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.tz_convert(TZ)


def ny_hour(idx: pd.DatetimeIndex) -> np.ndarray:
    loc = _ny(idx)
    return loc.hour.to_numpy(float) + loc.minute.to_numpy(float) / 60.0


def ny_dow(idx: pd.DatetimeIndex) -> np.ndarray:
    return _ny(idx).dayofweek.to_numpy(float)


def kz_onehot(idx: pd.DatetimeIndex) -> np.ndarray:
    """(n, 3) membership of the instant itself — used at the plan instant."""
    h = ny_hour(idx)
    return np.stack([((h >= a) & (h < b)).astype(np.float32) for _, a, b in KZ], axis=1)


def kz_overlap(idx: pd.DatetimeIndex, tf_minutes: float) -> np.ndarray:
    """(n, 3) fraction of the BAR [open, open+tf) that lies inside each window.

    One definition for every timeframe (PREREG §4A.1): ~0/1 on 15m-1H, fractional
    on 4H, constant on Daily (3h/24h per window — degenerate by construction, and
    harmless). Windows are taken on the NY clock, so a bar spanning a DST switch
    is measured against local wall-clock hours, which is what a trader reads.
    """
    start = ny_hour(idx)
    dur = tf_minutes / 60.0
    end = start + dur
    out = np.zeros((len(start), len(KZ)), np.float32)
    for j, (_, a, b) in enumerate(KZ):
        total = np.zeros_like(start)
        # a bar may wrap past midnight: measure against this window and its +/-24h copies
        for shift in (-24.0, 0.0, 24.0):
            lo, hi = a + shift, b + shift
            total += np.clip(np.minimum(end, hi) - np.maximum(start, lo), 0, None)
        out[:, j] = total / dur
    return np.clip(out, 0.0, 1.0)


def clock_features(idx: pd.DatetimeIndex, tf_minutes: float) -> tuple[np.ndarray, list[str]]:
    """Cyclical clock + killzone overlap + position-in-day, for ANY timeframe."""
    h = ny_hour(idx)
    d = ny_dow(idx)
    cols = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7), h / 24.0]
    names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "day_pos"]
    ov = kz_overlap(idx, tf_minutes)
    for j, name in enumerate(KZ_NAMES):
        cols.append(ov[:, j])
        names.append(f"kz_{name}")
    return np.stack(cols, axis=1).astype(np.float32), names
