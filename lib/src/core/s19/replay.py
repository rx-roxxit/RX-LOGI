"""Feed closed bars to a live engine the way the market would have.

merged_stream concatenates htf, mtf and path and stable-sorts by knowability, so
tokens that become knowable at the same instant come out coarse-first. A live
engine appending one bar at a time only reproduces that if the bars arrive in
close-time order and ties are served coarse-first. That is the whole contract,
and it lives here so replays cannot get it wrong by accident.

The platform must do the same thing with real bars - it is written down in the
P4 handoff.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from .stage2 import TF_MIN          # labels.TF_MIN stops at 15m; this one has 3m/5m

TF_ORDER = ("D", "4H", "1H", "30m", "15m", "5m", "3m")     # coarse first


def replay_feed(bars_by_tf: dict) -> Iterator[list[tuple[str, tuple]]]:
    """Yield one GROUP per close instant: every bar that closed at that moment,
    across every timeframe, coarse-first inside the group.

    The group, not the bar, is the unit. A plan made at known_ts may see
    everything that closed AT known_ts - enumerate_candidates rejects only
    `born_ts > known_ts`, and closed_index is right-inclusive for the same
    reason. At 08:30 the 3m, 5m, 15m and 30m bars all close together and the
    path timeframe sits in the middle of any per-bar ordering, so no ordering of
    single bars can be correct: the engine has to take them all, then decide.

    `bars_by_tf[tf] = (dt, o, h, l, c)`.
    """
    rank = {tf: i for i, tf in enumerate(TF_ORDER)}
    idx_of = {}
    rows = []
    for tf, (dt, o, h, l, c) in bars_by_tf.items():
        idx = pd.DatetimeIndex(dt)
        idx_of[tf] = idx
        close = idx + pd.Timedelta(minutes=TF_MIN[tf])
        r = rank[tf]
        for k in range(len(idx)):
            rows.append((close[k], r, tf, k))
    rows.sort(key=lambda t: (t[0], t[1]))
    group: list[tuple[str, tuple]] = []
    cur = None
    for close, _r, tf, k in rows:
        if cur is not None and close != cur:
            yield group
            group = []
        cur = close
        _dt, o, h, l, c = bars_by_tf[tf]
        group.append((tf, (idx_of[tf][k], float(o[k]), float(h[k]), float(l[k]),
                           float(c[k]))))
    if group:
        yield group
