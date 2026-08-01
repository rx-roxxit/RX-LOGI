"""ZoneCtxStream == context.zone_state, built one bar at a time.

The legacy loop rebuilds four filtered lists out of the whole live set on every
bar. Three of its four questions - how many above/below, which is nearest,
what is the global min/max - are bisect queries on a level-sorted multiset. Only
"which live POI on this side was created most recently" needs a walk, and POIs
are the smallest of the four categories (about 400 live on 15m against 3,000
total), because EQH/EQL never die and dominate the rest.

The EQ distance features look like they split at price but do not: near() is
called with up=True for EQH and up=False for EQL over the WHOLE live set, so
they are a global min and a global max.

Two legacy behaviours are preserved deliberately:
  * `live` is a MULTISET - the legacy list.remove() drops the first entry equal
    to (kind, level, birth), and equal entries do occur (an FVG and an OB with
    the same midpoint on the same bar);
  * within one bar births are added BEFORE deaths are removed, so a liquidity
    line swept on its own confirm bar is added and then dropped.
"""
from __future__ import annotations

import numpy as np

from ..context import ZONE_FEATS
from .multiset import SortedMultiset


class ZoneCtxStream:
    def __init__(self) -> None:
        self.poi = SortedMultiset()
        self.pool = SortedMultiset()
        self.eqh = SortedMultiset()
        self.eql = SortedMultiset()
        self._rows: list[list[float]] = []
        self._fvg_seen = 0
        self._ob_seen = 0
        self._liq_seen = 0
        self._eqhl_seen = 0
        self._fvg_key: list[tuple[float, int]] = []     # index-parallel with mirror.boxes
        self._ob_key: list[tuple[float, int]] = []
        self._liq_key: list[tuple[float, int]] = []

    @property
    def visits(self) -> int:
        return self.poi.visits + self.pool.visits + self.eqh.visits + self.eql.visits

    def feed(self, i: int, c, fvg, ob, liq, eqhl) -> None:
        # ── births first (legacy walks born[k] before gone[k]) ──────────────
        while self._fvg_seen < len(fvg.boxes):
            b = fvg.boxes[self._fvg_seen]
            key = ((b["lo"] + b["hi"]) / 2.0, i)
            self._fvg_key.append(key)
            self.poi.add(*key)
            self._fvg_seen += 1
        while self._ob_seen < len(ob.boxes):
            b = ob.boxes[self._ob_seen]
            key = ((b["lo"] + b["hi"]) / 2.0, i)
            self._ob_key.append(key)
            self.poi.add(*key)
            self._ob_seen += 1
        while self._liq_seen < len(liq.lines):
            q = liq.lines[self._liq_seen]
            key = (q["level"], i)
            self._liq_key.append(key)
            self.pool.add(*key)
            self._liq_seen += 1
        while self._eqhl_seen < len(eqhl.events):
            e = eqhl.events[self._eqhl_seen]
            lvl = (e["p1"] + e["p2"]) / 2.0
            (self.eqh if e["kind"] == "EQH" else self.eql).add(lvl, i)
            self._eqhl_seen += 1

        # ── deaths ──────────────────────────────────────────────────────────
        for k in fvg.died_at_bar():
            self.poi.discard(*self._fvg_key[k])
        for k in ob.died_at_bar():
            self.poi.discard(*self._ob_key[k])
        for k in liq.died_at_bar():
            self.pool.discard(*self._liq_key[k])
        # EQH/EQL never die (context._events passes gone=None for them)

        # ── the 14 features ─────────────────────────────────────────────────
        p = float(c[i])
        scale = p if p > 1e-9 else 1e-9
        j = self.poi.split(p)
        n_b, n_a = j, len(self.poi) - j
        near_a = (self.poi.nearest_above(j) - p) / scale if n_a else 0.0
        near_b = (p - self.poi.nearest_below(j)) / scale if n_b else 0.0
        age_a = (i - self.poi.max_birth(j, len(self.poi))) / 100.0 if n_a else 0.0
        age_b = (i - self.poi.max_birth(0, j)) / 100.0 if n_b else 0.0

        m = self.pool.split(p)
        pn_b, pn_a = m, len(self.pool) - m
        pnear_a = (self.pool.nearest_above(m) - p) / scale if pn_a else 0.0
        pnear_b = (p - self.pool.nearest_below(m)) / scale if pn_b else 0.0

        has_eqh = 1.0 if len(self.eqh) else 0.0
        has_eql = 1.0 if len(self.eql) else 0.0
        dist_eqh = (self.eqh.min_level() - p) / scale if len(self.eqh) else 0.0
        dist_eql = (p - self.eql.max_level()) / scale if len(self.eql) else 0.0

        self._rows.append([n_a / 10.0, n_b / 10.0, near_a, near_b,
                           pn_a / 10.0, pn_b / 10.0, pnear_a, pnear_b,
                           has_eqh, has_eql, dist_eqh, dist_eql,
                           age_a, age_b])

    def cap(self) -> None:
        """Drop rows no future plan can see: live only ever reads the newest."""
        if len(self._rows) > 2:
            self._rows = self._rows[-1:]

    def last_row(self) -> np.ndarray:
        out = np.zeros((1, len(ZONE_FEATS)), np.float32)
        out[0] = self._rows[-1]
        return out

    def arrays(self) -> tuple[np.ndarray, list[str]]:
        out = np.zeros((len(self._rows), len(ZONE_FEATS)), np.float32)
        for k, row in enumerate(self._rows):
            out[k] = row
        return out, ZONE_FEATS
