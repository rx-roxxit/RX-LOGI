"""ZoneStream == stage2.zones_on_tf, built one bar at a time.

Two things are load-bearing and easy to get wrong:

  * ORDER. The legacy builder appends every FVG, then every OB, then every
    liquidity line, then every EQ pool, and finally stable-sorts by `start`.
    Zones sharing a start bar therefore come out FVG, OB, LINE, EQ - so the four
    kinds are kept in separate lists and concatenated in that order at read time.

  * LIFECYCLE. This is stage-2's own rule, NOT the mirrors' mit_at: a box lives
    until a bar CLOSES beyond its far edge (a wick that only touches does not
    spend it), and an EQ pool dies on the first strict pierce. Only BSL/SSL take
    their death straight from liquidity's swept_at. The unspent boxes and pools
    sit in heaps keyed by the edge that would kill them, so a bar pops only what
    it actually breaks.

The legacy scan starts at s+1, so a zone is never killed by its own confirm bar:
pop before push, every bar.
"""
from __future__ import annotations

import heapq

import pandas as pd

from ..stage2 import TF_MIN as TF_MIN_ZONES
from ..stage2 import eq_level


class ZoneStream:
    def __init__(self) -> None:
        self.fvg: list[dict] = []
        self.ob: list[dict] = []
        self.line: list[dict] = []
        self.eq: list[dict] = []
        self._bull: list[tuple[float, int, list]] = []   # (-far, seq, bucket)
        self._bear: list[tuple[float, int, list]] = []   # (far,  seq, bucket)
        self._eq_low: list[tuple[float, int]] = []       # (-lvl, idx)
        self._eq_high: list[tuple[float, int]] = []      # (lvl,  idx)
        self._start = 0
        self.heap_push = 0
        self.heap_pop = 0
        self._live: dict[int, dict] = {}      # zone id -> record, still virgin
        self._dead: list[int] = []            # ids that died on the bar just fed
        self._next_id = 0
        self.live_count = 0

    # ── birth / death bookkeeping (live retention needs to find the survivors) ─
    def _register(self, rec: dict) -> None:
        rec["_id"] = self._next_id
        self._live[self._next_id] = rec
        self._next_id += 1
        self.live_count += 1

    def _mark_dead(self, rec: dict, i: int) -> None:
        rec["virgin_end"] = i
        self._dead.append(rec["_id"])

    # ── one bar ──────────────────────────────────────────────────────────────
    def feed(self, i: int, h, l, c, fvg, ob, liq, eqhl) -> None:
        self._start = i
        ci, hi_, li_ = float(c[i]), float(h[i]), float(l[i])

        # 1. kill first - a zone cannot be spent by its own confirm bar
        while self._bull and -self._bull[0][0] > ci:          # close < far (far = lo)
            _, k, bucket = heapq.heappop(self._bull)
            self._mark_dead(bucket[k], i)
            self.heap_pop += 1
        while self._bear and self._bear[0][0] < ci:           # close > far (far = hi)
            _, k, bucket = heapq.heappop(self._bear)
            self._mark_dead(bucket[k], i)
            self.heap_pop += 1
        while self._eq_low and -self._eq_low[0][0] > li_:     # low < lvl
            _, k = heapq.heappop(self._eq_low)
            self._mark_dead(self.eq[k], i)
            self.heap_pop += 1
        while self._eq_high and self._eq_high[0][0] < hi_:    # high > lvl
            _, k = heapq.heappop(self._eq_high)
            self._mark_dead(self.eq[k], i)
            self.heap_pop += 1

        # 2. liquidity lines take their death from the mirror, no heap of our own
        for k in liq.died_at_bar():
            self._mark_dead(self.line[k], i)

        # 3. births (index-parallel with each mirror's own object list)
        self._born_boxes(self.fvg, fvg.boxes, "FVG")
        self._born_boxes(self.ob, ob.boxes, "OB")
        self._born_lines(liq.lines)
        self._born_eq(eqhl.events)

    # ── births, kept index-parallel with the mirror's own object list ────────
    def _born_boxes(self, bucket, boxes, kind) -> None:
        while len(bucket) < len(boxes):
            b = boxes[len(bucket)]
            up = b["side"] == "bull"
            far = b["lo"] if up else b["hi"]
            bucket.append({"kind": kind, "up": up, "lo": b["lo"], "hi": b["hi"],
                           "touch": b["hi"] if up else b["lo"], "far": far,
                           "width": b["hi"] - b["lo"], "gap": float(b.get("gap", 0.0)),
                           "start": self._start, "virgin_end": None})
            self._register(bucket[-1])
            k = len(bucket) - 1
            heapq.heappush(self._bull if up else self._bear,
                           ((-far, k, bucket) if up else (far, k, bucket)))
            self.heap_push += 1

    def _born_lines(self, lines) -> None:
        while len(self.line) < len(lines):
            q = lines[len(self.line)]
            up = q["side"] == "BSL"
            self.line.append({"kind": "BSL" if up else "SSL", "up": up,
                              "lo": q["level"], "hi": q["level"], "touch": q["level"],
                              "far": q["level"], "width": 0.0, "gap": 0.0,
                              "start": self._start, "virgin_end": None})
            self._register(self.line[-1])

    def _born_eq(self, events) -> None:
        while len(self.eq) < len(events):
            e = events[len(self.eq)]
            is_low = e["kind"] == "EQL"
            lvl = eq_level(e["p1"], e["p2"], is_low)
            self.eq.append({"kind": "EQL" if is_low else "EQH", "up": is_low,
                            "lo": lvl, "hi": lvl, "touch": lvl, "far": lvl,
                            "width": 0.0, "gap": 0.0,
                            "start": self._start, "virgin_end": None})
            self._register(self.eq[-1])
            k = len(self.eq) - 1
            if is_low:
                heapq.heappush(self._eq_low, (-lvl, k))
            else:
                heapq.heappush(self._eq_high, (lvl, k))
            self.heap_push += 1

    # ── live retention ──────────────────────────────────────────────────────
    def reap(self) -> None:
        """Forget the zones that died on the bar just fed. Called AFTER every
        stream has read them - TokenStream reads a box at the moment it dies."""
        for zid in self._dead:
            if self._live.pop(zid, None) is not None:
                self.live_count -= 1
        self._dead = []

    def tombstone_buckets(self) -> int:
        """Replace dead records in the ordered buckets with None: the slot stays,
        so every index-parallel contract still points where it did, but the dict
        goes. O(objects), so the store amortizes the call."""
        n = 0
        for bucket in (self.fvg, self.ob, self.line, self.eq):
            for k, z in enumerate(bucket):
                if z is not None and z["virgin_end"] is not None:
                    bucket[k] = None
                    n += 1
        return n

    def virgin_in_band(self, cdt, tf: str, known_ts, band_lo: float,
                       band_hi: float, up: bool) -> list[dict]:
        """Exactly stage2.enumerate_candidates' predicate, over this timeframe.

        born_ts  = confirm bar close <= known_ts     (knowable)
        spent_ts = dying bar close    >  known_ts     (still virgin)
        plus side and band.

        In live retention known_ts is always NOW, so anything already dead has a
        dying bar that closed at or before it and is excluded here anyway - which
        is what makes tombstoning safe. W6r is the proof, not this comment.
        """
        dur = pd.Timedelta(minutes=TF_MIN_ZONES[tf])
        out: list[dict] = []
        for z in self._live.values():
            if z["up"] != up or not (band_lo <= z["touch"] <= band_hi):
                continue
            if cdt[z["start"]] + dur > known_ts:
                continue
            ve = z["virgin_end"]
            if ve is not None and cdt[ve] + dur <= known_ts:
                continue
            out.append({**{k: v for k, v in z.items() if k != "_id"}, "tf": tf})
        out.sort(key=lambda z: z["start"])
        return out

    # ── read ────────────────────────────────────────────────────────────────
    def zones(self, n: int) -> list[dict]:
        """Legacy shape and legacy order: FVG, OB, LINE, EQ concatenated, then a
        STABLE sort by start. `virgin_end = n` is the legacy 'never spent'
        sentinel and depends on the tape length, so it is filled in here."""
        out = []
        for bucket in (self.fvg, self.ob, self.line, self.eq):
            for z in bucket:
                r = {k: v for k, v in z.items() if k != "_id"}   # _id is bookkeeping
                if r["virgin_end"] is None:
                    r["virgin_end"] = n
                out.append(r)
        out.sort(key=lambda z: z["start"])
        return out
