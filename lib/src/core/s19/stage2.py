"""RX Logi V2.0 stage-2 primitives — PREREG §3B (pinned 2026-07-21).

Everything a stage-2 candidate needs, computed ONLY through the frozen features
(imported via `frozen`, which re-checks the hash on import). No market
definition is invented here; the two numbers that ARE stage-2's own — the
reclaim window (2 bars) and the EQL/EQH level choice (deeper swing) — live here
and NOWHERE in FEATURES_VERIFIED, so the freeze wall stays green.

Causality contract (walled in tests/test_s19_stage2_walls.py):
  * a zone is a candidate for a plan only once its confirming bar has CLOSED by
    the plan's known_ts, and only while it is still virgin at that instant;
  * that decision is a pure function of bars up to known_ts — appending future
    bars cannot change it (truncation invariance);
  * the reclaim state machine reproduces frozen sweep.py byte-for-byte at
    RECLAIM_BARS=3, so switching to 2 changes the constant and nothing else.

The label (touch / broken / plan-completes) is deliberately a function of FUTURE
bars — that is the outcome being learned, exactly like stage-1's done_at — and
is therefore NOT subject to the truncation wall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .frozen import eqhl, fvg, liquidity, ob, swing

# zone timeframes per cell = path TF + two finer levels (USER 2026-07-21)
LADDER = {"1H": ("1H", "30m", "15m"),
          "30m": ("30m", "15m", "5m"),
          "15m": ("15m", "5m", "3m")}
TF_MIN = {"3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240, "D": 1440}
RECLAIM_S2 = 2                     # stage-2 line rule: frozen sweep's 3, shortened
MAX_SWINGS = 80                    # frozen sweep.MAX_SWINGS (parity)
BOX = ("FVG", "OB")
LINE = ("BSL", "SSL", "EQL", "EQH")     # ALL line kinds — SSL/EQH are the LL-side
#                                         mirror; omitting them sends LL line zones
#                                         through the box grading branch (wrong rule)


# ── zone universe on ONE timeframe ───────────────────────────────────────────
def eq_level(p1: float, p2: float, is_low: bool) -> float:
    """The level an EQ pool actually rests at: the DEEPER of the two equal
    swings (EQL = the lower low, EQH = the higher high). PREREG §3B.2."""
    return min(p1, p2) if is_low else max(p1, p2)


def zones_on_tf(dt, o, h, l, c) -> list[dict]:
    """Every zone the frozen features emit on this tape, with its candidate life
    as bar indices [start, virgin_end): start = the bar whose close confirmed it,
    virgin_end = the bar that ENDS its eligibility (else len(c)).

    Eligibility rule (USER 2026-07-21, after eye-check):
      * BOX (FVG/OB) — alive until CLOSE-BROKEN (a bar closes beyond the far
        edge). Wicks / mitigation do NOT spend it: a mitigated-but-unbroken POI
        is still valid on re-test. (This differs from the frozen mit_at, which
        marks first wick for the DISPLAY; candidacy is a stage-2 concept.)
      * LINE (BSL/SSL) — liquidity swept_at (first wick through the level).
      * EQ (EQL/EQH) — the same strict wick-pierce, computed here.
    'Held after touch' is a LABEL question and plays no part here.
    """
    n = len(c)
    ix = pd.DatetimeIndex(dt)
    bar = {t: i for i, t in enumerate(ix)}
    cc = np.asarray(c, float)
    out: list[dict] = []

    for kind, boxes in (("FVG", fvg.compute(dt, o, h, l, c)["boxes"]),
                        ("OB", ob.compute(dt, o, h, l, c)["boxes"])):
        for b in boxes:
            up = b["side"] == "bull"
            far = b["lo"] if up else b["hi"]
            s = bar[b["confirm_at"]]
            seg = cc[s + 1:]                            # first CLOSE beyond far edge
            hit = (seg < far) if up else (seg > far)
            w = int(np.argmax(hit)) if hit.size else 0
            ve = (s + 1 + w) if (hit.size and hit[w]) else n
            out.append({"kind": kind, "up": up, "lo": b["lo"], "hi": b["hi"],
                        "touch": b["hi"] if up else b["lo"],       # near edge
                        "far": far, "width": b["hi"] - b["lo"],
                        "gap": float(b.get("gap", 0.0)), "start": s, "virgin_end": ve})

    for q in liquidity.compute(dt, o, h, l, c)["lines"]:
        up = q["side"] == "BSL"                       # BSL below (green), SSL above
        out.append({"kind": "BSL" if up else "SSL", "up": up,
                    "lo": q["level"], "hi": q["level"], "touch": q["level"],
                    "far": q["level"], "width": 0.0, "gap": 0.0,
                    "start": bar[q["confirm_at"]],
                    "virgin_end": bar[q["swept_at"]] if q["swept_at"] is not None else n})

    for e in eqhl.compute(dt, o, h, l, c)["events"]:
        is_low = e["kind"] == "EQL"
        lvl = eq_level(e["p1"], e["p2"], is_low)
        s = bar[e["confirm_at"]]
        ve = n
        for j in range(s + 1, n):                     # first strict pierce (as liquidity)
            if (l[j] < lvl) if is_low else (h[j] > lvl):
                ve = j
                break
        out.append({"kind": "EQL" if is_low else "EQH", "up": is_low,
                    "lo": lvl, "hi": lvl, "touch": lvl, "far": lvl,
                    "width": 0.0, "gap": 0.0, "start": s, "virgin_end": ve})

    out.sort(key=lambda z: z["start"])
    return out


# ── reclaim primitive — proven == frozen sweep.py at RECLAIM_BARS=3 ──────────
def sweep_markers(dt, o, h, l, c, reclaim_bars: int = RECLAIM_S2) -> list[dict]:
    """Reimplementation of FEATURES_VERIFIED/sweep.py with the reclaim window as
    a parameter. At reclaim_bars=3 the output is identical to the frozen module
    (tests/test_s19_stage2_walls.py::test_sweep_reclaim_parity_at_3); stage-2
    grades line zones with reclaim_bars=2 — the same rule, one constant changed.
    """
    sm = swing.SwingSM(h, l, c)
    ssl: list[dict] = []
    bsl: list[dict] = []
    markers: list[dict] = []
    n = len(c)
    for i in range(n):
        ev = sm.update(i)
        if ev is not None:
            pool = ssl if ev["side"] == "high" else bsl
            pool.append({"lvl": ev["price"], "bar": ev["bar"], "brk": -1})
            if len(pool) > MAX_SWINGS:
                del pool[0: len(pool) - MAX_SWINGS]
        for j in range(len(ssl) - 1, -1, -1):
            q = ssl[j]
            if q["brk"] < 0:
                if h[i] > q["lvl"]:
                    if c[i] <= q["lvl"]:
                        markers.append({"at": dt[i], "side": "SSL", "level": q["lvl"]})
                        del ssl[j]
                    else:
                        q["brk"] = i
            else:
                if c[i] <= q["lvl"]:
                    markers.append({"at": dt[i], "side": "SSL", "level": q["lvl"]})
                    del ssl[j]
                elif i - q["brk"] >= reclaim_bars:
                    del ssl[j]
        for j in range(len(bsl) - 1, -1, -1):
            q = bsl[j]
            if q["brk"] < 0:
                if l[i] < q["lvl"]:
                    if c[i] >= q["lvl"]:
                        markers.append({"at": dt[i], "side": "BSL", "level": q["lvl"]})
                        del bsl[j]
                    else:
                        q["brk"] = i
            else:
                if c[i] >= q["lvl"]:
                    markers.append({"at": dt[i], "side": "BSL", "level": q["lvl"]})
                    del bsl[j]
                elif i - q["brk"] >= reclaim_bars:
                    del bsl[j]
    return markers


# ── grade one candidate over the leg (k, end] on its OWN timeframe ───────────
def resolve(z: dict, h, l, c, k: int, end: int, reclaim_bars: int = RECLAIM_S2
            ) -> tuple[bool, bool]:
    """(touched, held). `end` MUST be the bar the plan's target is reached, not
    the next break's confirm bar (§3B.5). Box: touch = wick reaches near edge,
    broken = a bar CLOSES beyond the far edge (same bar as touch -> broken).
    Line: strict pierce = touch; a CLOSE beyond the level that is not reclaimed
    within reclaim_bars = broken. 'not broken' is checked over the WHOLE leg, not
    just the first sweep — an early sweep-and-reclaim followed by a later
    unreclaimed close-through is still broken (the frozen sweep.py rule, applied
    to one candidate across the leg)."""
    up, t, far = z["up"], z["touch"], z["far"]
    line = z["kind"] in LINE
    if line:
        touched = False
        pend = -1                                     # bar of an unreclaimed close-through
        for j in range(k + 1, end + 1):
            if (l[j] < t) if up else (h[j] > t):      # strict pierce
                touched = True
            through = (c[j] < t) if up else (c[j] > t)     # close beyond the level
            back = not through                             # close reclaimed (c<=t / c>=t)
            if pend < 0:
                if through:
                    pend = j
            elif back:
                pend = -1                             # reclaimed within the window
            elif j - pend >= reclaim_bars:
                return touched, False                 # unreclaimed break
        return touched, touched and pend < 0
    for j in range(k + 1, end + 1):
        if not ((l[j] <= t) if up else (h[j] >= t)):
            continue
        if (c[j] < far) if up else (c[j] > far):
            return True, False                        # touch and break same bar
        for m in range(j + 1, end + 1):
            if (c[m] < far) if up else (c[m] > far):
                return True, False
        return True, True
    return False, False


# ── candidate list for one plan (causal: only bars closed by known_ts) ───────
def plan_band(plan: dict, ref_close: float) -> tuple[float, float, bool]:
    """(band_lo, band_hi, up). HH: above erl_low (incl. promotions), below the
    path-TF close at known_at. LL mirrors. §3B.3."""
    up = plan["state"] == "HH"
    return (plan["erl_lo"], ref_close, True) if up else (ref_close, plan["erl_hi"], False)


def ref_at(path_tape, known_at) -> float:
    dt, _o, _h, _l, c = path_tape
    i = int(np.searchsorted(pd.DatetimeIndex(dt).to_numpy(),
                            np.datetime64(pd.Timestamp(known_at)), side="left"))
    return float(np.asarray(c, float)[min(i, len(c) - 1)])


def enumerate_cached(cell: str, zones_by_tf: dict, plan: dict, ref_close: float
                     ) -> list[dict]:
    """Same candidate list as `enumerate_candidates`, but reading pre-cached zone
    records (each carrying `start_at` and `virgin_end_at` timestamps) instead of
    re-touching the tape — the frozen features are expensive on fine TFs, so the
    labeler pays that cost once via stage2_zones and enumerates from the cache.

    zones_by_tf[tf] = list of dicts with at least start_at, virgin_end_at (NaT if
    never spent), up, touch, kind. Walled == enumerate_candidates.
    """
    tfs = LADDER[cell]
    up = plan["state"] == "HH"
    known_ts = pd.Timestamp(plan["known_at"]) + pd.Timedelta(minutes=TF_MIN[tfs[0]])
    blo, bhi, _ = plan_band(plan, ref_close)
    out: list[dict] = []
    for tf in tfs:
        dur = pd.Timedelta(minutes=TF_MIN[tf])
        for z in zones_by_tf[tf]:
            if z["start_at"] + dur > known_ts:                     # not closed yet
                continue
            ve = z["virgin_end_at"]
            if pd.notna(ve) and ve + dur <= known_ts:              # already spent
                continue
            if z["up"] != up or not (blo <= z["touch"] <= bhi):
                continue
            out.append({**z, "tf": tf})
    return out


_NAT_MAX = np.iinfo(np.int64).max


def _zone_arrays(zones: list[dict], tf: str) -> tuple:
    """born_ns / spent_ns / touch / up as numpy arrays for vectorized filtering.
    born = confirm-bar close, spent = first-touch-bar close (max if never)."""
    dur = int(pd.Timedelta(minutes=TF_MIN[tf]).value)
    born = np.fromiter((pd.Timestamp(z["start_at"]).value for z in zones),
                       np.int64, len(zones)) + dur
    ve = np.fromiter((pd.Timestamp(z["virgin_end_at"]).value if pd.notna(z["virgin_end_at"])
                      else _NAT_MAX for z in zones), np.int64, len(zones))
    spent = np.where(ve == _NAT_MAX, ve, ve + dur)
    touch = np.fromiter((z["touch"] for z in zones), np.float64, len(zones))
    up = np.fromiter((bool(z["up"]) for z in zones), np.bool_, len(zones))
    return born, spent, touch, up


def enumerate_batch(cell: str, zones_by_tf: dict, plans: list, refs) -> list[list[tuple]]:
    """Vectorized equivalent of calling enumerate_cached per plan — returns, for
    each plan, a list of (tf, zone_index) into zones_by_tf. O(plans x zones) but
    in numpy, so the labeler runs in seconds instead of the O(...) Python scan.
    Walled == enumerate_cached in tests/test_s19_stage2_walls.py."""
    tfs = LADDER[cell]
    arr = {tf: _zone_arrays(zones_by_tf[tf], tf) for tf in tfs}
    pathdur = int(pd.Timedelta(minutes=TF_MIN[tfs[0]]).value)
    out: list[list[tuple]] = []
    for i, p in enumerate(plans):
        pu = p["state"] == "HH"
        kv = pd.Timestamp(p["known_at"]).value + pathdur
        blo, bhi, _ = plan_band(p, float(refs[i]))
        cur: list[tuple] = []
        for tf in tfs:
            born, spent, touch, up = arr[tf]
            m = (born <= kv) & (spent > kv) & (up == pu) & (touch >= blo) & (touch <= bhi)
            cur.extend((tf, int(j)) for j in np.nonzero(m)[0])
        out.append(cur)
    return out


def enumerate_candidates(cell: str, tapes: dict, plan: dict,
                         _zcache: dict | None = None) -> list[dict]:
    """Virgin, in-band candidates for one plan across the cell's three zone TFs.

    A zone qualifies iff, at the plan's known_ts (= path confirm-bar close):
      born_ts  = confirm bar close  <=  known_ts     (knowable)
      spent_ts = first-touch bar close  >   known_ts  (still virgin)
    and it is on the plan's side and inside the band. Pure function of bars up
    to known_ts (walled). SSL/EQH fall outside an HH band on their own.
    """
    tfs = LADDER[cell]
    up = plan["state"] == "HH"
    path = tfs[0]
    known_ts = pd.Timestamp(plan["known_at"]) + pd.Timedelta(minutes=TF_MIN[path])
    ref = ref_at(tapes[path], plan["known_at"])
    blo, bhi, _ = plan_band(plan, ref)

    out: list[dict] = []
    for tf in tfs:
        dt, o, h, l, c = tapes[tf]
        Z = (_zcache or {}).get(tf) if _zcache is not None else None
        if Z is None:
            Z = zones_on_tf(dt, o, h, l, c)
            if _zcache is not None:
                _zcache[tf] = Z
        cdt = pd.DatetimeIndex(dt)
        dur = pd.Timedelta(minutes=TF_MIN[tf])
        n = len(c)
        for z in Z:
            born_ts = cdt[z["start"]] + dur
            if born_ts > known_ts:                    # confirm bar not closed yet
                continue
            spent_ts = (cdt[z["virgin_end"]] + dur) if z["virgin_end"] < n else pd.Timestamp.max
            if spent_ts <= known_ts:                  # already touched/swept
                continue
            if z["up"] != up or not (blo <= z["touch"] <= bhi):
                continue
            out.append({**z, "tf": tf})
    return out


# ── merge zones into POI clusters by CONTAINMENT (USER 2026-07-21) ────────────
CLUSTER_FEATS = ["dist_rel", "pos_band", "width_rel", "n_zones", "n_tf",
                 "has_path", "has_ob", "has_fvg", "has_line", "finest_depth"]
LINE_EPS_FRAC = 0.001                   # co-located line merge tolerance (of band)


def cluster_candidates(cands: list[dict], band_lo: float, band_hi: float,
                       ref: float, up: bool) -> list[dict]:
    """Merge zones into POI clusters by CONTAINMENT, not overlap (USER rule).

    A smaller zone folds into a POI only if it sits ENTIRELY INSIDE the POI's
    anchor (anchor.lo <= z.lo and z.hi <= anchor.hi) — no spilling out. So:
      * boxes that only TOUCH (an OB's hi == its FVG's lo) or partially overlap
        stay SEPARATE POIs — an OB and its FVG on the same TF are NOT merged;
      * a finer-TF zone fully inside a coarser box folds in (dedup across TFs),
        and different kinds may merge if contained (a 5m OB inside a 1H FVG);
      * a line whose level lies inside a box folds into that box; lines not in
        any box merge only with co-located lines (same level +/- a tiny eps).
    Anchor = the largest containing zone (processed biggest-first, path TF wins
    ties). Members kept as confluence features; OK if ANY member is the launch.
    Each `cands[i]` needs kind, tf_depth, lo, hi, touch, ok.
    """
    band = band_hi - band_lo
    eps = max(band, 1e-9) * LINE_EPS_FRAC
    order = sorted(range(len(cands)),                 # containers before contents
                   key=lambda i: (-(cands[i]["hi"] - cands[i]["lo"]), cands[i]["tf_depth"]))
    pois: list[dict] = []
    for i in order:
        z = cands[i]
        zline = z["kind"] in LINE
        placed = False
        for poi in pois:
            if poi["line"]:
                if zline and abs(poi["hi"] - z["touch"]) <= eps:   # co-located lines
                    poi["members"].append(i); placed = True; break
            elif poi["lo"] <= z["lo"] and z["hi"] <= poi["hi"]:    # fully contained
                poi["members"].append(i); placed = True; break
        if not placed:
            pois.append({"lo": z["lo"], "hi": z["hi"], "line": zline, "members": [i]})
    out = []
    for poi in pois:
        mem = [cands[k] for k in poi["members"]]
        elo, ehi = poi["lo"], poi["hi"]               # the anchor's own range
        touch = ehi if up else elo                    # near edge (first hit)
        kinds = {z["kind"] for z in mem}
        out.append({
            "lo": elo, "hi": ehi, "touch": touch,
            "dist_rel": abs(ref - touch) / band if band > 0 else 0.0,
            "pos_band": (touch - band_lo) / band if band > 0 else 0.0,
            "width_rel": (ehi - elo) / band if band > 0 else 0.0,
            "n_zones": len(mem), "n_tf": len({z["tf_depth"] for z in mem}),
            "has_path": int(any(z["tf_depth"] == 0 for z in mem)),
            "has_ob": int("OB" in kinds), "has_fvg": int("FVG" in kinds),
            "has_line": int(bool(kinds & set(LINE))),
            "finest_depth": max(z["tf_depth"] for z in mem),
            "members": poi["members"], "ok": int(any(z["ok"] for z in mem)),
        })
    return out
