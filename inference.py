# -*- coding: utf-8 -*-
"""RX Logi V2.0 - the planner. ONE entry point: plan(bars_by_tf, cell).

This is a trade PLANNER, not an auto-trader and not an entry signal: on every
confirmed ERL break it reads whether structure will CONTINUE or REVERSE (stage-1)
and, when it reads continue, ranks the 1-3 zones price is most likely to launch
from (stage-2). Every result carries confidence; the human confirms and decides.

It runs the EXACT feature/structure/model code the models were trained on (vendored
byte-identical under lib/), so there is no re-implementation and no drift. Python
runtime required; do NOT port the feature pipeline to another language.

    from inference import plan
    out = plan(bars_by_tf, "1H")   # bars_by_tf: {timeframe: OHLC DataFrame in UTC}
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from joblib import load

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))          # the vendored pipeline (only dep)

from src.core.s19.labels import (TF_MIN, _edges_upto, breaks,  # noqa: E402
                                 edge_stream)
from src.core.s19.mtf import CELLS, assemble             # noqa: E402
from src.core.s19.tokens import merged_stream, sequences  # noqa: E402
from src.core.s19.model import ChainNet                   # noqa: E402
from src.core.s19.stage2 import (CLUSTER_FEATS, LADDER,    # noqa: E402
                                 cluster_candidates, enumerate_candidates,
                                 plan_band, ref_at)

FEATS = CLUSTER_FEATS + ["n_clusters"]                    # stage-2's 11 columns, in order
CELLS_ALL = ("1H", "30m", "15m")
WEIGHTS = os.path.join(HERE, "weights")

_S1: dict = {}      # cell -> [(net, mu, sd), ...]
_S2: dict = {}      # cell -> [gbm, ...]


def _load_stage1(cell: str):
    if cell not in _S1:
        ms = []
        for s in (0, 1, 2):
            ck = torch.load(os.path.join(WEIGHTS, "stage1", cell, f"prod_seed{s}.pt"),
                            map_location="cpu", weights_only=False)
            net = ChainNet(ck["n_ctx"], n_out=ck["n_out"], **ck["arch"])
            net.load_state_dict(ck["state_dict"])
            net.eval()
            ms.append((net, np.asarray(ck["ctx_mu"], np.float32),
                       np.asarray(ck["ctx_sd"], np.float32)))
        _S1[cell] = ms
    return _S1[cell]


def _load_stage2(cell: str):
    if cell not in _S2:
        _S2[cell] = [load(os.path.join(WEIGHTS, "stage2", cell, f"prod_seed{s}.joblib"))
                     for s in (0, 1, 2)]
    return _S2[cell]


def _to_tape(df: pd.DataFrame):
    """OHLC DataFrame (UTC-indexed) -> (DatetimeIndex, o, h, l, c) like load_tape."""
    df = df.sort_index()
    dt = pd.DatetimeIndex(df.index)
    if dt.tz is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    g = lambda k: df[k].to_numpy(float)
    return (dt, g("open"), g("high"), g("low"), g("close"))


def _latest_plan(dt, o, h, l, c):
    """The plan at the MOST RECENT ERL break, built from b0 + the edge stream only
    (no future breaks needed). Mirrors the input fields of labels.build_plans."""
    brk = breaks(h, l, c)
    stream = edge_stream(h, l, c)
    hi = lo = None
    cur = 0
    last = None
    for b0 in brk:                                    # replay edges up to each break
        k = b0["confirm"]
        nhi, nlo, cur = _edges_upto(stream, k, cur)
        hi = nhi or hi
        lo = nlo or lo
        if hi is None or lo is None:
            continue
        last = (b0, hi, lo)
    if last is None:
        return None
    b0, hi, lo = last
    up = b0["dir"] == "U"
    return {"known_at": dt[int(b0["confirm"])], "known_bar": int(b0["confirm"]),
            "state": "HH" if up else "LL",
            "erl_hi": float(hi[0]), "erl_lo": float(lo[0])}


def _stage1_pcont(cell, seq, lens, ctx):
    probs = []
    with torch.no_grad():
        for net, mu, sd in _load_stage1(cell):
            ctxn = (ctx - mu) / sd
            logit = net(torch.as_tensor(seq, dtype=torch.float32),
                        torch.as_tensor(lens, dtype=torch.long),
                        torch.as_tensor(ctxn, dtype=torch.float32))
            probs.append(torch.softmax(logit, 1)[:, 0].numpy())   # class 0 = continue
    return float(np.mean(probs, axis=0)[0])


def _stage2_scores(cell, X):
    return np.mean([g.predict_proba(X)[:, 1] for g in _load_stage2(cell)], axis=0)


def plan(bars_by_tf: dict, cell: str, require_fresh: bool = True) -> dict | None:
    """Returns None if the latest path-TF bar did not just confirm an ERL break;
    otherwise the planner output for that break. See module docstring / model_card."""
    if cell not in CELLS_ALL:
        raise ValueError(f"cell must be one of {CELLS_ALL}")
    htf, mtf, path = CELLS[cell]
    need = set((htf, mtf, path)) | set(LADDER[cell])
    missing = need - set(bars_by_tf)
    if missing:
        raise ValueError(f"cell {cell} needs timeframes {sorted(need)}; missing {sorted(missing)}")
    tapes = {tf: _to_tape(bars_by_tf[tf]) for tf in need}
    pdt, po, ph, pl, pc = tapes[path]

    p = _latest_plan(pdt, po, ph, pl, pc)
    if p is None:
        return None
    if require_fresh and p["known_bar"] != len(pc) - 1:   # break confirmed earlier, already served
        return None

    # ---- stage-1: continue vs reverse ----
    ref = ref_at(tapes[path], p["known_at"])
    width = p["erl_hi"] - p["erl_lo"]
    ctx, _ = assemble(cell, tapes, [p])
    known_ts = pd.DatetimeIndex([p["known_at"]]) + pd.Timedelta(minutes=TF_MIN[path])
    seq, lens = sequences(merged_stream((htf, mtf, path), tapes), known_ts,
                          np.array([ref], float), np.array([width], float))
    p_cont = _stage1_pcont(cell, seq, lens, ctx)
    up = p["state"] == "HH"

    out = {
        "instant": pd.Timestamp(p["known_at"]).isoformat(),
        "cell": cell,
        "direction": "up" if up else "down",
        "state": p["state"],
        "continue": bool(p_cont > 0.5),
        "dir_confidence": round(p_cont, 4),
        "erl_target": p["erl_hi"] if up else p["erl_lo"],       # the level a continuation makes new
        "invalidation": p["erl_lo"] if up else p["erl_hi"],     # break this and the read is wrong
        "zones": [],
    }

    # ---- stage-2: rank the launch zones (served always, with confidence) ----
    cands = enumerate_candidates(cell, tapes, p)
    for z in cands:                                    # cluster_candidates keys on tf_depth
        z["tf_depth"] = LADDER[cell].index(z["tf"])    # (stage2_label sets this identically)
        z["ok"] = 0                                    # ok is a training LABEL; unused at serve
    blo, bhi, _ = plan_band(p, ref)
    clusters = cluster_candidates(cands, blo, bhi, ref, up)
    if clusters:
        nc = len(clusters)
        X = np.array([[cl[f] for f in CLUSTER_FEATS] + [nc] for cl in clusters], float)
        sc = _stage2_scores(cell, X)
        for rank, idx in enumerate(np.argsort(-sc)[:3]):
            cl = clusters[int(idx)]
            out["zones"].append({
                "rank": rank + 1,
                "lo": round(float(cl["lo"]), 5), "hi": round(float(cl["hi"]), 5),
                "touch": round(float(cl["touch"]), 5),
                "confidence": round(float(sc[int(idx)]), 4),
                "n_zones": int(cl["n_zones"]), "n_tf": int(cl["n_tf"]),
                "has_ob": int(cl["has_ob"]), "has_fvg": int(cl["has_fvg"]),
                "has_line": int(cl["has_line"]),
            })
    return out


class LiveBook:
    """Every cell of one symbol, sharing one set of stores and fed once.

    plan() is a pure function of the whole tape and recomputes the market from
    2015 on every call. This holds that computation as state and appends one
    group to it, so a plan costs the bar rather than the decade.

    Feed CLOSED bars of EVERY timeframe the cells use, GROUPED BY CLOSE INSTANT:
    gather everything that closed at time T, hand it over as one group, then
    read the answer (src/core/s19/replay.py::replay_feed does this for a replay;
    a live feed must do the same).

    One book per symbol. Stores are per (symbol, timeframe), so there is nothing
    to share across symbols.
    """

    def __init__(self, cells=CELLS_ALL, retain: str = "live") -> None:
        from src.core.s19.book import StoreBook
        from src.core.s19.cell import CellEngine
        cells = tuple(cells)
        seen_cells = []
        for c in cells:
            if c not in CELLS_ALL:
                raise ValueError(f"cell must be one of {CELLS_ALL}, got {c!r}")
            if c not in seen_cells:
                seen_cells.append(c)
        self.cells = tuple(seen_cells)
        self.book = StoreBook(self.cells, retain=retain)
        self.engines = {c: CellEngine(c, self.book) for c in self.cells}

    def warm(self, bars_by_tf: dict) -> None:
        """Walk history once. Retention is whatever the book was built with, so
        live mode ends warm start already inside its memory budget.

        No model runs here - warming is ingest only.
        """
        from src.core.s19.replay import replay_feed
        need = set(self.book.stores)
        missing = need - set(bars_by_tf)
        if missing:
            raise ValueError(
                f"cells {self.cells} need timeframes {sorted(need)}; "
                f"missing {sorted(missing)}")
        tapes = {}
        for tf in need:
            b = bars_by_tf[tf]
            tapes[tf] = _to_tape(b) if hasattr(b, "columns") else b
        for group in replay_feed(tapes):
            self.book.extend_group(group)

    def on_closed_bars(self, items) -> dict:
        """Take EVERY bar that closed on one instant, then return the plans that
        instant makes, keyed by cell. Nothing happened -> {}, which is ~99% of
        calls.

        items = [(tf, (timestamp, open, high, low, close)), ...]
        """
        fed = self.book.extend_group(items)
        out = {}
        for cell, eng in self.engines.items():
            eng.observe(fed)
            if not eng.gate():
                continue
            p = self._plan(cell, eng)
            if p is not None:
                out[cell] = p
        return out

    def _plan(self, cell: str, eng) -> dict | None:
        p = eng.latest_plan()
        if p is None:
            return None
        path = CELLS[cell][2]
        st = eng.stores[path]
        ref = float(st.c[p["known_bar"]])
        width = p["erl_hi"] - p["erl_lo"]
        ctx = eng.ctx_row(p, ref)
        seq, lens = eng.seq_tail(p, ref, width)
        p_cont = _stage1_pcont(cell, seq, lens, ctx)
        up = p["state"] == "HH"

        out = {
            "instant": pd.Timestamp(p["known_at"]).isoformat(),
            "cell": cell,
            "direction": "up" if up else "down",
            "state": p["state"],
            "continue": bool(p_cont > 0.5),
            "dir_confidence": round(p_cont, 4),
            "erl_target": p["erl_hi"] if up else p["erl_lo"],
            "invalidation": p["erl_lo"] if up else p["erl_hi"],
            "zones": [],
        }

        cands = eng.candidates(p, ref)
        for z in cands:
            z["tf_depth"] = LADDER[cell].index(z["tf"])
            z["ok"] = 0
        blo, bhi, _ = plan_band(p, ref)
        clusters = cluster_candidates(cands, blo, bhi, ref, up)
        if clusters:
            nc = len(clusters)
            X = np.array([[cl[f] for f in CLUSTER_FEATS] + [nc] for cl in clusters], float)
            sc = _stage2_scores(cell, X)
            for rank, idx in enumerate(np.argsort(-sc)[:3]):
                cl = clusters[int(idx)]
                out["zones"].append({
                    "rank": rank + 1,
                    "lo": round(float(cl["lo"]), 5), "hi": round(float(cl["hi"]), 5),
                    "touch": round(float(cl["touch"]), 5),
                    "confidence": round(float(sc[int(idx)]), 4),
                    "n_zones": int(cl["n_zones"]), "n_tf": int(cl["n_tf"]),
                    "has_ob": int(cl["has_ob"]), "has_fvg": int(cl["has_fvg"]),
                    "has_line": int(cl["has_line"]),
                })
        return out


class LivePlanner:
    """One cell, kept warm across calls.

    This is a LiveBook with a single cell - the same code path, not a parallel
    copy - so every wall written against it covers the book too. Serving more
    than one cell of a symbol should use LiveBook: it holds 7 stores instead of
    15 and feeds each bar once instead of three times.
    """

    def __init__(self, cell: str, retain: str = "live") -> None:
        if cell not in CELLS_ALL:
            raise ValueError(f"cell must be one of {CELLS_ALL}")
        self.cell = cell
        self.path = CELLS[cell][2]
        self._lb = LiveBook((cell,), retain=retain)
        self.eng = self._lb.engines[cell]

    def warm(self, bars_by_tf: dict) -> None:
        self._lb.warm(bars_by_tf)

    def on_closed_bars(self, items) -> dict | None:
        """Take EVERY bar that closed on one instant, then return the plan that
        instant makes - or None. There is deliberately no single-bar entry
        point, because any per-bar ordering hides something that should be
        visible (see src/core/s19/replay.py)."""
        return self._lb.on_closed_bars(items).get(self.cell)
