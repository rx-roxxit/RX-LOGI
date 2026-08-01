"""Assemble a cell into arrays, and cut PURGED walk-forward folds (PREREG §4B.4).

Purge: a plan may only be TRAINED on once its own label has completed. A plan
made in December whose second break confirms in January carries information from
the year we are about to score — training on it would leak the test year in
through the label, and nothing downstream would ever show it.

Training uses every break (at serving a wrong plan is re-planned immediately, so
any break can be a plan instant); scoring uses the walk stream only.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from .labels import TF_MIN, build_plans
from .mtf import CELLS, assemble
from .tokens import merged_stream, sequences

ROOT = pathlib.Path(__file__).resolve().parents[3]
BARS = ROOT / "data" / "bars_full"
CACHE = ROOT / "data" / "core" / "s19"
CLS_ID = {"A": 0, "B": 1, "C": 2}
# S17 naming: product names differ from the tape file names
ASSET_FILE = {"XAUUSD": "XAUUSD", "EURUSD": "EURUSD", "GBPUSD": "GBPUSD",
              "USA30": "US30", "USATECH": "NAS100"}
AIO_ASSETS = tuple(ASSET_FILE)


def load_tape(tf: str, symbol: str = "XAUUSD"):
    f = ASSET_FILE.get(symbol, symbol)
    d = pd.read_parquet(BARS / f"{f}_{tf}.parquet").set_index("datetime_utc")
    d.index = pd.DatetimeIndex(d.index).tz_convert("UTC").tz_localize(None)
    return (d.index, *(d[k].to_numpy(float) for k in ("open", "high", "low", "close")))


def build_cell(cell: str, symbol: str = "XAUUSD", frm: str = "2015-01-01",
               to: str = "2026-01-01") -> dict:
    """Everything one cell needs. `frm` starts a year early so the frozen
    features and the running context states are warm before 2016."""
    htf, mtf, path = CELLS[cell]
    tapes = {}
    for tf in (htf, mtf, path):
        dt, o, h, l, c = load_tape(tf, symbol)
        m = (dt >= frm) & (dt < to)
        tapes[tf] = (dt[m], o[m], h[m], l[m], c[m])

    plans = build_plans(*tapes[path])
    ctx, ctx_names = assemble(cell, tapes, plans)
    known_at = pd.DatetimeIndex([p["known_at"] for p in plans])
    known_ts = known_at + pd.Timedelta(minutes=TF_MIN[path])
    pdt = pd.DatetimeIndex(tapes[path][0])
    pidx = np.clip(np.searchsorted(pdt.to_numpy(), known_at.to_numpy(), side="left"),
                   0, len(pdt) - 1)
    ref = np.asarray(tapes[path][4], float)[pidx]
    width = np.array([p["erl_hi"] - p["erl_lo"] for p in plans], float)

    seq, lens = sequences(merged_stream((htf, mtf, path), tapes), known_ts, ref, width)
    y = np.array([CLS_ID[p["cls"]] for p in plans], np.int64)
    return {
        "cell": cell, "plans": plans, "ctx": ctx, "ctx_names": ctx_names,
        "seq": seq, "seq_len": lens, "y": y,
        "known_at": known_at,
        "done_at": pd.DatetimeIndex([p["done_known_at"] for p in plans]),
        "walk": np.array([p["walk"] for p in plans], bool),
        "state_hh": np.array([p["state"] == "HH" for p in plans], bool),
    }


def cache_cell(cell: str, symbol: str = "XAUUSD") -> pathlib.Path:
    f = CACHE / f"cell_{symbol}_{cell}.npz"
    if f.exists():
        return f
    d = build_cell(cell, symbol)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f, ctx=d["ctx"], seq=d["seq"], seq_len=d["seq_len"], y=d["y"],
        walk=d["walk"], state_hh=d["state_hh"],
        known_at=d["known_at"].values.astype("datetime64[ns]"),
        done_at=d["done_at"].values.astype("datetime64[ns]"),
        ctx_names=np.array(d["ctx_names"], dtype=object), allow_pickle=True)
    return f


def load_cell(cell: str, symbol: str = "XAUUSD") -> dict:
    z = np.load(cache_cell(cell, symbol), allow_pickle=True)
    return {k: z[k] for k in z.files}


def folds(d: dict, years=(2020, 2021, 2022, 2023, 2024),
          train_from: str = "2016-01-01") -> list[dict]:
    """Purged walk-forward: train on plans finished before the year, score the
    year's WALK plans that also finish inside it."""
    known = pd.DatetimeIndex(d["known_at"])
    done = pd.DatetimeIndex(d["done_at"])
    out = []
    for y in years:
        a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        tr = np.where((known >= train_from) & (done < a))[0]
        te = np.where(d["walk"] & (known >= a) & (known < b) & (done < b))[0]
        out.append({"year": y, "train": tr, "test": te})
    return out


def exam_split(d: dict, train_from: str = "2016-01-01",
               exam: str = "2025-01-01", end: str = "2026-01-01") -> dict:
    known = pd.DatetimeIndex(d["known_at"])
    done = pd.DatetimeIndex(d["done_at"])
    a, b = pd.Timestamp(exam), pd.Timestamp(end)
    return {"train": np.where((known >= train_from) & (done < a))[0],
            "test": np.where(d["walk"] & (known >= a) & (known < b) & (done < b))[0]}


def inner_val(d: dict, train_idx: np.ndarray, months: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Last `months` of the training window = early-stopping / calibration set."""
    known = pd.DatetimeIndex(d["known_at"])[train_idx]
    cut = known.max() - pd.DateOffset(months=months)
    return train_idx[known < cut], train_idx[known >= cut]


def load_pooled(cell: str, symbols=AIO_ASSETS) -> dict:
    """AIO: five instruments in one training set, no asset identifier as a feature.

    Rows stay grouped by asset and time-ordered within each group; `asset` marks
    the boundaries so anything sequential (the Markov null's break history) can
    respect them. Splitting is by timestamp, so grouping does not affect folds.
    """
    ds = [load_cell(cell, s) for s in symbols]
    keys = ("ctx", "seq", "seq_len", "y", "walk", "state_hh", "known_at", "done_at")
    out = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    out["asset"] = np.concatenate([np.full(len(d["y"]), i, np.int64)
                                   for i, d in enumerate(ds)])
    out["asset_names"] = np.array(list(symbols), dtype=object)
    out["ctx_names"] = ds[0]["ctx_names"]
    return out


def to_binary(d: dict) -> dict:
    """PREREG 4D — one-event target, derived from the 3-class cache without
    rebuilding anything.

      y      : 0 = continue (was class A), 1 = reverse (was B or C)
      done_at: the plan resolves at its FIRST break, which is exactly the next
               plan's known_at inside the same instrument (verified identical to
               the stored done_at on every depth-1 row).

    Plans no longer overlap, so the walk filter is dropped and every plan counts.
    """
    out = dict(d)
    out["y"] = (np.asarray(d["y"]) != 0).astype(np.int64)
    kn = pd.DatetimeIndex(d["known_at"])
    a = np.asarray(d.get("asset", np.zeros(len(kn), np.int64)))
    nxt = np.full(len(kn), np.datetime64("NaT"), "datetime64[ns]")
    same = np.zeros(len(kn), bool)
    same[:-1] = a[1:] == a[:-1]
    nxt[:-1][same[:-1]] = kn.to_numpy()[1:][same[:-1]]
    out["done_at"] = nxt
    # NaT cast to float64 is a huge finite number, not nan — isnat is the only
    # safe test here (this bit silently marked every asset-boundary row usable)
    out["walk"] = ~np.isnat(nxt)                         # usable = has a next break
    out["n_out"] = 2
    return out
