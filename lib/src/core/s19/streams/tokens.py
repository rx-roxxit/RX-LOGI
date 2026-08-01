"""TokenStream == tokens.tf_tokens, built one bar at a time.

Every token key is the bar on which the thing became knowable, so all tokens a
bar produces share a sort key and their ORDER is the whole contract. The legacy
_rows() walks the nine features in a fixed sequence, and inside fvg/ob it walks
the box list in index order emitting the birth token and the *_used token in the
same pass. Two consequences, both reproduced below:

  * the nine features are polled in _rows order every bar;
  * inside fvg/ob, the *_used tokens of older boxes come BEFORE the birth token
    of a box confirmed on this bar (a box cannot mitigate on its own confirm bar,
    so every *_used belongs to a lower index).

tf_slot is NOT a property of a timeframe - 1H is slot 2 in the 1H cell, slot 0
in the 15m cell and slot 1 in the 30m cell - so the slot one-hot is stamped at
materialize() time, not stored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..killzone import clock_features
from ..labels import TF_MIN
from ..tokens import N_STATIC, TID, TYPES


class TokenStream:
    def __init__(self, tf: str) -> None:
        self.tf = tf
        self._at: list = []
        self._tid: list[int] = []
        self._level: list[float] = []
        self._size: list[float] = []
        self._seen = {k: 0 for k in ("swing", "trend", "sweep", "eqhl",
                                     "fvg", "ob", "fib")}
        self.visits = 0

    def _add(self, at, t: str, level: float, size: float = 0.0) -> None:
        self._at.append(np.datetime64(at, "ns"))
        self._tid.append(TID[t])
        self._level.append(float(level))
        self._size.append(float(size))
        self.visits += 1

    def feed(self, i: int, M: dict) -> None:
        s = self._seen
        while s["swing"] < len(M["swing"].markers):
            m = M["swing"].markers[s["swing"]]
            if m["cls"] in ("HH", "HL", "LH", "LL"):
                self._add(m["confirm_at"], f"sw_{m['cls']}", m["price"])
            s["swing"] += 1

        dt_of = M["erl_irl"].dt
        for m in M["erl_irl"].items_at_bar():
            if m["role"] == "ERL" and m["confirm"] == i:
                self._add(dt_of[i], "erl_up" if m["side"] == "high" else "erl_dn", m["price"])
            elif m["promote"] == i:
                self._add(dt_of[i], "promo_up" if m["side"] == "high" else "promo_dn",
                          m["price"])

        while s["trend"] < len(M["trend"].labels):
            lab = M["trend"].labels[s["trend"]]
            key = {"E-BOS": "ebos", "E-MSS": "emss", "i-BOS": "ibos", "i-MSS": "imss"}[lab["kind"]]
            self._add(lab["confirm_at"], f"{key}_{'up' if lab['dir'] == 'up' else 'dn'}",
                      lab["level"])
            s["trend"] += 1

        while s["sweep"] < len(M["sweep"].markers):
            m = M["sweep"].markers[s["sweep"]]
            self._add(m["confirm_at"], "sweep_ssl" if m["side"] == "SSL" else "sweep_bsl",
                      m["price"])
            s["sweep"] += 1

        while s["eqhl"] < len(M["eqhl"].events):
            e = M["eqhl"].events[s["eqhl"]]
            self._add(e["confirm_at"], "eqh" if e["kind"] == "EQH" else "eql",
                      (e["p1"] + e["p2"]) / 2)
            s["eqhl"] += 1

        for k in M["fvg"].died_at_bar():                 # older boxes first
            b = M["fvg"].boxes[k]
            self._add(b["mit_at"], "fvg_used", (b["lo"] + b["hi"]) / 2)
        while s["fvg"] < len(M["fvg"].boxes):
            b = M["fvg"].boxes[s["fvg"]]
            self._add(b["confirm_at"], "fvg_bull" if b["side"] == "bull" else "fvg_bear",
                      (b["lo"] + b["hi"]) / 2, abs(b["hi"] - b["lo"]))
            s["fvg"] += 1

        for k in M["ob"].died_at_bar():
            b = M["ob"].boxes[k]
            self._add(b["mit_at"], "ob_used", (b["lo"] + b["hi"]) / 2)
        while s["ob"] < len(M["ob"].boxes):
            b = M["ob"].boxes[s["ob"]]
            self._add(b["confirm_at"], "ob_bull" if b["side"] == "bull" else "ob_bear",
                      (b["lo"] + b["hi"]) / 2, abs(b.get("gap", 0.0)))
            s["ob"] += 1

        for k in M["liquidity"].died_at_bar():
            q = M["liquidity"].lines[k]
            self._add(q["swept_at"],
                      "liq_swept_ssl" if q["side"] == "SSL" else "liq_swept_bsl", q["level"])

        while s["fib"] < len(M["fib"].fibs):
            f = M["fib"].fibs[s["fib"]]
            self._add(f["confirm_at"], "fib_up" if f["dir"] == "up" else "fib_dn",
                      f["end_price"], abs(f["start_price"] - f["end_price"]))
            s["fib"] += 1

    def stream_tail(self, tf_slot: int, keep: int = 256) -> dict:
        """The newest `keep` tokens, in materialize()'s shape.

        sequences() takes the last SEQ_CAP (96) tokens whose knowability is at or
        before the plan's, and every token this stream holds is already knowable,
        so any tail at least SEQ_CAP long gives the identical tensor.
        """
        n = len(self._at)
        if n == 0:
            return self.materialize(tf_slot)
        a = max(0, n - keep)
        sub = TokenStream(self.tf)
        sub._at, sub._tid = self._at[a:], self._tid[a:]
        sub._level, sub._size = self._level[a:], self._size[a:]
        return sub.materialize(tf_slot)

    def cap(self, keep: int = 256) -> None:
        """sequences() reads at most SEQ_CAP (96) tokens, so anything older than
        a comfortable multiple of that is unreachable. Trim only past twice the
        target so the cost stays amortized O(1) rather than O(n) every bar."""
        n = len(self._at)
        if n <= 2 * keep:
            return
        self._at = self._at[-keep:]
        self._tid = self._tid[-keep:]
        self._level = self._level[-keep:]
        self._size = self._size[-keep:]

    def materialize(self, tf_slot: int) -> dict:
        """Same arrays legacy tf_tokens returns, for this slot."""
        n = len(self._at)
        if n == 0:
            # tf_min belongs here too: a live cell reads this before the coarse
            # timeframe has produced anything, and merged_stream needs the value
            # to build its per-token array. (The legacy builder omits it, but no
            # batch tape is ever empty so nothing there could notice.)
            return {"known": np.array([], "datetime64[ns]"), "level": np.zeros(0),
                    "static": np.zeros((0, N_STATIC), np.float32),
                    "tf_min": float(TF_MIN[self.tf])}
        at = np.array(self._at, "datetime64[ns]")
        tid = np.array(self._tid, np.int64)
        level = np.array(self._level, float)
        size = np.array(self._size, float)
        known = at + np.timedelta64(int(TF_MIN[self.tf]), "m")
        static = np.zeros((n, N_STATIC), np.float32)
        static[np.arange(n), tid] = 1.0
        static[:, len(TYPES) + tf_slot] = 1.0
        rel = size / np.maximum(np.abs(level), 1e-9)
        static[:, len(TYPES) + 3] = np.log1p(rel * 1000.0)
        clock, _ = clock_features(pd.DatetimeIndex(at), TF_MIN[self.tf])
        static[:, len(TYPES) + 4:] = clock
        return {"known": known, "level": level, "static": static,
                "tf_min": float(TF_MIN[self.tf])}
