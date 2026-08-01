"""FeatureStore - one per (symbol, timeframe), the thing that turns the mirrors
into the arrays the model pipeline asks for.

Two economies matter here and both come from reading the call graph rather than
guessing at it:

  * ONE swing stream. Every frozen module that needs swings builds its own, so
    P1's mirrors do too. Here they share one, which is what turns the five
    mirrors that were slower than frozen back into a win.

  * STREAMS ARE OPT-IN. assemble() only ever reads the cell's three context
    timeframes, and zones_on_tf only ever reads the ladder's three - so a 3m or
    5m store exists purely to answer `zones` and never builds a token or a zone
    context, and only the mirrors those streams need are constructed at all.

The buffers are plain lists so the mirrors, which read by index, work the same
whether the tape arrives in one call or one bar at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .incr import (IncrEqhl, IncrErlIrl, IncrFib, IncrFvg, IncrLiquidity, IncrOb,
                   IncrSweep, IncrSwing, IncrTrend)
from .streams import (FibCtxStream, GrowableF64, TfStateStream, TokenStream,
                      ZoneCtxStream, ZoneStream)

STREAM_NAMES = ("tf_state", "zone_ctx", "fib_ctx", "tokens", "zones")

# stream -> the mirrors it reads (swing is implicit; every mirror below needs it)
_NEEDS = {
    "tf_state": ("erl_irl",),
    "zone_ctx": ("fvg", "ob", "liquidity", "eqhl"),
    "fib_ctx": ("erl_irl", "fib"),
    "tokens": ("erl_irl", "trend", "sweep", "eqhl", "fvg", "ob", "liquidity", "fib"),
    "zones": ("fvg", "ob", "liquidity", "eqhl"),
}
_ROWS_ORDER = ("swing", "erl_irl", "trend", "sweep", "eqhl", "fvg", "ob", "liquidity", "fib")


class FeatureStore:
    _SWEEP_EVERY = 4096          # amortize the bucket walk; it is O(objects)

    def __init__(self, tf: str, streams: set[str], retain: str = "all") -> None:
        if retain not in ("all", "live"):
            raise ValueError(f"retain must be 'all' or 'live', got {retain!r}")
        self.retain = retain
        self.tombstoned = 0
        self._since_sweep = 0
        bad = set(streams) - set(STREAM_NAMES)
        if bad:
            raise ValueError(f"unknown streams {sorted(bad)}; expected {STREAM_NAMES}")
        self.tf = tf
        self.enabled = set(streams)
        self.n = 0
        self.dt: list = []                     # MUST stay list[pd.Timestamp]
        self.o = GrowableF64()
        self.h = GrowableF64()
        self.l = GrowableF64()
        self.c = GrowableF64()

        want = {"swing"}
        for s in self.enabled:
            want.update(_NEEDS[s])
        if "fib" in want:
            want.add("erl_irl")

        B = (self.dt, self.o, self.h, self.l, self.c)
        self.mirrors: dict = {"swing": IncrSwing(*B)}
        sw = self.mirrors["swing"]
        if "erl_irl" in want:
            self.mirrors["erl_irl"] = IncrErlIrl(*B, swings=sw)
        for name, cls in (("trend", IncrTrend), ("sweep", IncrSweep), ("eqhl", IncrEqhl),
                          ("liquidity", IncrLiquidity)):
            if name in want:
                self.mirrors[name] = cls(*B, swings=sw)
        for name, cls in (("fvg", IncrFvg), ("ob", IncrOb)):
            if name in want:
                self.mirrors[name] = cls(*B)
        if "fib" in want:
            self.mirrors["fib"] = IncrFib(*B, erl=self.mirrors["erl_irl"])

        self.streams: dict = {}
        if "tf_state" in self.enabled:
            self.streams["tf_state"] = TfStateStream()
        if "zone_ctx" in self.enabled:
            self.streams["zone_ctx"] = ZoneCtxStream()
        if "fib_ctx" in self.enabled:
            self.streams["fib_ctx"] = FibCtxStream()
        if "tokens" in self.enabled:
            self.streams["tokens"] = TokenStream(tf)
        if "zones" in self.enabled:
            self.streams["zones"] = ZoneStream()

    def enable(self, streams: set[str]) -> None:
        """Turn on more streams before any bar has been fed. A store that has
        already ingested cannot grow a stream - the new one would start blind."""
        if self.n:
            raise RuntimeError("enable() must be called before the first bar")
        bad = set(streams) - set(STREAM_NAMES)
        if bad:
            raise ValueError(f"unknown streams {sorted(bad)}")
        new = set(streams) - self.enabled
        if not new:
            return
        self.enabled |= new
        want: set[str] = set()
        for s in new:
            want.update(_NEEDS[s])
        if "fib" in want:
            want.add("erl_irl")
        B = (self.dt, self.o, self.h, self.l, self.c)
        sw = self.mirrors["swing"]
        if "erl_irl" in want and "erl_irl" not in self.mirrors:
            self.mirrors["erl_irl"] = IncrErlIrl(*B, swings=sw)
        for name, cls in (("trend", IncrTrend), ("sweep", IncrSweep),
                          ("eqhl", IncrEqhl), ("liquidity", IncrLiquidity)):
            if name in want and name not in self.mirrors:
                self.mirrors[name] = cls(*B, swings=sw)
        for name, cls in (("fvg", IncrFvg), ("ob", IncrOb)):
            if name in want and name not in self.mirrors:
                self.mirrors[name] = cls(*B)
        if "fib" in want and "fib" not in self.mirrors:
            self.mirrors["fib"] = IncrFib(*B, erl=self.mirrors["erl_irl"])
        if "tf_state" in new:
            self.streams["tf_state"] = TfStateStream()
        if "zone_ctx" in new:
            self.streams["zone_ctx"] = ZoneCtxStream()
        if "fib_ctx" in new:
            self.streams["fib_ctx"] = FibCtxStream()
        if "tokens" in new:
            self.streams["tokens"] = TokenStream(self.tf)
        if "zones" in new:
            self.streams["zones"] = ZoneStream()

    # ── ingest ───────────────────────────────────────────────────────────────
    def extend(self, dt, o, h, l, c) -> None:
        """Append bars. Any number, including one."""
        idx = pd.DatetimeIndex(dt)
        for k in range(len(idx)):
            self.dt.append(idx[k])
            self.o.append(float(o[k]))
            self.h.append(float(h[k]))
            self.l.append(float(l[k]))
            self.c.append(float(c[k]))
            self._step(self.n)
            self.n += 1

    def _step(self, i: int) -> None:
        M = self.mirrors
        born: dict[str, list] = {}
        M["swing"].feed(i)
        if "erl_irl" in M:                       # fib reads erl, so erl goes first
            born["erl_irl"] = M["erl_irl"].feed(i)
        for name in _ROWS_ORDER:
            if name in ("swing", "erl_irl") or name not in M:
                continue
            born[name] = M[name].feed(i)
        S = self.streams
        if "tf_state" in S:
            S["tf_state"].feed(i, born["erl_irl"], M["erl_irl"].items_at_bar())
        if "zone_ctx" in S:
            S["zone_ctx"].feed(i, self.c, M["fvg"], M["ob"], M["liquidity"], M["eqhl"])
        if "fib_ctx" in S:
            S["fib_ctx"].feed(i, self.h, self.l, self.c, born["fib"])
        if "tokens" in S:
            S["tokens"].feed(i, M)
        if "zones" in S:
            S["zones"].feed(i, self.h, self.l, self.c, M["fvg"], M["ob"],
                            M["liquidity"], M["eqhl"])
        if self.retain == "live":
            self._reap(i)

    def _reap(self, i: int) -> None:
        """Give back the memory of everything that died on this bar - AFTER every
        stream has read it (TokenStream reads a box at the moment it dies to emit
        its *_used token)."""
        M = self.mirrors
        for name, attr in (("fvg", "boxes"), ("ob", "boxes"), ("liquidity", "lines")):
            m = M.get(name)
            if m is None:
                continue
            objs = getattr(m, attr)
            for k in m.died_at_bar():
                if objs[k] is not None:
                    objs[k] = None
                    self.tombstoned += 1
        z = self.streams.get("zones")
        if z is not None:
            z.reap()
            self._since_sweep += 1
            if self._since_sweep >= self._SWEEP_EVERY:
                self.tombstoned += z.tombstone_buckets()
                self._since_sweep = 0
        for name in ("tf_state", "zone_ctx", "fib_ctx"):
            s = self.streams.get(name)
            if s is not None:
                s.cap()
        t = self.streams.get("tokens")
        if t is not None:
            t.cap()

    @property
    def live_objects(self) -> int:
        z = self.streams.get("zones")
        return z.live_count if z is not None else 0

    def _batch_only(self, name: str) -> None:
        if self.retain == "live":
            raise RuntimeError(
                f"{name}() needs the whole history but this store was built with "
                f"retain='live', which drops what no future plan can see. Use the "
                f"live queries instead: latest_plan / tail_state / stream_tail / "
                f"virgin_in_band.")

    # ── read ─────────────────────────────────────────────────────────────────
    def tf_state(self) -> dict:
        self._batch_only("tf_state")
        st = dict(self.streams["tf_state"].arrays())
        zs, zn = self.zone_ctx()
        fs, fn = self.fib_ctx()
        st.update({"zone": zs, "zone_names": zn, "fib": fs, "fib_names": fn,
                   "close": self.c.array.copy(),
                   "dt": pd.DatetimeIndex(self.dt), "tf": self.tf})
        return st

    def tail_state(self) -> dict:
        """tf_state() shaped for one bar - the newest. mtf.block() takes it as-is,
        which is why the keys and dtypes match exactly."""
        from .context import FIB_FEATS, ZONE_FEATS
        if self.n == 0:
            # No bar of this timeframe has closed yet - the daily bar of the
            # first session does not close until the next day, while the path
            # timeframe is already confirming breaks. block() is built for
            # exactly this: the dt below has not closed, so closed_index returns
            # -1 and the row comes out zeroed, which is what assemble() does.
            return {"last_dir": np.zeros(1, np.int8), "since": np.full(1, -1.0),
                    "n_recent": np.zeros(1, np.float32), "hi": np.full(1, np.nan),
                    "lo": np.full(1, np.nan), "hi_pro": np.zeros(1, np.float32),
                    "lo_pro": np.zeros(1, np.float32),
                    "zone": np.zeros((1, len(ZONE_FEATS)), np.float32),
                    "zone_names": ZONE_FEATS,
                    "fib": np.zeros((1, len(FIB_FEATS)), np.float32),
                    "fib_names": FIB_FEATS,
                    "close": np.zeros(1, float),
                    "dt": pd.DatetimeIndex([pd.Timestamp("2100-01-01")]),
                    "tf": self.tf}
        st = dict(self.streams["tf_state"].last_row())
        st.update({"zone": self.streams["zone_ctx"].last_row(), "zone_names": ZONE_FEATS,
                   "fib": self.streams["fib_ctx"].last_row(), "fib_names": FIB_FEATS,
                   "close": np.array([self.c[self.n - 1]], float),
                   "dt": pd.DatetimeIndex([self.dt[self.n - 1]]), "tf": self.tf})
        return st

    def latest_plan(self) -> dict | None:
        """The plan at the most recent ERL break, from the carried edges rather
        than a replay. Same fields inference._latest_plan returns."""
        k = self.streams["tf_state"].carry()
        if k["bar"] < 0 or k["hi"] != k["hi"] or k["lo"] != k["lo"]:      # NaN check
            return None
        up = k["dir"] > 0
        return {"known_at": self.dt[k["bar"]], "known_bar": int(k["bar"]),
                "state": "HH" if up else "LL",
                "erl_hi": float(k["hi"]), "erl_lo": float(k["lo"])}

    def zone_ctx(self):
        self._batch_only("zone_ctx")
        return self.streams["zone_ctx"].arrays()

    def fib_ctx(self):
        self._batch_only("fib_ctx")
        return self.streams["fib_ctx"].arrays()

    def tokens(self, tf_slot: int) -> dict:
        self._batch_only("tokens")
        return self.streams["tokens"].materialize(tf_slot)

    def zones(self) -> list[dict]:
        self._batch_only("zones")
        return self.streams["zones"].zones(self.n)

    def stream_tail(self, tf_slot: int, keep: int = 256) -> dict:
        """Token tail for live mode - no retain restriction."""
        return self.streams["tokens"].stream_tail(tf_slot, keep)

    def virgin_in_band(self, known_ts, band_lo: float, band_hi: float,
                       up: bool) -> list[dict]:
        """Candidates on THIS timeframe, tagged with it. No retain restriction -
        this is the query live mode is meant to use."""
        return self.streams["zones"].virgin_in_band(
            pd.DatetimeIndex(self.dt), self.tf, known_ts, band_lo, band_hi, up)

    @property
    def visits(self) -> dict:
        out = {}
        for name, s in self.streams.items():
            v = getattr(s, "visits", None)
            if v is not None:
                out[name] = v
        return out
