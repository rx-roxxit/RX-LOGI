"""Incremental mirror of app/features/trend.py.

trend.py embeds its own copy of the ERL/IRL promotion state (it does not call
erl_irl), so this mirror embeds it too - deliberately, to stay a transcription
rather than a refactor. Within one bar the EXTERNAL label is appended before any
INTERNAL label, and note_state() runs last; that order is part of the output.
"""
from __future__ import annotations

from .swing import IncrSwing

_ST = {1: "BULL", -1: "BEAR", 0: "WAITING"}


class IncrTrend:
    def __init__(self, dt, o, h, l, c, *, swings: IncrSwing | None = None) -> None:
        self.dt, self.c = dt, c
        self.sw = swings if swings is not None else IncrSwing(dt, o, h, l, c)
        self._own_swings = swings is None
        self.labels: list[dict] = []
        self.states: list[dict] = []
        self._erl_tr = 0
        self._has_hi = self._has_lo = False
        self._erl_hi = self._erl_lo = 0.0
        self._erl_hi_bar = self._erl_lo_bar = 0
        self._has_lo_irl = self._has_hi_irl = False
        self._lo_irl_v = self._hi_irl_v = 0.0
        self._lo_irl_bar = self._hi_irl_bar = 0
        self._ext = 0
        self._int = 0
        self._has_int_hi = self._has_int_lo = False
        self._int_hi_broken = self._int_lo_broken = False
        self._int_hi = self._int_lo = 0.0
        self._int_hi_bar = self._int_lo_bar = 0

    def _reset_internal(self) -> None:
        self._has_int_hi = self._has_int_lo = False
        self._int_hi_broken = self._int_lo_broken = False
        self._int = 0

    def _note_state(self, i: int) -> None:
        cur = {"confirm_at": self.dt[i], "ext": _ST[self._ext], "int": _ST[self._int]}
        if not self.states or self.states[-1]["ext"] != cur["ext"] \
                or self.states[-1]["int"] != cur["int"]:
            self.states.append(cur)

    def feed(self, i: int) -> list[dict]:
        if self._own_swings:
            self.sw.feed(i)
        dt, c = self.dt, self.c
        born: list[dict] = []
        ev = self.sw.last
        if ev is not None and ev["confirm"] == i and ev["side"] == "high":
            hb, v = ev["bar"], ev["price"]
            if not self._has_hi:
                self._erl_hi = v; self._erl_hi_bar = hb; self._has_hi = True; self._erl_tr = 1
            elif v > self._erl_hi:
                mss = self._ext == -1
                lab = {"at": dt[hb], "confirm_at": dt[i], "level": float(self._erl_hi),
                       "from_at": dt[self._erl_hi_bar],
                       "kind": "E-MSS" if mss else "E-BOS", "dir": "up"}
                self.labels.append(lab); born.append(lab)
                self._ext = 0 if mss else 1
                self._erl_hi = v; self._erl_hi_bar = hb
                if self._erl_tr == 1 and self._has_lo_irl:
                    self._erl_lo = self._lo_irl_v
                    self._erl_lo_bar = self._lo_irl_bar
                    self._has_lo = True
                self._erl_tr = 1
                self._has_lo_irl = self._has_hi_irl = False
                self._reset_internal()
            else:
                if not self._has_hi_irl or v > self._hi_irl_v:
                    self._hi_irl_v = v; self._hi_irl_bar = hb; self._has_hi_irl = True
                self._int_hi = v; self._int_hi_bar = hb
                self._has_int_hi = True; self._int_hi_broken = False
        elif ev is not None and ev["confirm"] == i:
            lb, v = ev["bar"], ev["price"]
            if not self._has_lo:
                self._erl_lo = v; self._erl_lo_bar = lb; self._has_lo = True; self._erl_tr = -1
            elif v < self._erl_lo:
                mss = self._ext == 1
                lab = {"at": dt[lb], "confirm_at": dt[i], "level": float(self._erl_lo),
                       "from_at": dt[self._erl_lo_bar],
                       "kind": "E-MSS" if mss else "E-BOS", "dir": "down"}
                self.labels.append(lab); born.append(lab)
                self._ext = 0 if mss else -1
                self._erl_lo = v; self._erl_lo_bar = lb
                if self._erl_tr == -1 and self._has_hi_irl:
                    self._erl_hi = self._hi_irl_v
                    self._erl_hi_bar = self._hi_irl_bar
                    self._has_hi = True
                self._erl_tr = -1
                self._has_lo_irl = self._has_hi_irl = False
                self._reset_internal()
            else:
                if not self._has_lo_irl or v < self._lo_irl_v:
                    self._lo_irl_v = v; self._lo_irl_bar = lb; self._has_lo_irl = True
                self._int_lo = v; self._int_lo_bar = lb
                self._has_int_lo = True; self._int_lo_broken = False

        if self._has_int_hi and not self._int_hi_broken and c[i] > self._int_hi \
                and (not self._has_hi or c[i] < self._erl_hi):
            mss = self._int == -1
            lab = {"at": dt[i], "confirm_at": dt[i], "level": float(self._int_hi),
                   "from_at": dt[self._int_hi_bar],
                   "kind": "i-MSS" if mss else "i-BOS", "dir": "up"}
            self.labels.append(lab); born.append(lab)
            self._int = 0 if mss else 1
            self._int_hi_broken = True
        if self._has_int_lo and not self._int_lo_broken and c[i] < self._int_lo \
                and (not self._has_lo or c[i] > self._erl_lo):
            mss = self._int == 1
            lab = {"at": dt[i], "confirm_at": dt[i], "level": float(self._int_lo),
                   "from_at": dt[self._int_lo_bar],
                   "kind": "i-MSS" if mss else "i-BOS", "dir": "down"}
            self.labels.append(lab); born.append(lab)
            self._int = 0 if mss else -1
            self._int_lo_broken = True

        self._note_state(i)
        return born

    def result(self) -> dict:
        return {"labels": self.labels, "states": self.states}
