"""TfStateStream == the per-bar loop inside mtf.tf_state.

The legacy loop reads two streams the ERL/IRL feature produces: the ERL breaks
(for direction, how long since, and how many in the last 200 bars) and the edge
stream (for the current range high/low and whether each was promoted). Both come
straight off the IncrErlIrl mirror here, so erl_events is not recomputed.

Order matters inside a bar: the legacy code drains the break cursor completely
before the edge cursor. And the edge stream is sorted by (knowability, origin
bar) - everything generated on this bar shares a knowability, so the mirror's
items_at_bar(), already sorted by origin bar, is that order exactly.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class TfStateStream:
    def __init__(self) -> None:
        self.last_dir: list[int] = []
        self.since: list[float] = []
        self.n_recent: list[float] = []
        self.hi: list[float] = []
        self.lo: list[float] = []
        self.hi_pro: list[float] = []
        self.lo_pro: list[float] = []
        self._dir = 0
        self._bar = -1
        self._hi = np.nan
        self._lo = np.nan
        self._hip = 0.0
        self._lop = 0.0
        self._stamps: deque[int] = deque()
        self.stamps_ops = 0
        self.n_breaks = 0

    def feed(self, i: int, erl_born: list[dict], erl_items: list[dict]) -> None:
        for m in erl_born:                      # breaks first, as the legacy loop does
            if m["role"] == "ERL":
                self._dir = 1 if m["side"] == "high" else -1
                self._bar = m["confirm"]
                self._stamps.append(self._bar)
                self.stamps_ops += 1
                self.n_breaks += 1
        for m in erl_items:                     # then the edge stream, in origin order
            promote = m["promote"] == i
            price, kind = m["price"], ("promote" if promote else "break")
            if m["side"] == "high":
                self._hi, self._hip = price, 1.0 if kind == "promote" else 0.0
            else:
                self._lo, self._lop = price, 1.0 if kind == "promote" else 0.0

        self.last_dir.append(self._dir)
        self.since.append(float(i - self._bar) if self._bar >= 0 else -1.0)
        while self._stamps and self._stamps[0] < i - 200:
            self._stamps.popleft()
            self.stamps_ops += 1
        self.n_recent.append(float(len(self._stamps)))
        self.hi.append(self._hi)
        self.lo.append(self._lo)
        self.hi_pro.append(self._hip)
        self.lo_pro.append(self._lop)

    _KEEP = 1

    def cap(self) -> None:
        """Drop rows no future plan can see: live only ever reads the newest."""
        if len(self.last_dir) <= 2 * self._KEEP:
            return
        k = self._KEEP
        self.last_dir = self.last_dir[-k:]
        self.since = self.since[-k:]
        self.n_recent = self.n_recent[-k:]
        self.hi = self.hi[-k:]
        self.lo = self.lo[-k:]
        self.hi_pro = self.hi_pro[-k:]
        self.lo_pro = self.lo_pro[-k:]

    def carry(self) -> dict:
        """The running state itself, for consumers that need the plan rather than
        the row - the edges here are exactly what _latest_plan replays to find."""
        return {"dir": self._dir, "bar": self._bar, "hi": self._hi, "lo": self._lo,
                "hi_pro": self._hip, "lo_pro": self._lop,
                "since": self.since[-1] if self.since else -1.0,
                "n_recent": self.n_recent[-1] if self.n_recent else 0.0}

    def last_row(self) -> dict:
        return {"last_dir": np.array(self.last_dir[-1:], np.int8),
                "since": np.array(self.since[-1:], float),
                "n_recent": np.array(self.n_recent[-1:], np.float32),
                "hi": np.array(self.hi[-1:], float),
                "lo": np.array(self.lo[-1:], float),
                "hi_pro": np.array(self.hi_pro[-1:], np.float32),
                "lo_pro": np.array(self.lo_pro[-1:], np.float32)}

    def arrays(self) -> dict:
        return {"last_dir": np.array(self.last_dir, np.int8),
                "since": np.array(self.since, float),
                "n_recent": np.array(self.n_recent, np.float32),
                "hi": np.array(self.hi, float),
                "lo": np.array(self.lo, float),
                "hi_pro": np.array(self.hi_pro, np.float32),
                "lo_pro": np.array(self.lo_pro, np.float32)}
