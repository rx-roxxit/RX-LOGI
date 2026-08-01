"""StoreBook - the FeatureStores of one symbol, owned in one place and fed once.

P3 gave every CellEngine its own stores, so a process serving all three cells
held 15 where there are only 7 distinct timeframes: 1,261 MB measured over
2015-2025 against a union of about 754 MB, and 4,263k bar-steps of ingest
against a union of 2,555k.

Sharing is safe for a reason worth stating out loud: not one store changes
shape. CellEngine already calls enable(LADDER_STREAMS) when a timeframe is both
context and ladder inside its own cell, so the merged shape a shared store gets
is a shape that already exists and is already walled. test_s19_book.py holds
that claim to a test rather than leaving it an argument.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .mtf import CELLS
from .stage2 import LADDER, TF_MIN
from .store import FeatureStore

CTX_STREAMS = {"tf_state", "zone_ctx", "fib_ctx", "tokens"}
LADDER_STREAMS = {"zones"}


def tf_streams_for(cells: Iterable[str]) -> dict[str, set[str]]:
    """The streams each timeframe must carry to serve every cell in `cells`.

    This is the whole of the sharing logic: a timeframe that is context
    somewhere and ladder somewhere else carries both.
    """
    out: dict[str, set[str]] = {}
    for cell in cells:
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {tuple(CELLS)}, got {cell!r}")
        for tf in CELLS[cell]:
            out.setdefault(tf, set()).update(CTX_STREAMS)
        for tf in LADDER[cell]:
            out.setdefault(tf, set()).update(LADDER_STREAMS)
    return out


class StoreBook:
    """The stores of one symbol, and the only thing that feeds them.

    P3 put ingest inside CellEngine, which was right when a cell owned its
    stores. Once they are shared, three cells calling extend_group would feed
    the same store three times, so ingest lives here and CellEngine reads.
    That is a structural guarantee rather than a convention: there is no
    other way in.
    """

    def __init__(self, cells, retain: str = "live") -> None:
        self.cells = tuple(cells)
        if not self.cells:
            raise ValueError("a book needs at least one cell")
        self.retain = retain
        self.stores: dict[str, FeatureStore] = {
            tf: FeatureStore(tf, streams, retain=retain)
            for tf, streams in tf_streams_for(self.cells).items()}
        self._last_close = None
        self._broken = None

    def extend_group(self, items) -> set[str]:
        """Take every bar that closed on ONE instant; feed each store once.

        The group is the unit because the as-of rule is inclusive at the
        instant: a zone whose confirm bar closes at known_ts is a candidate,
        and a context bar closing at known_ts is visible. At 08:30 the 3m, 5m,
        15m and 30m bars close together and the path timeframe sits in the
        middle, so no per-bar order can be right. See replay.replay_feed.

        Validation covers every bar in the group, including timeframes this
        book does not hold - a group that spans two instants is malformed
        whoever it was meant for. Only feeding skips the unheld ones.

        Validation failures leave the clock untouched and feed nothing, so the
        group can be retried. A mid-feed failure poisons the book because
        FeatureStore has no undo; every later call refuses to answer.

        items = [(tf, (timestamp, open, high, low, close)), ...]
        Returns the timeframes actually fed.
        """
        if not items:
            return set()
        if self._broken:
            raise RuntimeError(self._broken)
        for tf, _bar in items:
            if tf not in TF_MIN:
                raise RuntimeError(
                    f"unknown timeframe {tf!r}; known: {sorted(TF_MIN)}")
        closes = {pd.Timestamp(bar[0]) + pd.Timedelta(minutes=TF_MIN[tf])
                  for tf, bar in items}
        if len(closes) != 1:
            raise RuntimeError(
                f"a group must be the bars that closed on ONE instant; got "
                f"{sorted(closes)}. See replay.replay_feed.")
        close = closes.pop()
        if self._last_close is not None and close <= self._last_close:
            raise RuntimeError(
                f"groups must arrive in strictly increasing close time; got "
                f"{close} after {self._last_close}. See replay.replay_feed.")
        fed: set[str] = set()
        try:
            for tf, bar in items:
                st = self.stores.get(tf)
                if st is None:                  # a timeframe no cell here needs
                    continue
                ts, o, h, l, c = bar
                st.extend([ts], [o], [h], [l], [c])
                fed.add(tf)
        except Exception as exc:
            # Some stores took this bar and some did not, and FeatureStore has
            # no undo. The book can no longer answer anything truthfully, so it
            # stops answering rather than answering wrongly.
            self._broken = (
                f"a group at {close} failed part way through, after feeding "
                f"{sorted(fed)}: {exc!r}. The stores now disagree about what "
                f"time it is and there is no way to roll that back.")
            raise
        self._last_close = close
        return fed

    # -- totals: read these, never sum a cell's view (shared stores double-count)
    @property
    def tombstoned(self) -> int:
        return sum(st.tombstoned for st in self.stores.values())

    @property
    def live_objects(self) -> int:
        return sum(st.live_objects for st in self.stores.values())

    @property
    def bar_steps(self) -> int:
        """Bars fed across every store - the ingest cost, as a count."""
        return sum(st.n for st in self.stores.values())
