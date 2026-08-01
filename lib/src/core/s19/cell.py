"""CellEngine - reads one cell's answers out of a StoreBook.

P3 had this class build and feed its own stores. Once the cells of a symbol
share stores (see book.py), feeding here would feed the same store once per
cell, so ingest moved to StoreBook and this became a reader. It keeps the two
things that ARE per-cell: which timeframes it looks at, and whether its own path
bar just confirmed an ERL break.

The merged token stream is append-only. That is only correct because groups
arrive in close-time order - StoreBook asserts it rather than trusting it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .book import tf_streams_for
from .killzone import kz_onehot
from .mtf import CELLS, block, closed_index
from .stage2 import LADDER, TF_MIN, plan_band
from .tokens import sequences


class CellEngine:
    def __init__(self, cell: str, book) -> None:
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {tuple(CELLS)}")
        self.cell = cell
        self.book = book
        self.ctx_tfs = CELLS[cell]
        self.ladder_tfs = LADDER[cell]
        for tf, want in tf_streams_for((cell,)).items():
            st = book.stores.get(tf)
            if st is None:
                raise ValueError(
                    f"cell {cell} needs a {tf} store; this book has "
                    f"{sorted(book.stores)}")
            missing = want - st.enabled
            if missing:
                raise ValueError(
                    f"cell {cell} needs {sorted(missing)} on the {tf} store, "
                    f"which carries {sorted(st.enabled)}")
        # References into the book, not copies - W11 asserts the identity,
        # because that identity IS the memory saving.
        self.stores = {tf: book.stores[tf]
                       for tf in set(self.ctx_tfs) | set(self.ladder_tfs)}
        self._fresh_break = False

    # -- the one per-cell decision -------------------------------------------
    def observe(self, fed) -> None:
        """After the book has taken a whole group: did MY path bar confirm a new
        ERL break? ~99% of groups: no."""
        path = self.ctx_tfs[2]
        if path not in fed:
            self._fresh_break = False
            return
        st = self.stores[path]
        i = st.n - 1
        self._fresh_break = any(
            m["role"] == "ERL" and m["confirm"] == i
            for m in st.mirrors["erl_irl"].items_at_bar())

    def gate(self) -> bool:
        return self._fresh_break

    def latest_plan(self):
        return self.stores[self.ctx_tfs[2]].latest_plan()

    # -- model inputs ---------------------------------------------------------
    def _known_ts(self, plan) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([plan["known_at"]]) + \
            pd.Timedelta(minutes=TF_MIN[self.ctx_tfs[2]])

    def ctx_row(self, plan, ref: float) -> np.ndarray:
        """assemble()'s row for this one plan, from one row per timeframe."""
        known_ts = self._known_ts(plan)
        refs = np.array([ref], float)
        blocks = []
        for tf in self.ctx_tfs:
            tail = self.stores[tf].tail_state()
            idx = closed_index(tail["dt"], TF_MIN[tf], known_ts)
            blocks.append(block(tail, idx, refs))
        blocks.append(kz_onehot(known_ts))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def seq_tail(self, plan, ref: float, width: float):
        """The same (seq, lens) tokens.sequences returns for this plan."""
        known_ts = self._known_ts(plan)
        parts = [self.stores[tf].stream_tail(i) for i, tf in enumerate(self.ctx_tfs)]
        known = np.concatenate([p["known"] for p in parts])
        level = np.concatenate([p["level"] for p in parts])
        static = np.concatenate([p["static"] for p in parts], axis=0)
        tfmin = np.concatenate([np.full(len(p["known"]), p["tf_min"]) for p in parts])
        order = np.argsort(known, kind="stable")     # htf, mtf, path on ties
        stream = {"known": known[order], "level": level[order],
                  "static": static[order], "tf_min": tfmin[order]}
        return sequences(stream, known_ts, np.array([ref], float),
                         np.array([width], float))

    def candidates(self, plan, ref: float) -> list[dict]:
        known_ts = self._known_ts(plan)[0]
        blo, bhi, up = plan_band(plan, ref)
        out: list[dict] = []
        for tf in self.ladder_tfs:
            out.extend(self.stores[tf].virgin_in_band(known_ts, blo, bhi, up))
        return out
