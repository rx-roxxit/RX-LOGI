"""A float64 array that grows, and that a mirror can index like a list.

The mirrors read bars by position - self.h[i], self.h[self.hi_bar] - so the
buffer only has to answer __getitem__ with an int. Holding them as a numpy
array instead of a Python list costs 8 bytes a bar instead of about 32, which
on a 3m tape is the difference between 208 MB and 52 MB per store.

`dt` deliberately does NOT use this: dt[i] flows into the frozen features'
output as confirm_at/mit_at, and a datetime64 there is not a Timestamp - the
parity wall compares dtypes.
"""
from __future__ import annotations

import numpy as np


class GrowableF64:
    __slots__ = ("_a", "_n")

    def __init__(self, cap: int = 1024) -> None:
        self._a = np.empty(max(cap, 1), np.float64)
        self._n = 0

    def append(self, x: float) -> None:
        if self._n == len(self._a):
            bigger = np.empty(len(self._a) * 2, np.float64)
            bigger[:self._n] = self._a
            self._a = bigger
        self._a[self._n] = x
        self._n += 1

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        return self._a[i]

    @property
    def array(self) -> np.ndarray:
        return self._a[:self._n]
