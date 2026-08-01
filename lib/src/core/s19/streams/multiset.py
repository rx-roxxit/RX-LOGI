"""A multiset of (level, birth) kept sorted by level.

zone_state asks four questions of its live set on every bar: how many sit above
/ below the current price, which one is nearest on each side, and - for POIs -
which of them was created most recently. The legacy loop answers all four by
walking the whole live set; a sorted list answers the first three with bisect
and the fourth by walking only the side that was asked for.

Only comparisons touch the levels - no arithmetic - so every value handed back
is the same float that went in.

`visits` counts elements actually inspected. It is the counter gate's evidence
that this is not the legacy scan wearing a new hat.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort


class SortedMultiset:
    def __init__(self) -> None:
        self._x: list[tuple[float, int]] = []
        self.visits = 0

    def __len__(self) -> int:
        return len(self._x)

    def add(self, level: float, birth: int) -> None:
        insort(self._x, (level, birth))

    def discard(self, level: float, birth: int) -> None:
        """Remove ONE entry equal to (level, birth) - the legacy list.remove()
        drops the first match too, so duplicates behave identically."""
        i = bisect_left(self._x, (level, birth))
        if i < len(self._x) and self._x[i] == (level, birth):
            del self._x[i]

    def split(self, p: float) -> int:
        """How many entries have level <= p (legacy `below` uses <=)."""
        return bisect_right(self._x, (p, float("inf")))

    def min_level(self) -> float:
        return self._x[0][0]

    def max_level(self) -> float:
        return self._x[-1][0]

    def nearest_above(self, i: int) -> float:
        return self._x[i][0]

    def nearest_below(self, i: int) -> float:
        return self._x[i - 1][0]

    def max_birth(self, lo: int, hi: int) -> int:
        self.visits += hi - lo
        best = -1
        for j in range(lo, hi):
            b = self._x[j][1]
            if b > best:
                best = b
        return best
