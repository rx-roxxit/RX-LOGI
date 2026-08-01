"""Incremental mirrors of the frozen feature modules.

Each class carries the state its frozen counterpart rebuilds from scratch on
every call, so a live engine can append one bar instead of rescanning the tape.
Correctness is not argued - it is walled: tests/test_s19_incr_parity.py demands
element-wise equality with the frozen module (W1) and pin.py fails the moment a
mirrored source changes (W2).

Every mirror takes (dt, o, h, l, c) as INDEXABLE, APPEND-ABLE sequences, exposes
feed(i) for one closed bar at a time, and result() for the frozen compute()
shape. Mirrors that need another feature build their own by default; pass the
dependency in to share one - and then feed the dependency FIRST.
"""
from __future__ import annotations

from .eqhl import IncrEqhl
from .erl_irl import IncrErlIrl
from .fib import IncrFib
from .fvg import IncrFvg
from .liquidity import IncrLiquidity
from .ob import IncrOb
from .pin import MIRRORED, verify_mirrors
from .sweep import IncrSweep
from .swing import IncrSwing
from .trend import IncrTrend

MIRRORS = {
    "swing": IncrSwing,
    "erl_irl": IncrErlIrl,
    "trend": IncrTrend,
    "sweep": IncrSweep,
    "eqhl": IncrEqhl,
    "fvg": IncrFvg,
    "ob": IncrOb,
    "liquidity": IncrLiquidity,
    "fib": IncrFib,
}

__all__ = ["IncrSwing", "IncrErlIrl", "IncrTrend", "IncrSweep", "IncrEqhl",
           "IncrFvg", "IncrOb", "IncrLiquidity", "IncrFib",
           "MIRRORS", "MIRRORED", "verify_mirrors"]
