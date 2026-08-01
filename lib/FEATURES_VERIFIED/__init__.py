"""RX Feature Foundry — ICT features written ONCE in Python, drawn on the desk,
eye-verified by the USER, then reused everywhere (no porting, no drift).

Source of truth for every definition: `mq5/RX_*.mq5` (USER TIER-1 spec).
Every feature is causal (candle-closure): each object carries
  at         — bar where the object lives (swing bar / OB bar / gap bar)
  confirm_at — bar whose CLOSE made it knowable; replay may show it only
               when that bar is on screen (confirm_at <= shown bar time)
"""
from __future__ import annotations

from . import eqhl, erl_irl, fib, fvg, liquidity, ob, sweep, swing, trend

# name -> compute(dt, o, h, l, c) -> dict of draw-lists
FEATURES = {
    "swing": swing.compute,
    "erl_irl": erl_irl.compute,
    "trend": trend.compute,
    "fvg": fvg.compute,
    "ob": ob.compute,
    "liquidity": liquidity.compute,
    "sweep": sweep.compute,
    "fib": fib.compute,
    "eqhl": eqhl.compute,
}
