"""Gateway to the USER-verified feature modules — S19 may ONLY see the market
through these. Importing this module re-checks the freeze: if any live feature
file drifted from FEATURES_VERIFIED/MANIFEST.sha256 the import fails loudly, so
no S19 number can ever be produced from unverified features.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / "release" / "TheRXDesk" / "platform" / "backend"
FROZEN = ROOT / "FEATURES_VERIFIED"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r", b"")).hexdigest()


def verify_freeze() -> None:
    man = {}
    for line in (FROZEN / "MANIFEST.sha256").read_text().splitlines():
        if line.strip():
            h, name = line.split()
            man[name] = h
    for name, h in man.items():
        if _sha(FROZEN / name) != h:
            raise RuntimeError(f"FEATURES_VERIFIED/{name} was modified — frozen archive is untouchable")
        if _sha(LIVE / "app" / "features" / name) != h:
            raise RuntimeError(
                f"app/features/{name} diverged from the USER-verified freeze. "
                f"S19 refuses to run on unverified features.")


verify_freeze()
sys.path.insert(0, str(LIVE))

from app.features import eqhl, erl_irl, fib, fvg, liquidity, ob, sweep, swing, trend  # noqa: E402,F401
from app.features.erl_irl import events as erl_events  # noqa: E402
from app.features.swing import swing_events  # noqa: E402

__all__ = ["erl_events", "swing_events", "eqhl", "erl_irl", "fib", "fvg",
           "liquidity", "ob", "sweep", "swing", "trend", "verify_freeze"]
