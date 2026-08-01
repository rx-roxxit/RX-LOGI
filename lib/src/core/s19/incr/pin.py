"""W2 hash-pin - the frozen sources that `incr/` mirrors, by content hash.

`incr/` holds a SECOND copy of this market logic. That copy is only known to be
correct against the exact bytes listed here (proven by W1 in
tests/test_s19_incr_parity.py). If a frozen source changes, the mirror is no
longer known to match: this wall goes red and the mirror must be re-verified
BEFORE the hash is updated.
"""
from __future__ import annotations

from ..frozen import FROZEN, _sha

MIRRORED = {
    "swing.py": "de7da88094657b232b5f35d6d551acd6710e79f9fa7fa3e1122d8e57b0b6c38b",
    "erl_irl.py": "c1e9233f043db00275e40d1c134a37621a9d1fc5b0e30cea64038b772f8a3f31",
    "trend.py": "d1f0c86854126ac04843514d08b633b27301a58e03b1762faa22dd76ea030608",
    "sweep.py": "705dcc5586c352df1a7590bec2aeb1da5855c0521e0f7cb0da09e391dc1997aa",
    "eqhl.py": "b2e2efaa3edbeca9e295a68c6e68a90ee91e3cd1eba2c9a81e94d17881870312",
    "fvg.py": "59c4c4980cd6796df54387b6f403ebbe0eaf7c470c23de4321e0b84d64bf7c51",
    "ob.py": "2cfa02c2af688d8a0a133343b9c656bdceb818a88b4ced98082a583152460ac2",
    "liquidity.py": "691aa34a361dd0b3aafe0291ad137bf0be052f8fee803b34c13285c2136d968a",
    "fib.py": "303d42bc3a2e26772fb8238a4a55292c5b57f4afb1ce2db4e43a13b651b1f935",
}


def verify_mirrors() -> None:
    for name, want in MIRRORED.items():
        got = _sha(FROZEN / name)
        if got != want:
            raise RuntimeError(
                "FEATURES_VERIFIED/%s changed since incr/ mirrored it. "
                "Re-run the W1 parity wall against the new bytes, then update "
                "MIRRORED in src/core/s19/incr/pin.py." % name)
