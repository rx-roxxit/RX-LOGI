"""RX Logi V2.0 stage-1 model + metrics (PREREG §4B.3, §6A).

Transformer over the multi-timeframe event sequence, MLP over the context vector,
one 3-class softmax head. Sizes are fixed per cell in the pre-registration and are
NOT tuned after seeing results. No class weighting: calibrated probability is the
claim being tested, and reweighting would distort exactly that.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .tokens import N_TOKEN, SEQ_CAP

ARCH = {"15m": dict(d_model=64, layers=3, heads=4),
        "30m": dict(d_model=64, layers=2, heads=4),
        "1H": dict(d_model=32, layers=2, heads=2)}
EPS = 1e-12


class ChainNet(nn.Module):
    def __init__(self, n_ctx: int, d_model=64, layers=2, heads=4, dropout=0.15,
                 n_out: int = 3):
        super().__init__()
        self.tok = nn.Linear(N_TOKEN, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_CAP, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d_model), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Dropout(dropout),
                                  nn.Linear(2 * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, n_out))

    def forward(self, seq, seq_len, ctx):
        pad = torch.arange(SEQ_CAP, device=seq.device)[None, :] < (SEQ_CAP - seq_len[:, None])
        z = self.enc(self.tok(seq) + self.pos, src_key_padding_mask=pad)
        z = z.masked_fill(pad[:, :, None], 0.0).sum(1) / seq_len.clamp(min=1)[:, None]
        return self.head(torch.cat([z, self.ctx(ctx)], dim=1))


# ── metrics (PREREG §6A: log-loss primary, both decision levels, calibration) ──
def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(np.asarray(p, float))) + 1.0
    npos = int(y.sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the predicted-class confidence."""
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    p = p / p.sum(1, keepdims=True)
    rev = y != 0
    m = {"n": int(len(y)),
         "logloss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
         "acc": float((p.argmax(1) == y).mean()),
         "auc_lvl1": auc(rev, 1.0 - p[:, 0]),
         "ece": ece(y, p)}
    if rev.sum() > 1:
        denom = p[rev][:, 1] + p[rev][:, 2]
        m["auc_lvl2"] = auc(y[rev] == 2, p[rev][:, 2] / np.clip(denom, EPS, None))
        m["n_lvl2"] = int(rev.sum())
    else:
        m["auc_lvl2"], m["n_lvl2"] = float("nan"), int(rev.sum())
    return m


def boot_ci(y: np.ndarray, p: np.ndarray, key: str, n: int = 400, seed: int = 0) -> tuple:
    """Bootstrap CI — the 1H cell is thin enough that a point estimate alone
    would be misleading (PREREG §6A.1)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = metrics(y[i], p[i]).get(key, float("nan"))
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ── binary target (PREREG 4D) ────────────────────────────────────────────────
CONF_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)


def coverage_table(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """The product-facing view: at each confidence floor, how many plans do we
    still act on and how often are we right. This is the number that decides
    whether the model is useful, not AUC."""
    conf, pred = p.max(1), p.argmax(1)
    out = []
    for lo in CONF_LEVELS:
        m = conf >= lo
        out.append({"min_conf": lo, "coverage": float(m.mean()), "n": int(m.sum()),
                    "acc": float((pred[m] == y[m]).mean()) if m.sum() else float("nan")})
    return out


def metrics_bin(y: np.ndarray, p: np.ndarray) -> dict:
    """Binary: 0 = continue, 1 = reverse."""
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    p = p / p.sum(1, keepdims=True)
    y = np.asarray(y, int)
    return {"n": int(len(y)),
            "logloss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
            "acc": float((p.argmax(1) == y).mean()),
            "auc": auc(y == 1, p[:, 1]),
            "base": float(max(y.mean(), 1 - y.mean())),
            "ece": ece(y, p),
            "coverage": coverage_table(y, p)}
