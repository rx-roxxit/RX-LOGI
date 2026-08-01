"""Fair Value Gap — BISI/SIBI (mq5/RX_05_FVG.mq5 geometry)
+ USER 2026-07-20: a gap is USED ONCE, like the OB — first return touch spends it.

3-bar logic, confirmed at the close of bar 3 (causal):
  Bullish (BISI): high[i-2] < low[i]  -> zone [high[i-2] .. low[i]]
  Bearish (SIBI): low[i-2]  > high[i] -> zone [high[i] .. low[i-2]]
Mitigated on the FIRST TOUCH of the NEAR edge, counted from the bar AFTER the
confirm bar (the confirm bar's own wick IS the edge — it must not self-trigger):
  bull: a later bar's low  <= zone top (low of candle 3)
  bear: a later bar's high >= zone bottom (high of candle 3)
-> box cut at the touch bar + dimmed. (The ref's close-through rule is replaced;
active boxes = virgin gaps price has never returned to.)

Display: green box = bullish, red box = bearish, label "FVG" mid-right.
Box starts at the middle candle (i-1), as the ref draws it.
"""
from __future__ import annotations


def compute(dt, o, h, l, c) -> dict:
    boxes: list[dict] = []
    cbars: list[int] = []
    n = len(c)
    for i in range(2, n):
        bull = h[i - 2] < l[i]
        bear = l[i - 2] > h[i]
        if bull or bear:
            lo = float(h[i - 2]) if bull else float(h[i])
            hi = float(l[i]) if bull else float(l[i - 2])
            boxes.append({"at": dt[i - 1], "confirm_at": dt[i],
                          "side": "bull" if bull else "bear",
                          "lo": lo, "hi": hi, "mit_at": None})
            cbars.append(i)

    for box, cb in zip(boxes, cbars):
        for j in range(cb + 1, n):
            touched = (l[j] <= box["hi"]) if box["side"] == "bull" else (h[j] >= box["lo"])
            if touched:
                box["mit_at"] = dt[j]
                break
    return {"boxes": boxes}
