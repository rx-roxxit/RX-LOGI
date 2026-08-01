"""Order Block — USER 4-bar spec (2026-07-20, FVG position amended 2026-07-21).

Bullish OB, four consecutive bars:
  [OB]   red candle (the zone: its low..high)
  [OB+1] green, closing ABOVE the OB high
  [OB+2] green
  [OB+3] green — the third push bar (แรงดันอย่างน้อย 3 แท่ง)
  + a bullish FVG somewhere in the displacement — either straddling OB+1
    (low[OB+2] > high[OB]) OR straddling OB+2 (low[OB+3] > high[OB+1]).
Bearish OB mirrors every condition.

★ 2026-07-21 (USER eye-check, 31-Oct-2024): the FVG need not sit immediately on
the OB candle; it may be one bar into the displacement. Earlier code required the
first position only and missed valid OBs (e.g. when OB+2's wick pokes back into
the OB zone but the gap forms at OB+2->OB+3 instead). Accepting EITHER position
only ADDS blocks the old rule dropped; it never removes an old one.

`gap` = the width of the FVG that qualified it (the wider of the two if both are
present) — USER: the wider, the higher quality (knowledge for the model/eye;
NOT used as a filter).

Confirmed at the close of OB+3 (replay may show the box only from there).
Mitigated on first RETURN TOUCH after the confirm bar (bull: low <= zone top;
bear: high >= zone bottom) -> box cut + dimmed.
"""
from __future__ import annotations


def compute(dt, o, h, l, c) -> dict:
    boxes: list[dict] = []
    cbars: list[int] = []
    n = len(c)
    for b in range(0, n - 3):
        # bullish FVG at either straddle position (OB+1 or OB+2 as the middle)
        fvg_a = l[b + 2] - h[b]                            # straddles OB+1
        fvg_b = l[b + 3] - h[b + 1]                        # straddles OB+2
        if (c[b] < o[b]                                   # red OB
                and c[b + 1] > o[b + 1] and c[b + 1] > h[b]
                and c[b + 2] > o[b + 2]
                and c[b + 3] > o[b + 3]
                and (fvg_a > 0 or fvg_b > 0)):
            boxes.append({"at": dt[b], "confirm_at": dt[b + 3], "side": "bull",
                          "lo": float(l[b]), "hi": float(h[b]),
                          "gap": round(float(max(fvg_a, fvg_b)), 6), "mit_at": None})
            cbars.append(b + 3)
            continue
        # bearish FVG at either straddle position
        gvf_a = l[b] - h[b + 2]                            # straddles OB+1
        gvf_b = l[b + 1] - h[b + 3]                        # straddles OB+2
        if (c[b] > o[b]                                   # green OB (bear)
                and c[b + 1] < o[b + 1] and c[b + 1] < l[b]
                and c[b + 2] < o[b + 2]
                and c[b + 3] < o[b + 3]
                and (gvf_a > 0 or gvf_b > 0)):
            boxes.append({"at": dt[b], "confirm_at": dt[b + 3], "side": "bear",
                          "lo": float(l[b]), "hi": float(h[b]),
                          "gap": round(float(max(gvf_a, gvf_b)), 6), "mit_at": None})
            cbars.append(b + 3)

    for box, cb in zip(boxes, cbars):
        for i in range(cb + 1, n):
            hit = (l[i] <= box["hi"]) if box["side"] == "bull" else (h[i] >= box["lo"])
            if hit:
                box["mit_at"] = dt[i]
                break
    return {"boxes": boxes}
