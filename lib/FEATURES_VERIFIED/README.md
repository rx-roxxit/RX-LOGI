# FEATURES_VERIFIED — คลัง feature ที่ USER ตรวจด้วยตาแล้ว (FROZEN)

> **⛔ ห้ามแก้ไขไฟล์ในโฟลเดอร์นี้เด็ดขาด — คำสั่ง USER 2026-07-20**
> นี่คือสำเนาอ้างอิงของ 10 feature modules ณ วันที่ผ่านการตรวจด้วยตาบน therxdesk
> (commit 62b8fd8 · Foundry batch 1+2 + FVG used-once)

## กติกา

1. **ไฟล์ที่นี่ = ความจริงอ้างอิง** — ห้ามแตะทุกกรณี (read-only + hash ล็อกใน `MANIFEST.sha256`)
2. **ตัว live** (`release/TheRXDesk/platform/backend/app/features/`) ต้อง**เหมือนที่นี่ทุก byte**
   — จะแก้ตัว live ได้ก็ต่อเมื่อ USER สั่งแก้/ตรวจใหม่ แล้ว re-freeze ที่นี่พร้อมกันเท่านั้น
3. กำแพงที่แดงได้: `platform/backend/tests/test_features_frozen.py`
   - แดงถ้าไฟล์ในนี้ถูกแก้ (hash ไม่ตรง manifest)
   - แดงถ้าตัว live ถูกแก้จนไม่ตรงกับที่ freeze ไว้ (drift = ต้องให้ USER ตรวจใหม่ก่อน)

## ของข้างใน (nิยามย่อ)

| ไฟล์ | feature | นิยามหลัก |
|---|---|---|
| `swing.py` | Swing | **SwingSM รากเดียวของทุกตัว** — candle-closure (RX_01) + คลาส HH/HL/LH/LL |
| `erl_irl.py` | ERL/IRL | ขอบ range vs ภายใน + BOS promotion / CHoCH (RX_02) — promote แบบ retroactive-honest |
| `trend.py` | Trend | EXT/INT แยกชั้น BULL/BEAR/WAITING + เส้น BOS/MSS ที่ระดับที่ถูกทะลุ (RX_03 + USER) |
| `fvg.py` | FVG | 3-bar gap (RX_05) + **used-once**: แตะขอบใกล้ครั้งแรกหลังแท่งยืนยัน = ใช้แล้ว |
| `ob.py` | OB | **USER 4-bar pattern**: แดง → เขียวปิด>high → เขียว FVG คร่อม OB → เขียว · gap = คุณภาพ |
| `liquidity.py` | Liquidity | **USER naming**: BSL ใต้ swing low (เขียว) · SSL เหนือ swing high (แดง) · sweep ด้วยไส้ |
| `sweep.py` | Sweep | stop hunt: ทะลุไส้+ปิดกลับ / delayed reclaim ≤3 แท่ง / เบรกจริง = ถอน (RX_09) |
| `fib.py` | Fib | ขา ERL expansion (จบ HH/LL เท่านั้น) 1=ซ้าย→0=ขวา · เก็บทุกขาให้เทรน |
| `eqhl.py` | EQHL | swing เรา + LuxAlgo `0.1×ATR(200)` — **ATR = indicator เดียวที่ USER อนุญาต** |
| `__init__.py` | registry | ชื่อ → compute |

ทุก object มี `at` (จุดเกิด) + `confirm_at` (แท่งยืนยัน) + lifecycle — causal, prefix-invariant
(กำแพง: `tests/test_features.py`)
