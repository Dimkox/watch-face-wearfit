#!/usr/bin/env python3
"""Independent verification: re-parse x5pro_rings.bin with the analysis parser,
check byte coverage / 0x1A table consistency / flag conventions vs 149.bin (+1005 for
steps/HR/cal), then render preview.png + preview.gif FROM THE BIN (not from sources)."""
import sys, struct, os
import numpy as np
from PIL import Image
sys.path.insert(0, '/workspace/watchface/out'); sys.path.insert(0, '/workspace/watchface/rings/re')
from parse_dial import parse, decode
from decode_analog import decode_1a

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "x5pro_rings.bin")
d, els = parse(path)
print(f"size {len(d)} bytes, {len(els)} components, trailer {d[-13:]!r}")
assert d[-13:] == open('/workspace/watchface/dials/X5Pro/149.bin', 'rb').read()[-13:]

# 1. byte coverage
cover = np.zeros(len(d), np.int32)
cover[:4 + 20 * len(els)] += 1
cover[len(d) - 13:] += 1
seen = set()
for e in els:
    if (e['clt'], e['dat']) in seen: continue
    seen.add((e['clt'], e['dat']))
    if e['type'] == 0x1A:
        h = e['h']
        ents = [struct.unpack_from('<HHHH', d, e['dat'] + 8 * i) for i in range(16 * h)]
        run = 0
        for hi, lo, x0, c in ents:
            assert (hi << 16 | lo) == run and x0 + c <= e['w'], 'bad 0x1A table'
            run += 3 * c
        assert run == e['dat'] - e['clt']
        cover[e['clt']:e['dat'] + 16 * h * 8] += 1
    else:
        cover[e['clt']:e['dat'] + e['w'] * e['h']] += 1
        idx = np.frombuffer(d[e['dat']:e['dat'] + e['w'] * e['h']], np.uint8)
        assert idx.max() < e['pal_entries'], 'index outside palette'
print("byte coverage: every byte used exactly once:", bool((cover == 1).all()))

# 2. conventions vs 149.bin + 1005.bin (for steps/HR/cal not in 149)
_, ref149 = parse('/workspace/watchface/dials/X5Pro/149.bin')
_, ref1005 = parse('/workspace/watchface/dials/X5Pro/1005.bin')
refflags = {}
for e in ref149 + ref1005:
    refflags.setdefault(e['type'], set()).add((e['b1'], e['b2'] if e['type'] != 0x1A else 'P', e['b3']))
ok_all = True
for e in els:
    f = (e['b1'], e['b2'] if e['type'] != 0x1A else 'P', e['b3'])
    ok = f in refflags.get(e['type'], set())
    if not ok: ok_all = False
    print(f"  #{e['idx']:2} type {e['type']:02x} {e['name']:14} b1={e['b1']:02x} b2={e['b2']:3} b3={e['b3']} "
          f"xy=({e['x']},{e['y']}) {e['w']}x{e['h']}  same flags as ref: {ok}")
print("all component flags match known-good refs:", ok_all)

# 3. render from bin
bg = decode(d, els[0])
hands = {e['b3']: (e, decode_1a(d, e)) for e in els if e['type'] == 0x1A}

def hand_layer(e, frames, step):           # step 0..59 (6 deg)
    q, k = divmod(step % 60, 15)
    if q == 0: fr, fx, fy = frames[k], False, False
    elif q == 1: fr, fx, fy = frames[15 - k], False, True
    elif q == 2: fr, fx, fy = frames[k], True, True
    else: fr, fx, fy = frames[15 - k], True, False
    P, hb = e['b2'], e['h']
    lay = np.zeros((360 * 2, 360 * 2, 4), np.uint8)
    lay[360 - (hb - P):360 + P, 360 - P:360 - P + hb] = fr
    if fx: lay = lay[:, ::-1]; lay = np.roll(lay, 1, 1)
    if fy: lay = lay[::-1]; lay = np.roll(lay, 1, 0)
    ox, oy = 360 - e['x'], 360 - e['y']
    return Image.fromarray(np.ascontiguousarray(lay[oy:oy + 360, ox:ox + 360]), 'RGBA')

def render(hh=10, mm=8, ss=36, batt=85, mon=10, day=7, steps=2735, hr=72, kcal=163, weekday=2):
    """weekday: 0=Mon .. 6=Sun (frame index)."""
    can = Image.fromarray(bg).convert('RGBA')
    vals = {
        0x00: hh, 0x01: mm, 0x0B: batt, 0x03: mon, 0x02: day,
        0x0E: steps, 0x0F: kcal, 0x10: hr, 0x06: weekday,
    }
    place = {}
    for e in els[1:]:
        if e['type'] == 0x1A: continue
        img = decode(d, e); fr = e['frames']; h1 = e['h'] // fr
        k = 0
        if e['type'] == 0x06 and fr > 1:
            k = int(vals[0x06]) % fr
        elif e['type'] in vals and fr > 1:
            p = place.get(e['type'], 0); place[e['type']] = p + 1
            k = (vals[e['type']] // 10 ** p) % 10
        sub = img[k * h1:(k + 1) * h1]
        a = np.where(sub.reshape(-1, 3).sum(1) == 0, 0, 255).astype(np.uint8).reshape(h1, e['w'], 1)
        can.alpha_composite(Image.fromarray(np.concatenate([sub, a], 2), 'RGBA'), (e['x'], e['y']))
    steps_h = {0: ((hh % 12) * 5 + mm // 12), 1: mm, 2: ss}
    for b3 in (0, 1, 2):
        e, frames = hands[b3]
        can.alpha_composite(hand_layer(e, frames, steps_h[b3]))
    return can.convert('RGB')

render(10, 8, 36).save(os.path.join(HERE, 'preview.png'))
frames = []
for s in range(60):
    frames.append(render(10, 8, s).quantize(colors=255, method=Image.Quantize.MEDIANCUT))
for s in range(0, 60, 2):
    m = (9 + s * 2) % 60; h = 10 + (9 + s * 2) // 60
    frames.append(render(h, m, (s * 7) % 60).quantize(colors=255, method=Image.Quantize.MEDIANCUT))
frames[0].save(os.path.join(HERE, 'preview.gif'), save_all=True, append_images=frames[1:],
               duration=[250] * 60 + [120] * 30, loop=0)
print('rendered preview.png and preview.gif')
print(f'OK size={len(d)} under_980k={len(d) < 980000}')
