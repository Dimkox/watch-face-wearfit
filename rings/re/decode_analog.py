#!/usr/bin/env python3
"""Decode type 0x1A ('Analog x5') hands from WearFit/Chronos dial.bin (static analysis)."""
import struct, sys, os
import numpy as np
from PIL import Image
sys.path.insert(0, '/workspace/watchface/out')
from parse_dial import parse

def decode_1a(d, e, nframes=None):
    w, h = e['w'], e['h']
    clt, dat = e['clt'], e['dat']
    frames = []
    k = 0
    # number of frames: until table runs out; caller can pass
    nframes = nframes or 16
    for f in range(nframes):
        img = np.zeros((h, w, 4), np.uint8)
        for r in range(h):
            hi, lo, x0, cnt = struct.unpack_from('<HHHH', d, dat + 8 * (f * h + r))
            off = (hi << 16) | lo
            for i in range(cnt):
                a, c0, c1 = d[clt + off + 3*i: clt + off + 3*i + 3]
                v = c0 | (c1 << 8)
                R = (v >> 11) & 31; G = (v >> 5) & 63; B = v & 31
                img[r, x0 + i] = ((R << 3) | (R >> 2), (G << 2) | (G >> 4), (B << 3) | (B >> 2), a)
        frames.append(img)
    return frames

if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d, els = parse(path)
    for e in els:
        if e['type'] != 0x1a: continue
        fr = decode_1a(d, e)
        sheet = Image.new('RGBA', (e['w'] * 4, e['h'] * 4), (40, 40, 40, 255))
        for i, f in enumerate(fr):
            im = Image.fromarray(f, 'RGBA')
            sheet.alpha_composite(im, ((i % 4) * e['w'], (i // 4) * e['h']))
        sheet.save(os.path.join(out, f"hand_{e['idx']}_b3_{e['b3']}.png"))
        print(e['idx'], e['b1'], e['b2'], e['b3'], e['w'], e['h'])
