#!/usr/bin/env python3
"""Static parser for WearFit/Chronos 'dial.bin' (fbiego bin2lvgl layout). Reads only, never executes."""
import struct, sys, os, json
from PIL import Image
import numpy as np

TYPES = {0x00:"Hour",0x01:"Minute",0x02:"Date(day)",0x03:"Month",0x06:"Weekday",0x07:"Year",0x08:"AM/PM",
 0x09:"Image/Icon",0x0A:"Connection(BT)",0x0B:"Battery",0x0C:"Sleep label",0x0D:"Analog hands",0x0E:"Steps",
 0x0F:"Calories",0x10:"HeartRate",0x11:"SpO2",0x13:"Sleep no",0x14:"Distance",0x15:"Distance label",
 0x16:"Weather temp",0x17:"Weather icon",0x19:"Solid color",0x1A:"Analog x5",0x1B:"Seconds",0x1D:"Unknown(1D)",
 0x1E:"Analog x8",0xFA:"Weather no+label",0xFB:"Click",0xFD:"Animation"}

def rgb565le(buf):
    v = np.frombuffer(buf, dtype='<u2').astype(np.uint32)
    r = (v >> 11) & 0x1F; g = (v >> 5) & 0x3F; b = v & 0x1F
    return np.stack([(r<<3)|(r>>2), (g<<2)|(g>>4), (b<<3)|(b>>2)], -1).astype(np.uint8)

def parse(path):
    d = open(path,'rb').read()
    n = struct.unpack_from('<I', d, 0)[0]
    els = []
    for i in range(n):
        o = 4 + 20*i
        t, f1, a, f3, x, y, w, h, clt, dat = struct.unpack_from('<BBBBHHHHII', d, o)
        grouped = bool(f1 & 0x80) or t == 0x08
        frames = (f1 & 0x7F) if grouped else 1
        els.append(dict(idx=i, off=o, type=t, name=TYPES.get(t, f"?0x{t:02x}"), b1=f1, frames=frames,
                        b2=a, b3=f3, x=x, y=y, w=w, h=h, clt=clt, dat=dat,
                        mode=("solid" if t==0x19 else "raw565" if clt==dat else "idx8"),
                        pal_entries=((dat-clt)//2 if dat>clt else 0)))
    return d, els

def decode(d, e):
    w, h = e['w'], e['h']
    if w == 0 or h == 0: return None
    if e['mode'] == 'solid':
        return None
    if e['mode'] == 'raw565':
        buf = d[e['dat']: e['dat'] + w*h*2]
        if len(buf) < w*h*2: return None
        return rgb565le(buf).reshape(h, w, 3)
    pal = rgb565le(d[e['clt']: e['dat']])
    idx = np.frombuffer(d[e['dat']: e['dat'] + w*h], dtype=np.uint8)
    if len(idx) < w*h or idx.max() >= len(pal):
        # palette may be shorter than referenced indexes -> flag it
        pal = rgb565le(d[e['clt']: e['clt'] + 512])
        if len(idx) < w*h: return None
    return pal[idx].reshape(h, w, 3)

if __name__ == '__main__':
    path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    d, els = parse(path)
    print(f"file size {len(d)}  components {len(els)}  table end 0x{4+20*len(els):x}")
    print("idx type name              b1  fr b2  b3    x    y    w    h     clt      dat  mode  pal")
    for e in els:
        print(f"{e['idx']:3} {e['type']:02x}  {e['name']:17} {e['b1']:02x} {e['frames']:3} {e['b2']:02x} {e['b3']:02x} {e['x']:4} {e['y']:4} {e['w']:4} {e['h']:4} {e['clt']:#8x} {e['dat']:#8x} {e['mode']:6} {e['pal_entries']}")
    json.dump(els, open(os.path.join(outdir,'components.json'),'w'), indent=1)
    seen = {}
    for e in els:
        key = (e['clt'], e['dat'], e['w'], e['h'])
        if key in seen: continue
        img = decode(d, e)
        if img is None: continue
        seen[key] = e['idx']
        Image.fromarray(img).save(os.path.join(outdir, f"res_{e['idx']:02d}_{e['type']:02x}_{e['w']}x{e['h']}.png"))
