#!/usr/bin/env python3
# Composite preview of the dial with sample values (static analysis only).
import sys, numpy as np
from PIL import Image
sys.path.insert(0,'.')
from parse_dial import parse, decode
d, els = parse(sys.argv[1])
vals = {0x00:10, 0x01:8, 0x1B:36, 0x02:7, 0x03:10, 0x0B:85, 0x10:72, 0x14:35, 0x0E:2735, 0x0F:163, 0x13:730}
LANG = 2  # 1 = first variant (Chinese), 2 = second (English)
bg = decode(d, els[0])
nfr = els[0]['frames']; fh = els[0]['h']//nfr
frames=[]
for f in range(nfr):
    canvas = Image.fromarray(bg[f*fh:(f+1)*fh]).convert('RGBA')
    place = {}; lan = 0
    for e in els[1:]:
        img = decode(d, e)
        if img is None: continue
        fr = e['frames']; h1 = e['h']//fr
        if e['b3'] & 0x80:
            lan += 1
            cg = e['b3'] & 0x7F
            use = (lan == LANG)
            if lan == cg: lan = 0
            if not use: continue
        t = e['type']
        if t in vals and fr > 1:
            if t == 0x0B and e['b2'] == 0:      # battery level bar
                k = min(vals[t] // (100 // fr), fr-1)
            else:
                p = place.get(t, 0); place[t] = p+1
                k = (vals[t] // 10**p) % 10
                if k >= fr: k = 0
        else:
            k = 0
        sub = img[k*h1:(k+1)*h1]
        a = np.where(sub.reshape(-1,3).sum(1)==0, 0, 255).astype(np.uint8).reshape(h1, e['w'],1)
        canvas.alpha_composite(Image.fromarray(np.concatenate([sub,a],2),'RGBA'), (e['x'], e['y']))
    frames.append(canvas.convert('RGB'))
frames[0].save('preview_full_dial.png')
q=[f.convert('P', palette=Image.ADAPTIVE) for f in frames]
q[0].save('preview_full_dial_anim.gif', save_all=True, append_images=q[1:], duration=200, loop=0)
print('ok', nfr)
