#!/usr/bin/env python3
"""Layout mockup only — does NOT write x5pro_rings.bin.
Shows three concentric rings + markers, and places
steps / HR / calories / date / time in the annulus between
the second (outer) and minute (middle) rings."""
import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = os.path.dirname(os.path.abspath(__file__))
W = H = 360
CX = CY = 180
S = 3  # supersample

R_HOUR, R_MIN, R_SEC = 84, 121, 158
C_HOUR = (255, 190, 60)
C_MIN = (60, 220, 255)
C_SEC = (255, 70, 60)

FONT_BIG = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Bold.ttf"
FONT_MED = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-SemiBold.ttf"
FONT_SM = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Medium.ttf"

def dim(c, k, a=255):
    return tuple(int(v * k) for v in c) + (a,)

def make_bg():
    yy, xx = np.mgrid[0:H * S, 0:W * S] / S
    r = np.hypot(xx - CX + 0.5, yy - CY + 0.5)
    t = np.clip(r / 190.0, 0, 1)
    bg = np.stack([10 + 8 * (1 - t), 14 + 14 * (1 - t), 26 + 26 * (1 - t)], -1)
    img = Image.fromarray(bg.astype(np.uint8), "RGB").convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")

    def circ(rad, col, width):
        rad = rad + width / 2
        d.ellipse([(CX - rad) * S, (CY - rad) * S, (CX + rad) * S, (CY + rad) * S],
                  outline=col, width=int(width * S))

    def tick(rad1, rad2, ang, col, width):
        a = math.radians(ang)
        x1, y1 = CX + rad1 * math.sin(a), CY - rad1 * math.cos(a)
        x2, y2 = CX + rad2 * math.sin(a), CY - rad2 * math.cos(a)
        d.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=col, width=int(width * S))

    # faint fill of the "widget band" between sec & min rings (layout zone)
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bd.ellipse([(CX - R_SEC) * S, (CY - R_SEC) * S, (CX + R_SEC) * S, (CY + R_SEC) * S],
               fill=(40, 70, 100, 40))
    bd.ellipse([(CX - R_MIN) * S, (CY - R_MIN) * S, (CX + R_MIN) * S, (CY + R_MIN) * S],
               fill=(0, 0, 0, 0))
    img.alpha_composite(band)

    circ(R_SEC, dim(C_SEC, 0.4), 3)
    for i in range(60):
        L = 9 if i % 5 == 0 else 4
        tick(R_SEC + 6, R_SEC + 6 + L, i * 6,
             (200, 205, 215, 255) if i % 5 == 0 else (110, 115, 130, 255),
             2 if i % 5 == 0 else 1)
    circ(R_MIN, dim(C_MIN, 0.4), 3)
    for i in range(12):
        tick(R_MIN - 12, R_MIN - 6, i * 30, dim(C_MIN, 0.55), 2)
    circ(R_HOUR, dim(C_HOUR, 0.4), 3)
    for i in range(12):
        a = math.radians(i * 30)
        x, y = CX + (R_HOUR + 9) * math.sin(a), CY - (R_HOUR + 9) * math.cos(a)
        d.ellipse([(x - 1.6) * S, (y - 1.6) * S, (x + 1.6) * S, (y + 1.6) * S],
                  fill=dim(C_HOUR, 0.7))

    # 12 o'clock index
    tri = [(CX - 7, 4), (CX + 7, 4), (CX, 15)]
    d.polygon([(x * S, y * S) for x, y in tri], fill=(235, 240, 250, 255))
    for rad in (R_SEC, R_MIN, R_HOUR):
        d.line([CX * S, (CY - rad - 7) * S, CX * S, (CY - rad + 7) * S],
               fill=(235, 240, 250, 200), width=2 * S)

    return img.resize((W, H), Image.LANCZOS)

def marker(R, half_deg, thick, color, angle_deg):
    """RGBA layer 360x360 with one glowing arc marker at angle_deg (0=12, clockwise)."""
    layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ld = ImageDraw.Draw(layer)
    px, py = CX * S, CY * S

    def bb(rad):
        return [px - rad * S, py - rad * S, px + rad * S, py + rad * S]

    a0 = angle_deg - 90 - half_deg
    a1 = angle_deg - 90 + half_deg
    gd.arc(bb(R + (thick + 8) / 2), a0 - 2, a1 + 2, fill=color + (150,), width=int((thick + 8) * S))
    glow = glow.filter(ImageFilter.GaussianBlur(3 * S))
    layer.alpha_composite(glow)
    ld.arc(bb(R + thick / 2), a0, a1, fill=color + (255,), width=int(thick * S))
    for a in (a0, a1):
        ar = math.radians(a)
        ex, ey = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
        rr = thick * S / 2
        ld.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], fill=color + (255,))
    ar = math.radians(angle_deg - 90)
    bx, by = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
    rb = (thick * 0.42) * S
    ld.ellipse([bx - rb, by - rb, bx + rb, by + rb], fill=(255, 255, 255, 255))
    return layer.resize((W, H), Image.LANCZOS)

def text_centered(draw, xy, text, font, fill, stroke_fill=(0, 0, 0), stroke=1):
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = xy
    draw.text((x - tw / 2 - bb[0], y - th / 2 - bb[1]), text, font=font,
              fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)

def polar(ang_deg, rad):
    a = math.radians(ang_deg)
    return CX + rad * math.sin(a), CY - rad * math.cos(a)

def draw_widgets(can, hh=10, mm=8, ss=36, steps=2735, hr=72, kcal=163, mon=10, day=7):
    """Place widgets in the annulus between R_SEC and R_MIN (mid ~139.5).
    Angular slots chosen so they don't overlap the 12-index or each other."""
    d = ImageDraw.Draw(can, "RGBA")
    f_time = ImageFont.truetype(FONT_BIG, 28)
    f_val = ImageFont.truetype(FONT_MED, 16)
    f_lab = ImageFont.truetype(FONT_SM, 11)
    f_tiny = ImageFont.truetype(FONT_SM, 10)

    # Mid-radius of the widget band
    R_BAND = (R_SEC + R_MIN) / 2  # ~139.5

    # --- TIME at top of the band (just below 12 index) ---
    # HH:MM sits straddling the band at ~0° but slightly below the triangle
    tx, ty = polar(0, R_BAND - 2)
    # Actually place time digits horizontally across the top arc zone
    # Use a flat placement: time centered at y corresponding to upper band
    time_y = CY - R_BAND
    text_centered(d, (CX, time_y), f"{hh:02d}:{mm:02d}", f_time, (235, 242, 255, 255), stroke=2)

    # --- DATE just under time, still in band ---
    date_y = time_y + 18
    text_centered(d, (CX, date_y), f"{mon:02d}-{day:02d}", f_lab, (170, 215, 235, 255), stroke=1)

    # Helper: label above value, centered on a polar point
    def widget(ang, label, value, accent):
        lx, ly = polar(ang, R_BAND)
        text_centered(d, (lx, ly - 8), label, f_tiny, accent + (220,), stroke=1)
        text_centered(d, (lx, ly + 6), value, f_val, (235, 242, 255, 255), stroke=1)
        # small accent tick toward center
        ix, iy = polar(ang, R_MIN + 4)
        ox, oy = polar(ang, R_SEC - 4)
        d.line([ix, iy, ox, oy], fill=accent + (80,), width=1)

    # Three activity widgets in the lower/side band (avoid 12 and avoid marker clutter)
    # Angles clockwise from 12: HR ~ 4 o'clock (120°), Steps ~ 7 o'clock (210°), Cal ~ 9:30 (285°)
    # Spread: right, bottom-left, left — leaving top for time/date
    widget(95, "HR", f"{hr:03d}", C_SEC)            # ~3:10, right (away from min@48°)
    widget(175, "STEPS", f"{steps}", C_MIN)         # ~5:50, bottom (away from sec@216°)
    widget(265, "KCAL", f"{kcal:04d}", C_HOUR)      # ~8:50, left (away from hour@304°)

    # Tiny legend in dead center (inside hour ring) — optional labels for rings
    f_leg = ImageFont.truetype(FONT_SM, 9)
    text_centered(d, (CX, CY - 14), "H", f_leg, dim(C_HOUR, 0.85)[:3] + (180,), stroke=0)
    text_centered(d, (CX, CY), "M", f_leg, dim(C_MIN, 0.85)[:3] + (180,), stroke=0)
    text_centered(d, (CX, CY + 14), "S", f_leg, dim(C_SEC, 0.85)[:3] + (180,), stroke=0)

    # Caption strip at very bottom (outside usable area note) — keep inside circle soft
    # Zone label (mockup only): faint "widget band" callout
    f_note = ImageFont.truetype(FONT_SM, 9)
    text_centered(d, (CX, CY + R_SEC - 18), "band: between sec↔min rings", f_note,
                  (140, 160, 180, 160), stroke=0)

def render(hh, mm, ss):
    can = make_bg()
    # markers: hour / min / sec angles
    h_ang = ((hh % 12) * 30 + mm * 0.5)
    m_ang = mm * 6 + ss * 0.1
    s_ang = ss * 6
    can.alpha_composite(marker(R_HOUR, 14, 8, C_HOUR, h_ang))
    can.alpha_composite(marker(R_MIN, 10, 7, C_MIN, m_ang))
    can.alpha_composite(marker(R_SEC, 7, 6, C_SEC, s_ang))
    draw_widgets(can, hh=hh, mm=mm, ss=ss)
    return can.convert("RGB")

def main():
    still = render(10, 8, 36)
    still.save(os.path.join(OUT, "layout_preview.png"))
    print("wrote layout_preview.png", still.size)

    # short GIF: seconds marker moves, widgets stay (sample time digits update every second for seconds? time is HH:MM only)
    frames = []
    for s in range(0, 60, 2):
        frames.append(render(10, 8, s).quantize(colors=200, method=Image.Quantize.MEDIANCUT))
    # then show minute advancing a bit
    for m in range(9, 21):
        frames.append(render(10, m, (m * 7) % 60).quantize(colors=200, method=Image.Quantize.MEDIANCUT))
    frames[0].save(os.path.join(OUT, "layout_preview.gif"), save_all=True,
                   append_images=frames[1:], duration=120, loop=0)
    print("wrote layout_preview.gif", len(frames), "frames")

if __name__ == "__main__":
    main()
