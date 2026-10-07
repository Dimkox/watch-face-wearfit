#!/usr/bin/env python3
"""
make_rings_dial.py - WearFit/Chronos dial.bin for X5 Pro (360x360, tag 0x0D).

Matches approved layout_preview_v2:
  black sunburst dial, noble-metal concentric rings, 0x1A markers
  (hour bronze / min silver / sec black enamel), center weekday+HH:MM+date,
  STEPS (top, arc label right), BAT left (% under), HR right, KAL bottom (arc label left).
"""
import struct, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OUT = os.path.dirname(os.path.abspath(__file__))
W = H = 360
CX = CY = 180
TAG = b"screenConfig\x0d"
FONT_BIG = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Bold.ttf"
FONT_MED = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-SemiBold.ttf"
FONT_SM = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Medium.ttf"

R_HOUR, R_MIN, R_SEC = 84, 121, 158
R_BAND = (R_SEC + R_MIN) / 2  # ~139.5 — widget annulus

C_BRONZE = (196, 152, 86)
C_SILVER = (198, 204, 212)
C_GUN = (168, 172, 180)
C_BAT = (176, 180, 160)
CHAMPAGNE = (236, 230, 218)
PEWTER = (186, 178, 160)

WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]  # Mon-first (WearFit)


# ---------------------------------------------------------------- helpers
def to565(r, g, b):
    return ((int(r) >> 3) << 11) | ((int(g) >> 2) << 5) | (int(b) >> 3)


def quantize_idx8(rgb, ncolors=255, dither=False, black_transparent=True):
    """rgb: HxWx3 uint8. Returns (palette565 list, index bytes).
    Pure-black source pixels -> palette[0] = 0x0000 (transparent on the watch).
    No other palette entry is allowed to become 0x0000."""
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    black = (flat.sum(1) == 0) if black_transparent else np.zeros(len(flat), bool)
    nb = flat[~black]
    idx = np.zeros(len(flat), np.uint8)
    pal = []
    if black.any():
        pal.append(0x0000)
    base = len(pal)
    if len(nb):
        k = ncolors - base
        im = Image.fromarray(nb.reshape(1, -1, 3), 'RGB')
        q = im.quantize(colors=k, method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE)
        qi = np.array(q).reshape(-1)
        qp = np.array(q.getpalette()[:3 * k], np.uint8).reshape(-1, 3)
        used = np.unique(qi)
        remap = np.zeros(256, np.uint8)
        for j, u in enumerate(used):
            c = to565(*qp[u])
            if c == 0x0000:
                c = 0x0020
            pal.append(c)
            remap[u] = base + j
        idx[~black] = remap[qi]
    assert len(pal) <= 256
    return pal, idx.tobytes()


def idx8_blob(pal, idx):
    return struct.pack('<%dH' % len(pal), *pal), idx


def noise2d(h, w, scale=28.0, seed=3):
    rng = np.random.RandomState(seed)
    gh, gw = max(2, h // int(scale)), max(2, w // int(scale))
    grid = rng.rand(gh + 1, gw + 1).astype(np.float32)
    ys = np.linspace(0, gh, h, endpoint=False)
    xs = np.linspace(0, gw, w, endpoint=False)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    y1 = np.clip(y0 + 1, 0, gh); x1 = np.clip(x0 + 1, 0, gw)
    n00 = grid[y0][:, x0]; n10 = grid[y1][:, x0]
    n01 = grid[y0][:, x1]; n11 = grid[y1][:, x1]
    return (n00 * (1 - fy) * (1 - fx) + n10 * fy * (1 - fx)
            + n01 * (1 - fy) * fx + n11 * fy * fx)


def polar(ang_deg, rad):
    a = math.radians(ang_deg)
    return CX + rad * math.sin(a), CY - rad * math.cos(a)


# ---------------------------------------------------------------- background (from layout_preview_v2)
def brushed_disk(radius, base_rgb, hi_rgb, seed, S=2):
    size = int(2 * radius * S + 8 * S)
    cy = cx = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx - cx) / S; dy = (yy - cy) / S
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    brush = 0.55 + 0.45 * np.sin(ang * 48 + noise2d(size, size, 10, seed) * 6)
    grain = noise2d(size, size, 4, seed + 7) * 0.18
    light = np.clip(0.55 + 0.55 * np.cos(ang - math.radians(-40)), 0, 1)
    light = light * (0.35 + 0.65 * np.exp(-((r / (radius + 1e-3) - 0.55) ** 2) / 0.35))
    shade = np.clip(brush * 0.55 + grain + light * 0.55, 0.15, 1.15)
    rgb = np.zeros((size, size, 3), np.float32)
    for i in range(3):
        rgb[..., i] = base_rgb[i] * (0.55 + 0.45 * shade) + hi_rgb[i] * np.clip(shade - 0.7, 0, 1) * 0.9
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    edge = np.clip((radius + 1.2 - r) * S, 0, 1)
    alpha = (edge * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def ring_band(radius, width, base_rgb, hi_rgb, seed, S=2):
    disk = brushed_disk(radius + width / 2 + 2, base_rgb, hi_rgb, seed, S)
    size = disk.size[0]
    cy = cx = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot((xx - cx) / S, (yy - cy) / S)
    arr = np.array(disk)
    inner = radius - width / 2
    fade = np.clip((r - (inner - 0.8)) * S, 0, 1)
    outer = np.clip(((radius + width / 2 + 0.8) - r) * S, 0, 1)
    arr[..., 3] = (arr[..., 3].astype(np.float32) * fade * outer).astype(np.uint8)
    lip = np.exp(-((r - (radius + width / 2 * 0.85)) ** 2) / 1.8) * 0.35
    lip += np.exp(-((r - (radius - width / 2 * 0.85)) ** 2) / 1.8) * 0.25
    for i in range(3):
        arr[..., i] = np.clip(arr[..., i] + lip * (hi_rgb[i] - arr[..., i]), 0, 255)
    return Image.fromarray(arr, "RGBA")


def paste_centered(canvas, layer, cx=CX, cy=CY, S=2):
    x = int(cx * S - layer.size[0] / 2)
    y = int(cy * S - layer.size[1] / 2)
    canvas.alpha_composite(layer, (x, y))


def draw_arc_label(can, text, center_ang, radius, color, clockwise=True,
                   font_size=11, tracking_deg=7.2):
    """Rotated letter sprites along an arc (screen-res canvas)."""
    font = ImageFont.truetype(FONT_SM, font_size)
    n = len(text)
    span = (n - 1) * tracking_deg
    start = center_ang - span / 2 if clockwise else center_ang + span / 2
    step = tracking_deg if clockwise else -tracking_deg
    for i, ch in enumerate(text):
        ang = start + i * step
        bb = font.getbbox(ch)
        pad = 4
        gw = bb[2] - bb[0] + pad * 2
        gh = bb[3] - bb[1] + pad * 2
        glyph = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glyph)
        gd.text((pad - bb[0] + 1, pad - bb[1] + 1), ch, font=font, fill=(0, 0, 0, 110))
        gd.text((pad - bb[0], pad - bb[1]), ch, font=font, fill=color + (240,))
        rot = ang if clockwise else ang + 180
        glyph = glyph.rotate(-rot, resample=Image.BICUBIC, expand=True)
        lx, ly = polar(ang, radius)
        px = int(lx - glyph.size[0] / 2)
        py = int(ly - glyph.size[1] / 2)
        can.alpha_composite(glyph, (px, py))


def make_background():
    """Black sunburst + noble metal rings + ticks + baked STEPS/KAL/BAT/HR labels."""
    S = 2
    yy, xx = np.mgrid[0:H * S, 0:W * S]
    dx = (xx / S - CX).astype(np.float32)
    dy = (yy / S - CY).astype(np.float32)
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)

    n_ang = noise2d(H * S, W * S, 22, 5)
    g1 = np.sin(ang * 140.0 + n_ang * 1.8)
    g2 = np.sin(ang * 48.0 + noise2d(H * S, W * S, 30, 9) * 1.2)
    ridges = np.clip(g1, 0, 1) ** 2.2
    valleys = np.clip(-g1, 0, 1) ** 1.5
    brush2 = 0.5 + 0.5 * g2
    micro = noise2d(H * S, W * S, 4.0, 11) * 0.08

    light_dir = math.radians(-48)
    c1 = np.cos(ang - light_dir)
    lobe1 = np.clip(c1, 0, 1) ** 4.2
    c2 = np.cos(ang - light_dir - math.pi)
    lobe2 = np.clip(c2, 0, 1) ** 3.6 * 0.22
    c3 = np.cos(ang - light_dir - math.radians(95))
    lobe3 = np.clip(c3, 0, 1) ** 4.5 * 0.10

    r_norm = np.clip(r / 178.0, 0, 1.5)
    radial_spec = 0.18 + 0.62 * np.exp(-((r_norm - 0.58) ** 2) / 0.42)
    hub = np.clip(r / 22.0, 0, 1)
    hub = 0.48 + 0.52 * (hub ** 0.75)
    specular = (lobe1 * 0.48 + lobe2 + lobe3) * radial_spec

    shade = (
        0.07 + ridges * 0.22 + brush2 * 0.06 + micro * 0.85
        + specular * 0.48 - valleys * 0.07
    ) * hub
    shade = np.clip(shade, 0.03, 0.82)
    vignette = np.clip(1.02 - (r / 205.0) ** 1.2 * 0.52, 0.36, 1.0)
    shade = shade * vignette

    base = 4.0 + shade * 52.0
    hi = np.clip((shade - 0.42) / 0.58, 0, 1)
    glint = ridges * specular * 26.0
    r_ch = base * 0.90 + hi * 32 + glint * 0.75
    g_ch = base * 0.94 + hi * 36 + glint * 0.82
    b_ch = base * 1.05 + hi * 46 + glint * 0.95
    rgb = np.stack([
        np.clip(r_ch, 0, 255), np.clip(g_ch, 0, 255), np.clip(b_ch, 0, 255),
    ], -1).astype(np.uint8)
    amb = np.exp(-(((xx / S - 95) ** 2 + (yy / S - 72) ** 2) / 28000))
    rgb = np.clip(rgb.astype(np.float32) + amb[..., None] * np.array([5, 6, 8]), 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    img = img.filter(ImageFilter.GaussianBlur(0.45))

    paste_centered(img, ring_band(R_SEC, 4.5, (72, 76, 82), (190, 194, 200), 11, S), S=S)
    paste_centered(img, ring_band(R_MIN, 4.0, (88, 92, 100), (210, 214, 220), 22, S), S=S)
    paste_centered(img, ring_band(R_HOUR, 4.0, (108, 86, 52), (220, 180, 110), 33, S), S=S)

    d = ImageDraw.Draw(img, "RGBA")

    def tick(rad1, rad2, ang_d, col, width):
        a = math.radians(ang_d)
        x1, y1 = CX + rad1 * math.sin(a), CY - rad1 * math.cos(a)
        x2, y2 = CX + rad2 * math.sin(a), CY - rad2 * math.cos(a)
        d.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=col, width=max(1, int(width * S)))

    for i in range(60):
        L = 9 if i % 5 == 0 else 4
        col = (210, 208, 200, 230) if i % 5 == 0 else (100, 102, 108, 200)
        tick(R_SEC + 6, R_SEC + 6 + L, i * 6, col, 2 if i % 5 == 0 else 1)
    for i in range(12):
        tick(R_MIN - 11, R_MIN - 5, i * 30, (180, 186, 194, 210), 2)
    for i in range(12):
        a = math.radians(i * 30)
        x, y = CX + (R_HOUR + 9) * math.sin(a), CY - (R_HOUR + 9) * math.cos(a)
        d.ellipse([(x - 1.5) * S, (y - 1.5) * S, (x + 1.5) * S, (y + 1.5) * S],
                  fill=(200, 160, 90, 230))

    tri = [(CX - 7, 5), (CX + 7, 5), (CX, 16)]
    d.polygon([(x * S + 1.5, y * S + 1.5) for x, y in tri], fill=(0, 0, 0, 90))
    d.polygon([(x * S, y * S) for x, y in tri], fill=(220, 214, 200, 255))
    for rad in (R_SEC, R_MIN, R_HOUR):
        d.line([CX * S, (CY - rad - 6) * S, CX * S, (CY - rad + 6) * S],
               fill=(200, 196, 186, 160), width=2 * S)

    img = img.resize((W, H), Image.LANCZOS)

    # Bake static labels (arc STEPS/KAL + BAT/HR) into background
    draw_arc_label(img, "STEPS", center_ang=32, radius=R_BAND + 1,
                   color=(186, 194, 204), clockwise=True, font_size=11, tracking_deg=7.0)
    draw_arc_label(img, "KAL", center_ang=212, radius=R_BAND + 1,
                   color=C_BRONZE, clockwise=False, font_size=12, tracking_deg=8.0)

    f_lab = ImageFont.truetype(FONT_SM, 10)
    dd = ImageDraw.Draw(img, "RGBA")

    def text_centered(xy, text, font, fill, stroke=1):
        bb = dd.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        x, y = xy
        dd.text((x - tw / 2 - bb[0], y - th / 2 - bb[1]), text, font=font,
                fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0, 200))

    bx, by = polar(270, R_BAND)
    text_centered((bx, by - 6), "BAT", f_lab, C_BAT + (230,), 1)
    hx, hy = polar(90, R_BAND)
    text_centered((hx, hy - 8), "HR", f_lab, C_GUN + (230,), 1)

    # Mild contrast like preview
    rgb = np.array(ImageEnhance.Contrast(img.convert("RGB")).enhance(1.03))
    return rgb


# ---------------------------------------------------------------- digit / icon strips
def digit_strip(font_path, size, cw, ch, color, glyphs="0123456789"):
    font = ImageFont.truetype(font_path, size)
    strip = Image.new('RGB', (cw, ch * len(glyphs)), (0, 0, 0))
    for i, g in enumerate(glyphs):
        cell = Image.new('RGB', (cw * 4, ch * 4), (0, 0, 0))
        f4 = ImageFont.truetype(font_path, size * 4)
        dd = ImageDraw.Draw(cell)
        bb = dd.textbbox((0, 0), g, font=f4)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        dd.text(((cw * 4 - tw) / 2 - bb[0], (ch * 4 - th) / 2 - bb[1]), g, font=f4, fill=color)
        strip.paste(cell.resize((cw, ch), Image.LANCZOS), (0, i * ch))
    return np.array(strip)


def weekday_strip(cw, ch, color=PEWTER):
    """7 frames MON..SUN, Mon-first."""
    strip = Image.new('RGB', (cw, ch * 7), (0, 0, 0))
    font = ImageFont.truetype(FONT_MED, 13 * 4)
    for i, g in enumerate(WEEKDAYS):
        cell = Image.new('RGB', (cw * 4, ch * 4), (0, 0, 0))
        dd = ImageDraw.Draw(cell)
        bb = dd.textbbox((0, 0), g, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        dd.text(((cw * 4 - tw) / 2 - bb[0], (ch * 4 - th) / 2 - bb[1]), g, font=font, fill=color)
        strip.paste(cell.resize((cw, ch), Image.LANCZOS), (0, i * ch))
    return np.array(strip)


def small_icon(draw_fn, w, h):
    im = Image.new('RGB', (w * 4, h * 4), (0, 0, 0))
    draw_fn(ImageDraw.Draw(im), 4)
    return np.array(im.resize((w, h), Image.LANCZOS))


# ---------------------------------------------------------------- hands (type 0x1A) — metal / enamel markers
def hand_frames_metal(R, half_deg, thick, color, P, box):
    """Brushed-metal marker arc with specular highlight + champagne bead."""
    S = 4
    frames = []
    px, py = P * S, (box - P) * S
    hi_col = tuple(min(255, int(c * 1.25 + 40)) for c in color)
    for k in range(16):
        th = 6 * k
        layer = Image.new('RGBA', (box * S, box * S), (0, 0, 0, 0))
        shadow = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        body = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        hi = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        sd, ld, hd = ImageDraw.Draw(shadow), ImageDraw.Draw(body), ImageDraw.Draw(hi)

        def bb(rad, ox=0, oy=0):
            return [px - rad * S + ox, py - rad * S + oy, px + rad * S + ox, py + rad * S + oy]

        a0, a1 = th - 90 - half_deg, th - 90 + half_deg
        sd.arc(bb(R + thick / 2, 2 * S, 3 * S), a0, a1, fill=(0, 0, 0, 100),
               width=int((thick + 2) * S))
        ld.arc(bb(R + thick / 2), a0, a1, fill=color + (255,), width=int(thick * S))
        hd.arc(bb(R + thick / 2 - 1.2), a0 + 1, a1 - 1, fill=hi_col + (180,),
               width=max(1, int(thick * 0.35 * S)))
        for a in (a0, a1):
            ar = math.radians(a)
            ex, ey = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
            rr = thick * S / 2
            ld.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], fill=color + (255,))
            hd.ellipse([ex - rr * 0.45, ey - rr * 0.55, ex + rr * 0.35, ey + rr * 0.25],
                       fill=hi_col + (200,))
        ar = math.radians(th - 90)
        bx, by = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
        rb = (thick * 0.42) * S
        ld.ellipse([bx - rb + S, by - rb + S, bx + rb + S, by + rb + S], fill=(0, 0, 0, 100))
        ld.ellipse([bx - rb, by - rb, bx + rb, by + rb], fill=(236, 228, 210, 255))
        hd.ellipse([bx - rb * 0.5, by - rb * 0.7, bx + rb * 0.2, by - rb * 0.05],
                   fill=(255, 250, 240, 200))

        shadow = shadow.filter(ImageFilter.GaussianBlur(1.2 * S))
        layer.alpha_composite(shadow)
        layer.alpha_composite(body)
        layer.alpha_composite(hi)
        fr = np.array(layer.resize((box, box), Image.LANCZOS))
        fr[fr[:, :, 3] < 10] = 0
        frames.append(fr)
    return frames


def hand_frames_enamel(R, half_deg, thick, P, box, rim_rgb=C_SILVER):
    """Seconds plaque: glossy black jewelry enamel + thin polished metal rim."""
    S = 4
    frames = []
    px, py = P * S, (box - P) * S
    enamel = (6, 6, 8, 255)
    rim_hi = tuple(min(255, int(c * 1.2 + 36)) for c in rim_rgb)
    rim_w = max(0.9, thick * 0.18)
    for k in range(16):
        th = 6 * k
        layer = Image.new('RGBA', (box * S, box * S), (0, 0, 0, 0))
        shadow = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        body = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        rim_l = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        gloss = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        bd = ImageDraw.Draw(body)
        rd = ImageDraw.Draw(rim_l)
        gd = ImageDraw.Draw(gloss)

        def bb(rad, ox=0, oy=0):
            return [px - rad * S + ox, py - rad * S + oy, px + rad * S + ox, py + rad * S + oy]

        a0, a1 = th - 90 - half_deg, th - 90 + half_deg
        sd.arc(bb(R + thick / 2, 2 * S, 3 * S), a0, a1, fill=(0, 0, 0, 130),
               width=int((thick + 2.4) * S))
        bd.arc(bb(R + thick / 2), a0, a1, fill=enamel, width=int(thick * S))
        for a in (a0, a1):
            ar = math.radians(a)
            ex, ey = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
            rr = thick * S / 2
            bd.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], fill=enamel)

        outer_r = R + thick / 2
        rd.arc(bb(outer_r), a0, a1, fill=rim_rgb + (255,), width=max(1, int(rim_w * S)))
        rd.arc(bb(R + rim_w * 0.35), a0, a1, fill=rim_rgb + (235,),
               width=max(1, int(rim_w * 0.85 * S)))
        gd.arc(bb(outer_r - 0.25), a0 + 0.4, a1 - 0.4, fill=rim_hi + (210,),
               width=max(1, int(rim_w * 0.45 * S)))

        for a in (a0, a1):
            ar = math.radians(a)
            ex, ey = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
            rr = thick * S / 2
            rd.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], outline=rim_rgb + (255,),
                       width=max(1, int(rim_w * S)))
            ir = rr - max(1, int(rim_w * S))
            if ir > 1:
                bd.ellipse([ex - ir, ey - ir, ex + ir, ey + ir], fill=enamel)
            gd.ellipse([ex - rr * 0.5, ey - rr * 0.65, ex + rr * 0.1, ey - rr * 0.1],
                       fill=rim_hi + (160,))

        gd.arc(bb(R + thick / 2 - thick * 0.08), a0 + half_deg * 0.22, a1 - half_deg * 0.18,
               fill=(190, 198, 210, 70), width=max(1, int(thick * 0.22 * S)))
        gd.arc(bb(R + thick / 2 + thick * 0.02), a0 + half_deg * 0.38, a0 + half_deg * 0.78,
               fill=(255, 255, 255, 40), width=max(1, int(thick * 0.12 * S)))

        ar = math.radians(th - 90)
        bx, by = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
        rb = (thick * 0.36) * S
        bd.ellipse([bx - rb + S, by - rb + S, bx + rb + S, by + rb + S], fill=(0, 0, 0, 120))
        rd.ellipse([bx - rb, by - rb, bx + rb, by + rb], fill=(236, 228, 210, 255))
        gd.ellipse([bx - rb * 0.5, by - rb * 0.7, bx + rb * 0.2, by - rb * 0.05],
                   fill=(255, 250, 240, 200))

        shadow = shadow.filter(ImageFilter.GaussianBlur(1.2 * S))
        gloss = gloss.filter(ImageFilter.GaussianBlur(0.4 * S))
        layer.alpha_composite(shadow)
        layer.alpha_composite(body)
        layer.alpha_composite(rim_l)
        layer.alpha_composite(gloss)
        fr = np.array(layer.resize((box, box), Image.LANCZOS))
        fr[fr[:, :, 3] < 10] = 0
        frames.append(fr)
    return frames


def encode_hand(frames):
    pix = bytearray()
    tab = bytearray()
    for fr in frames:
        h, w, _ = fr.shape
        for r in range(h):
            a = fr[r, :, 3]
            nz = np.nonzero(a)[0]
            off = len(pix)
            if len(nz) == 0:
                tab += struct.pack('<HHHH', off >> 16, off & 0xFFFF, 0, 0)
                continue
            x0, x1 = int(nz[0]), int(nz[-1])
            for x in range(x0, x1 + 1):
                R_, G_, B_, A_ = fr[r, x]
                c = to565(R_, G_, B_) if A_ else 0
                pix += bytes((int(A_), c & 0xFF, c >> 8))
            tab += struct.pack('<HHHH', off >> 16, off & 0xFFFF, x0, x1 - x0 + 1)
    return bytes(pix), bytes(tab)


# ---------------------------------------------------------------- writer
class Dial:
    def __init__(self):
        self.comps = []
        self.res = {}
        self.order = []

    def add_res(self, key, a, b):
        if key not in self.res:
            self.res[key] = (a, b)
            self.order.append(key)

    def comp(self, t, b1, b2, b3, x, y, w, h, key):
        self.comps.append(dict(t=t, b1=b1, b2=b2, b3=b3, x=x, y=y, w=w, h=h, key=key))

    def build(self):
        n = len(self.comps)
        pos = 4 + 20 * n
        offs = {}
        body = bytearray()
        for key in self.order:
            a, b = self.res[key]
            offs[key] = (pos, pos + len(a))
            body += a + b
            pos += len(a) + len(b)
        out = bytearray(struct.pack('<I', n))
        for c in self.comps:
            clt, dat = offs[c['key']]
            out += struct.pack('<BBBBHHHHII', c['t'], c['b1'], c['b2'], c['b3'],
                               c['x'], c['y'], c['w'], c['h'], clt, dat)
        out += body + TAG
        return bytes(out)


def add_digits(dial, type_id, b1, b2, b3, xs, y, cw, ch, key):
    """Add digit components units-first (xs[0] = units = rightmost)."""
    for x in xs:
        dial.comp(type_id, b1, b2, b3, x, y, cw, ch * 10, key)


def main():
    dial = Dial()

    # 0: static background
    print("building background…", flush=True)
    bg = make_background()
    pal, idx = quantize_idx8(bg, 255, dither=True, black_transparent=False)
    dial.add_res('bg', *idx8_blob(pal, idx))
    dial.comp(0x09, 0x01, 0x00, 0x00, 0, 0, 360, 360, 'bg')

    # Digits
    CW, CH = 26, 44          # large HH:MM (closer to preview size-62 look)
    SW, SH = 10, 17          # widget values
    big = digit_strip(FONT_BIG, 48, CW, CH, CHAMPAGNE)
    pal, idx = quantize_idx8(big, 24)
    dial.add_res('big', *idx8_blob(pal, idx))
    small = digit_strip(FONT_MED, 18, SW, SH, CHAMPAGNE)
    pal, idx = quantize_idx8(small, 16)
    dial.add_res('small', *idx8_blob(pal, idx))

    # --- CENTER: weekday / HH:MM / date ---
    WD_W, WD_H = 42, 14
    wd = weekday_strip(WD_W, WD_H, PEWTER)
    pal, idx = quantize_idx8(wd, 16)
    dial.add_res('weekday', *idx8_blob(pal, idx))
    # weekday above time: preview CY-40 = 140
    dial.comp(0x06, 0x87, 0x00, 0x82, CX - WD_W // 2, 133, WD_W, WD_H * 7, 'weekday')

    y_t = CY - CH // 2 + 2          # ~160, time visually centered near CY+2
    # colon between HH and MM
    colon_w = 10
    gap = 2
    total_w = CW * 4 + colon_w + gap * 4
    x0 = CX - total_w // 2
    x_ht = x0
    x_hu = x0 + CW + gap
    x_col = x_hu + CW + gap
    x_mt = x_col + colon_w + gap
    x_mu = x_mt + CW + gap
    colon = small_icon(
        lambda d, s: [
            d.ellipse([3 * s, 12 * s, 7 * s, 16 * s], fill=CHAMPAGNE),
            d.ellipse([3 * s, 28 * s, 7 * s, 32 * s], fill=CHAMPAGNE),
        ], colon_w, CH)
    pal, idx = quantize_idx8(colon, 12)
    dial.add_res('colon', *idx8_blob(pal, idx))
    dial.comp(0x09, 0x01, 0x00, 0x00, x_col, y_t, colon_w, CH, 'colon')
    # hours units then tens; minutes units then tens (149.bin convention)
    dial.comp(0x00, 0x8A, 0x82, 0x00, x_hu, y_t, CW, CH * 10, 'big')
    dial.comp(0x00, 0x8A, 0x82, 0x00, x_ht, y_t, CW, CH * 10, 'big')
    dial.comp(0x01, 0x8A, 0x82, 0x00, x_mu, y_t, CW, CH * 10, 'big')
    dial.comp(0x01, 0x8A, 0x82, 0x00, x_mt, y_t, CW, CH * 10, 'big')

    # date MM-DD below time (preview CY+40 = 220)
    y_d = 214
    dash = small_icon(lambda d, s: d.rectangle([0, 0, 7 * s, 2 * s], fill=PEWTER), 8, 3)
    pal, idx = quantize_idx8(dash, 8)
    dial.add_res('dash', *idx8_blob(pal, idx))
    dial.comp(0x09, 0x01, 0x00, 0x00, 176, y_d + 7, 8, 3, 'dash')
    dial.comp(0x03, 0x8A, 0x82, 0x00, 164, y_d, SW, SH * 10, 'small')  # month units
    dial.comp(0x03, 0x8A, 0x82, 0x00, 153, y_d, SW, SH * 10, 'small')  # month tens
    dial.comp(0x02, 0x8A, 0x82, 0x00, 197, y_d, SW, SH * 10, 'small')  # day units
    dial.comp(0x02, 0x8A, 0x82, 0x00, 186, y_d, SW, SH * 10, 'small')  # day tens

    # --- TOP: STEPS (5 digits), label baked in BG to the right ---
    sx, sy = polar(0, R_BAND)
    steps_y = int(sy - SH / 2) + 1
    # value centered on sx; units-first xs right→left
    steps_total = SW * 5
    steps_left = int(sx - steps_total / 2)
    steps_xs = [steps_left + SW * i for i in range(4, -1, -1)]  # units..10000s
    add_digits(dial, 0x0E, 0x8A, 0x85, 0x00, steps_xs, steps_y, SW, SH, 'small')

    # --- LEFT: BAT value + % under (BAT label in BG) ---
    bx, by = polar(270, R_BAND)
    bat_y = int(by + 8 - SH / 2)
    bat_total = SW * 3
    bat_left = int(bx - bat_total / 2)
    bat_xs = [bat_left + SW * i for i in range(2, -1, -1)]  # units, tens, hundreds
    add_digits(dial, 0x0B, 0x8A, 0x03, 0x00, bat_xs, bat_y, SW, SH, 'small')
    pct = small_icon(
        lambda d, s: d.text((0, -1 * s), "%",
                            font=ImageFont.truetype(FONT_SM, 12 * s), fill=PEWTER),
        12, 14)
    pal, idx = quantize_idx8(pct, 12)
    dial.add_res('pct', *idx8_blob(pal, idx))
    pct_y = int(by + 22 - 7)
    dial.comp(0x09, 0x01, 0x00, 0x00, int(bx - 6), pct_y, 12, 14, 'pct')

    # --- RIGHT: HR (HR label in BG) ---
    hx, hy = polar(90, R_BAND)
    hr_y = int(hy + 7 - SH / 2)
    hr_total = SW * 3
    hr_left = int(hx - hr_total / 2)
    hr_xs = [hr_left + SW * i for i in range(2, -1, -1)]
    add_digits(dial, 0x10, 0x8A, 0x83, 0x00, hr_xs, hr_y, SW, SH, 'small')

    # --- BOTTOM: KAL / calories (4 digits), arc label baked left ---
    kx, ky = polar(180, R_BAND)
    kal_y = int(ky - SH / 2) + 1
    kal_total = SW * 4
    kal_left = int(kx - kal_total / 2)
    kal_xs = [kal_left + SW * i for i in range(3, -1, -1)]
    add_digits(dial, 0x0F, 0x8A, 0x84, 0x00, kal_xs, kal_y, SW, SH, 'small')

    # --- HANDS 0x1A ---
    print("building hands…", flush=True)
    hands = [
        # (b3, R, half, thick, kind)
        (0, R_HOUR, 14, 8, 'metal', C_BRONZE),
        (1, R_MIN, 10, 7, 'metal', C_SILVER),
        (2, R_SEC, 7, 6, 'enamel', C_SILVER),
    ]
    for b3, R, half, thick, kind, col in hands:
        P = int(math.ceil(R * math.sin(math.radians(half + 2)) + thick / 2 + 16))
        box = P + R + thick // 2 + 18
        if kind == 'enamel':
            frames = hand_frames_enamel(R, half, thick, P, box, rim_rgb=col)
        else:
            frames = hand_frames_metal(R, half, thick, col, P, box)
        pix, tab = encode_hand(frames)
        key = f'hand{b3}'
        dial.add_res(key, pix, tab)
        dial.comp(0x1A, 0x13, P, b3, CX, CY, box, box, key)
        print(f"  hand b3={b3} P={P} box={box} pix={len(pix)} tab={len(tab)}", flush=True)

    data = dial.build()
    path = os.path.join(OUT, 'x5pro_rings.bin')
    open(path, 'wb').write(data)
    print(f"wrote {path}: {len(data)} bytes, {len(dial.comps)} components")
    if len(data) > 980_000:
        print("WARNING: size exceeds known-safe X5Pro max (~980KB)", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
