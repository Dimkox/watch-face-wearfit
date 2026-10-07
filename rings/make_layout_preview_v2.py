#!/usr/bin/env python3
"""Layout mockup v2 — preview only, no .bin.
Black sunburst dial (Rolex Bright Black style); noble metal rings; arc labels for STEPS/KAL; BAT % under value."""
import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OUT = os.path.dirname(os.path.abspath(__file__))
W = H = 360
CX = CY = 180
S = 3  # supersample

R_HOUR, R_MIN, R_SEC = 84, 121, 158
# Marker / accent metals
C_BRONZE = (196, 152, 86)
C_SILVER = (198, 204, 212)
C_GUN = (168, 172, 180)
C_BAT = (176, 180, 160)

FONT_BIG = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Bold.ttf"
FONT_MED = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-SemiBold.ttf"
FONT_SM = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Medium.ttf"


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


def brushed_disk(radius, base_rgb, hi_rgb, seed):
    """Photographic brushed-metal annular/full disk with radial specular."""
    size = int(2 * radius * S + 8 * S)
    cy = cx = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx - cx) / S; dy = (yy - cy) / S
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    # circumferential brush strokes + fine grain
    brush = 0.55 + 0.45 * np.sin(ang * 48 + noise2d(size, size, 10, seed) * 6)
    grain = noise2d(size, size, 4, seed + 7) * 0.18
    # soft radial specular (top-left light)
    light = np.clip(0.55 + 0.55 * np.cos(ang - math.radians(-40)), 0, 1)
    light = light * (0.35 + 0.65 * np.exp(-((r / (radius + 1e-3) - 0.55) ** 2) / 0.35))
    shade = brush * 0.55 + grain + light * 0.55
    shade = np.clip(shade, 0.15, 1.15)
    rgb = np.zeros((size, size, 3), np.float32)
    for i in range(3):
        rgb[..., i] = base_rgb[i] * (0.55 + 0.45 * shade) + hi_rgb[i] * np.clip(shade - 0.7, 0, 1) * 0.9
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    # soft edge
    edge = np.clip((radius + 1.2 - r) * S, 0, 1)
    alpha = (edge * 255).astype(np.uint8)
    out = np.dstack([rgb, alpha])
    return Image.fromarray(out, "RGBA")


def ring_band(radius, width, base_rgb, hi_rgb, seed):
    """Brushed metal ring of given centerline radius and stroke width."""
    disk = brushed_disk(radius + width / 2 + 2, base_rgb, hi_rgb, seed)
    # punch inner hole with soft edge
    size = disk.size[0]
    cy = cx = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot((xx - cx) / S, (yy - cy) / S)
    arr = np.array(disk)
    inner = radius - width / 2
    fade = np.clip((r - (inner - 0.8)) * S, 0, 1)
    outer = np.clip(((radius + width / 2 + 0.8) - r) * S, 0, 1)
    arr[..., 3] = (arr[..., 3].astype(np.float32) * fade * outer).astype(np.uint8)
    # bevel highlight on outer/inner lips
    lip = np.exp(-((r - (radius + width / 2 * 0.85)) ** 2) / 1.8) * 0.35
    lip += np.exp(-((r - (radius - width / 2 * 0.85)) ** 2) / 1.8) * 0.25
    for i in range(3):
        arr[..., i] = np.clip(arr[..., i] + lip * (hi_rgb[i] - arr[..., i]), 0, 255)
    return Image.fromarray(arr, "RGBA")


def paste_centered(canvas, layer, cx=CX, cy=CY):
    x = int(cx * S - layer.size[0] / 2)
    y = int(cy * S - layer.size[1] / 2)
    canvas.alpha_composite(layer, (x, y))


def make_bg():
    # Photoreal black sunburst (sunray) dial — Rolex Bright Black style:
    # fine radial grooves from center + directional light lobes.
    yy, xx = np.mgrid[0:H * S, 0:W * S]
    dx = (xx / S - CX).astype(np.float32)
    dy = (yy / S - CY).astype(np.float32)
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)

    # Fine radial grooves — moderate freq to avoid moiré after downscale
    n_ang = noise2d(H * S, W * S, 22, 5)
    # Primary sunrays (~140 lines) + softer secondary set
    g1 = np.sin(ang * 140.0 + n_ang * 1.8)
    g2 = np.sin(ang * 48.0 + noise2d(H * S, W * S, 30, 9) * 1.2)
    # Squared lobes → thin bright ridges on dark valleys (brushed metal)
    ridges = np.clip(g1, 0, 1) ** 2.2
    valleys = np.clip(-g1, 0, 1) ** 1.5
    brush2 = 0.5 + 0.5 * g2
    micro = noise2d(H * S, W * S, 4.0, 11) * 0.08

    # Specular lobes (studio light upper-left) — softer flash so labels stay readable
    light_dir = math.radians(-48)
    # wrap-aware soft lobe: higher power → narrower, less washed-out hotspot
    c1 = np.cos(ang - light_dir)
    lobe1 = np.clip(c1, 0, 1) ** 4.2
    c2 = np.cos(ang - light_dir - math.pi)
    lobe2 = np.clip(c2, 0, 1) ** 3.6 * 0.22
    # mild tertiary lobe for richness
    c3 = np.cos(ang - light_dir - math.radians(95))
    lobe3 = np.clip(c3, 0, 1) ** 4.5 * 0.10

    r_norm = np.clip(r / 178.0, 0, 1.5)
    # Specular mid-radius, gentler peak; keep hub subdued
    radial_spec = 0.18 + 0.62 * np.exp(-((r_norm - 0.58) ** 2) / 0.42)
    hub = np.clip(r / 22.0, 0, 1)
    hub = 0.48 + 0.52 * (hub ** 0.75)

    specular = (lobe1 * 0.48 + lobe2 + lobe3) * radial_spec

    # Bright Black base: deeper ink, restrained shimmer (labels readable in Q1)
    # valleys deepen the black between rays
    shade = (
        0.07
        + ridges * 0.22
        + brush2 * 0.06
        + micro * 0.85
        + specular * 0.48
        - valleys * 0.07
    ) * hub
    shade = np.clip(shade, 0.03, 0.82)

    # Soft vignette toward case
    vignette = np.clip(1.02 - (r / 205.0) ** 1.2 * 0.52, 0.36, 1.0)
    shade = shade * vignette

    # Cool black metal palette — lower ceiling, softer highlights
    base = 4.0 + shade * 52.0
    hi = np.clip((shade - 0.42) / 0.58, 0, 1)
    glint = ridges * specular * 26.0
    r_ch = base * 0.90 + hi * 32 + glint * 0.75
    g_ch = base * 0.94 + hi * 36 + glint * 0.82
    b_ch = base * 1.05 + hi * 46 + glint * 0.95  # cooler specular

    rgb = np.stack([
        np.clip(r_ch, 0, 255),
        np.clip(g_ch, 0, 255),
        np.clip(b_ch, 0, 255),
    ], -1).astype(np.uint8)

    # Soft top-left ambient bloom (reduced — was washing STEPS)
    amb = np.exp(-(((xx / S - 95) ** 2 + (yy / S - 72) ** 2) / 28000))
    rgb = np.clip(rgb.astype(np.float32) + amb[..., None] * np.array([5, 6, 8]), 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    # Mild blur at supersample then rings — kills residual moiré, keeps rays
    img = img.filter(ImageFilter.GaussianBlur(0.55))

    # metal rings
    paste_centered(img, ring_band(R_SEC, 4.5, (72, 76, 82), (190, 194, 200), 11))
    paste_centered(img, ring_band(R_MIN, 4.0, (88, 92, 100), (210, 214, 220), 22))
    paste_centered(img, ring_band(R_HOUR, 4.0, (108, 86, 52), (220, 180, 110), 33))

    d = ImageDraw.Draw(img, "RGBA")

    def tick(rad1, rad2, ang, col, width):
        a = math.radians(ang)
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

    # 12 index — engraved silver triangle with soft shadow
    tri = [(CX - 7, 5), (CX + 7, 5), (CX, 16)]
    d.polygon([(x * S + 1.5, y * S + 1.5) for x, y in tri], fill=(0, 0, 0, 90))
    d.polygon([(x * S, y * S) for x, y in tri], fill=(220, 214, 200, 255))
    for rad in (R_SEC, R_MIN, R_HOUR):
        d.line([CX * S, (CY - rad - 6) * S, CX * S, (CY - rad + 6) * S],
               fill=(200, 196, 186, 160), width=2 * S)

    return img.resize((W, H), Image.LANCZOS)


def marker(R, half_deg, thick, color, angle_deg):
    """Metal marker arc with soft specular highlight (not neon glow)."""
    layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    body = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    hi = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ld = ImageDraw.Draw(body)
    hd = ImageDraw.Draw(hi)
    px, py = CX * S, CY * S

    def bb(rad, ox=0, oy=0):
        return [px - rad * S + ox, py - rad * S + oy, px + rad * S + ox, py + rad * S + oy]

    a0 = angle_deg - 90 - half_deg
    a1 = angle_deg - 90 + half_deg
    # drop shadow
    sd.arc(bb(R + thick / 2, 2 * S, 3 * S), a0, a1, fill=(0, 0, 0, 100), width=int((thick + 2) * S))
    # body
    ld.arc(bb(R + thick / 2), a0, a1, fill=color + (255,), width=int(thick * S))
    # brighter inner edge (brushed highlight)
    hi_col = tuple(min(255, int(c * 1.25 + 40)) for c in color)
    hd.arc(bb(R + thick / 2 - 1.2), a0 + 1, a1 - 1, fill=hi_col + (180,), width=max(1, int(thick * 0.35 * S)))
    for a in (a0, a1):
        ar = math.radians(a)
        ex, ey = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
        rr = thick * S / 2
        ld.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], fill=color + (255,))
        hd.ellipse([ex - rr * 0.45, ey - rr * 0.55, ex + rr * 0.35, ey + rr * 0.25],
                   fill=hi_col + (200,))
    # champagne bead
    ar = math.radians(angle_deg - 90)
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
    return layer.resize((W, H), Image.LANCZOS)


def marker_enamel(R, half_deg, thick, angle_deg, rim_rgb=(198, 204, 212)):
    """Seconds plaque: deep glossy black jewelry enamel + thin polished metal rim."""
    layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    body = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    rim_l = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    gloss = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    bd = ImageDraw.Draw(body)
    rd = ImageDraw.Draw(rim_l)
    gd = ImageDraw.Draw(gloss)
    px, py = CX * S, CY * S

    def bb(rad, ox=0, oy=0):
        return [px - rad * S + ox, py - rad * S + oy, px + rad * S + ox, py + rad * S + oy]

    a0 = angle_deg - 90 - half_deg
    a1 = angle_deg - 90 + half_deg
    rim_w = max(0.9, thick * 0.18)
    enamel = (6, 6, 8, 255)
    rim_hi = tuple(min(255, int(c * 1.2 + 36)) for c in rim_rgb)

    # drop shadow
    sd.arc(bb(R + thick / 2, 2 * S, 3 * S), a0, a1, fill=(0, 0, 0, 130), width=int((thick + 2.4) * S))

    # 1) deep black enamel body (full plaque mass)
    bd.arc(bb(R + thick / 2), a0, a1, fill=enamel, width=int(thick * S))
    for a in (a0, a1):
        ar = math.radians(a)
        ex, ey = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
        rr = thick * S / 2
        bd.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], fill=enamel)

    # 2) thin polished noble-metal rim on outer / inner lips + end bezels
    outer_r = R + thick / 2
    rd.arc(bb(outer_r), a0, a1, fill=rim_rgb + (255,), width=max(1, int(rim_w * S)))
    rd.arc(bb(R + rim_w * 0.35), a0, a1, fill=rim_rgb + (235,), width=max(1, int(rim_w * 0.85 * S)))
    gd.arc(bb(outer_r - 0.25), a0 + 0.4, a1 - 0.4, fill=rim_hi + (210,), width=max(1, int(rim_w * 0.45 * S)))

    for a in (a0, a1):
        ar = math.radians(a)
        ex, ey = px + R * S * math.cos(ar), py + R * S * math.sin(ar)
        rr = thick * S / 2
        rd.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], outline=rim_rgb + (255,), width=max(1, int(rim_w * S)))
        ir = rr - max(1, int(rim_w * S))
        if ir > 1:
            bd.ellipse([ex - ir, ey - ir, ex + ir, ey + ir], fill=enamel)
        gd.ellipse([ex - rr * 0.5, ey - rr * 0.65, ex + rr * 0.1, ey - rr * 0.1],
                   fill=rim_hi + (160,))

    # 3) wet enamel gloss (narrow cool specular, not brushed)
    gd.arc(bb(R + thick / 2 - thick * 0.08), a0 + half_deg * 0.22, a1 - half_deg * 0.18,
           fill=(190, 198, 210, 70), width=max(1, int(thick * 0.22 * S)))
    gd.arc(bb(R + thick / 2 + thick * 0.02), a0 + half_deg * 0.38, a0 + half_deg * 0.78,
           fill=(255, 255, 255, 40), width=max(1, int(thick * 0.12 * S)))

    # small champagne cabochon (jewelry accent)
    ar = math.radians(angle_deg - 90)
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


def draw_arc_label(can, text, center_ang, radius, color, clockwise=True, font_size=11, tracking_deg=7.2):
    """Place rotated letter sprites along a circular arc.
    center_ang: angle of the label's midpoint (0=12 o'clock, clockwise).
    clockwise=True means letters read clockwise along the arc."""
    font = ImageFont.truetype(FONT_SM, font_size * S)
    n = len(text)
    # total angular span
    span = (n - 1) * tracking_deg
    start = center_ang - span / 2 if clockwise else center_ang + span / 2
    step = tracking_deg if clockwise else -tracking_deg
    for i, ch in enumerate(text):
        ang = start + i * step
        # glyph sprite
        bb = font.getbbox(ch)
        pad = 4 * S
        gw = bb[2] - bb[0] + pad * 2
        gh = bb[3] - bb[1] + pad * 2
        glyph = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glyph)
        # soft shadow
        gd.text((pad - bb[0] + S, pad - bb[1] + S), ch, font=font, fill=(0, 0, 0, 110))
        gd.text((pad - bb[0], pad - bb[1]), ch, font=font, fill=color + (240,))
        # rotation: tangent to circle. For clockwise reading on top-right,
        # letter upright-ish relative to outward normal.
        # Angle of radius from 12 clockwise = ang; screen rotation of letter:
        # outward normal points along ang; we want letter baseline tangential.
        rot = ang if clockwise else ang + 180
        glyph = glyph.rotate(-rot, resample=Image.BICUBIC, expand=True)
        lx, ly = polar(ang, radius)
        # paste at supersampled coords onto a hi-res overlay then downscale contribution
        # Work on a full-res overlay matching can (already screen-res): scale glyph down
        g_small = glyph.resize((max(1, glyph.size[0] // S), max(1, glyph.size[1] // S)), Image.LANCZOS)
        px = int(lx - g_small.size[0] / 2)
        py = int(ly - g_small.size[1] / 2)
        can.alpha_composite(g_small, (px, py))


def draw_widgets(can, hh=10, mm=8, steps=2735, hr=72, kcal=163, batt=85,
                 weekday="WED", mon=10, day=7):
    d = ImageDraw.Draw(can, "RGBA")
    f_time = ImageFont.truetype(FONT_BIG, 62)
    f_wd = ImageFont.truetype(FONT_MED, 13)
    f_val = ImageFont.truetype(FONT_MED, 17)
    f_lab = ImageFont.truetype(FONT_SM, 10)
    f_date = ImageFont.truetype(FONT_MED, 18)
    f_pct = ImageFont.truetype(FONT_SM, 12)
    R_BAND = (R_SEC + R_MIN) / 2
    champagne = (236, 230, 218, 255)
    pewter = (186, 178, 160, 255)

    # CENTER
    text_centered(d, (CX, CY - 40), weekday, f_wd, pewter, stroke=1)
    text_centered(d, (CX, CY + 2), f"{hh:02d}:{mm:02d}", f_time, champagne, stroke=2)
    text_centered(d, (CX, CY + 40), f"{mon:02d}-{day:02d}", f_date, pewter, stroke=1)

    # TOP: steps value + arc label to the RIGHT of the value
    sx, sy = polar(0, R_BAND)
    text_centered(d, (sx, sy + 2), f"{steps:05d}", f_val, champagne, stroke=1)
    # arc "STEPS" along the band, to the right of top (angles ~12°..42°)
    draw_arc_label(can, "STEPS", center_ang=32, radius=R_BAND + 1,
                   color=(186, 194, 204), clockwise=True, font_size=11, tracking_deg=7.0)

    # LEFT: battery — value, percent UNDER
    bx, by = polar(270, R_BAND)
    text_centered(d, (bx, by - 6), "BAT", f_lab, C_BAT + (230,), stroke=1)
    text_centered(d, (bx, by + 8), f"{batt:03d}", f_val, champagne, stroke=1)
    text_centered(d, (bx, by + 22), "%", f_pct, pewter, stroke=1)

    # RIGHT: HR
    hx, hy = polar(90, R_BAND)
    text_centered(d, (hx, hy - 8), "HR", f_lab, C_GUN + (230,), stroke=1)
    text_centered(d, (hx, hy + 7), f"{hr:03d}", f_val, champagne, stroke=1)

    # BOTTOM: calories value + arc "KAL" to the LEFT of value
    cx_, cy_ = polar(180, R_BAND)
    text_centered(d, (cx_, cy_ + 2), f"{kcal:04d}", f_val, champagne, stroke=1)
    # arc to the LEFT of bottom value; reading COUNTERCLOCKWISE (against the clock)
    draw_arc_label(can, "KAL", center_ang=212, radius=R_BAND + 1,
                   color=C_BRONZE, clockwise=False, font_size=12, tracking_deg=8.0)


def render(hh, mm, ss):
    can = make_bg()
    h_ang = ((hh % 12) * 30 + mm * 0.5)
    m_ang = mm * 6 + ss * 0.1
    s_ang = ss * 6
    can.alpha_composite(marker(R_HOUR, 14, 8, C_BRONZE, h_ang))
    can.alpha_composite(marker(R_MIN, 10, 7, C_SILVER, m_ang))
    can.alpha_composite(marker_enamel(R_SEC, 7, 6, s_ang, rim_rgb=C_SILVER))
    draw_widgets(can, hh=hh, mm=mm)
    # subtle overall contrast
    enh = ImageEnhance.Contrast(can.convert("RGB"))
    return enh.enhance(1.03)


def main():
    still = render(10, 8, 36)
    path = os.path.join(OUT, "layout_preview_v2.png")
    still.save(path)
    print("wrote", path, still.size)

    frames = []
    for s in range(0, 60, 2):
        frames.append(render(10, 8, s).quantize(colors=220, method=Image.Quantize.MEDIANCUT))
    for m in range(9, 21):
        frames.append(render(10, m, (m * 7) % 60).quantize(colors=220, method=Image.Quantize.MEDIANCUT))
    gpath = os.path.join(OUT, "layout_preview_v2.gif")
    frames[0].save(gpath, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print("wrote", gpath, len(frames), "frames")


if __name__ == "__main__":
    main()
