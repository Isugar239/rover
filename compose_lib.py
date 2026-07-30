#!/usr/bin/env python3
"""Склейка: готовая Шапокляк + AI-фон + текст."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets", "poses")
POSES_JSON = os.path.join(ASSETS, "poses.json")
OUT_DIR = os.path.join(ROOT, "итог")
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass
class Pose:
    id: str
    file: str
    look: str  # left | right | front
    slot: str  # left | center | right
    scale: float
    y_bias: float
    light: str  # left | right | front
    notes: str = ""

    @property
    def path(self) -> str:
        return os.path.join(ASSETS, self.file)


def load_poses() -> list[Pose]:
    with open(POSES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Pose(**item) for item in data["poses"]]


def get_pose(pose_id: str) -> Pose:
    for p in load_poses():
        if p.id == pose_id:
            return p
    raise KeyError(f"Нет позы: {pose_id}")


THEMES = ("cake", "party", "family")


def pick_theme(rng: random.Random | None = None) -> str:
    rng = rng or random
    # cake чаще, но не всегда — иначе сцена однообразная
    return rng.choices(THEMES, weights=[0.70, 0.18, 0.12], k=1)[0]


def pick_pose(theme: str = "cake", rng: random.Random | None = None) -> Pose:
    """По умолчанию left_look_right — нормальный вырез; waist_up_cake исключён (битый файл)."""
    rng = rng or random
    poses = [p for p in load_poses() if os.path.isfile(p.path)]
    if not poses:
        raise FileNotFoundError("Нет ни одного выреза в assets/poses/")
    # почти всегда хороший профиль; front_gesture реже
    weights = {
        "cake": {"left_look_right": 5, "front_gesture": 1},
        "party": {"left_look_right": 4, "front_gesture": 2},
        "family": {"left_look_right": 4, "front_gesture": 2},
    }.get(theme, {})
    w = [weights.get(p.id, 1) for p in poses]
    return rng.choices(poses, weights=w, k=1)[0]


def trim_transparent(im: Image.Image, pad: int = 4) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    bbox = im.split()[-1].getbbox()
    if not bbox:
        return im
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(im.width, r + pad)
    b = min(im.height, b + pad)
    return im.crop((l, t, r, b))


def soft_mask(alpha: Image.Image, feather: float = 0.8) -> Image.Image:
    """Смягчить край маски без ореола (не расширять силуэт)."""
    a = alpha.point(lambda v: 255 if v > 35 else 0)
    radius = max(1, int(round(feather)))
    a = a.filter(ImageFilter.GaussianBlur(radius))
    # чуть ужать край — убирает cyan/white fringe от rembg
    a = a.point(lambda v: max(0, int(v * 0.97 - 4)))
    return a


def match_lighting(cutout: Image.Image, light: str) -> Image.Image:
    """Лёгкая тоновая подстройка под направление света фона."""
    img = cutout.convert("RGBA")
    rgb = img.convert("RGB")
    if light == "left":
        rgb = ImageEnhance.Brightness(rgb).enhance(1.04)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.05)
    elif light == "right":
        rgb = ImageEnhance.Brightness(rgb).enhance(1.02)
        rgb = ImageEnhance.Color(rgb).enhance(1.05)
    else:
        rgb = ImageEnhance.Contrast(rgb).enhance(1.04)
    out = Image.new("RGBA", img.size)
    out.paste(rgb, (0, 0))
    out.putalpha(img.split()[-1])
    return out


def make_contact_shadow(
    cutout: Image.Image,
    canvas_size: Tuple[int, int],
    xy: Tuple[int, int],
    opacity: int = 130,
) -> Image.Image:
    """Тень: силуэт + эллипс у низа — кукла «стоит» в сцене."""
    w, h = canvas_size
    alpha = cutout.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x, y = xy
    # 1) мягкая тень-силуэт со сдвигом
    sil = Image.new("RGBA", cutout.size, (0, 0, 0, 0))
    sil.putalpha(alpha.point(lambda v: int(v * 0.45) if v > 0 else 0))
    sil = sil.filter(ImageFilter.GaussianBlur(8))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    layer.paste(sil, (x + 10, y + 14), sil)
    # 2) контактный эллипс
    cx = x + (bbox[0] + bbox[2]) // 2
    cy = y + int(bbox[3] * 0.97)
    rw = max(48, int((bbox[2] - bbox[0]) * 0.48))
    rh = max(14, int((bbox[3] - bbox[1]) * 0.07))
    ell = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(ell).ellipse(
        [cx - rw, cy - rh, cx + rw, cy + rh], fill=(0, 0, 0, opacity)
    )
    ell = ell.filter(ImageFilter.GaussianBlur(max(8, rw // 7)))
    return Image.alpha_composite(layer, ell)


def match_to_background(cutout: Image.Image, background: Image.Image, strength: float = 0.18) -> Image.Image:
    """Подтянуть цвет куклы к тону фона (лёгкий color grade)."""
    bg = background.convert("RGB").resize((64, 64), Image.BOX)
    pixels = list(bg.getdata())
    br = sum(p[0] for p in pixels) / len(pixels)
    bg_ = sum(p[1] for p in pixels) / len(pixels)
    bb = sum(p[2] for p in pixels) / len(pixels)
    # целевой тёплый оттенок сцены
    tint = Image.new("RGB", cutout.size, (int(br), int(bg_), int(bb)))
    rgb = cutout.convert("RGB")
    blended = Image.blend(rgb, tint, strength)
    # чуть снизить контраст «наклейки»
    blended = ImageEnhance.Contrast(blended).enhance(0.96)
    blended = ImageEnhance.Color(blended).enhance(0.95)
    out = Image.new("RGBA", cutout.size)
    out.paste(blended, (0, 0))
    out.putalpha(cutout.split()[-1])
    return out


def place_character(
    background: Image.Image,
    pose: Pose,
    scale: Optional[float] = None,
) -> Image.Image:
    bg = background.convert("RGBA")
    W, H = bg.size
    cut = Image.open(pose.path).convert("RGBA")
    cut = trim_transparent(cut, pad=2)
    r, g, b, a = cut.split()
    a = soft_mask(a, feather=0.8)
    cut = Image.merge("RGBA", (r, g, b, a))
    cut = match_lighting(cut, pose.light)
    cut = match_to_background(cut, background, strength=0.08)

    sc = scale if scale is not None else pose.scale
    target_h = int(H * sc)
    ratio = target_h / cut.height
    target_w = max(1, int(cut.width * ratio))
    cut = cut.resize((target_w, target_h), Image.LANCZOS)
    # мягкий низ — не «отрезанная» линия
    r, g, b, a = cut.split()
    fade = Image.new("L", (target_w, target_h), 255)
    fd = ImageDraw.Draw(fade)
    band = max(18, target_h // 11)
    for i in range(band):
        alpha_v = int(255 * (i / band) ** 1.4)
        fd.line([(0, target_h - band + i), (target_w, target_h - band + i)], fill=alpha_v)
    a = ImageChops.multiply(a, fade)
    cut = Image.merge("RGBA", (r, g, b, a))

    if pose.slot == "left":
        x = int(W * 0.02)
    elif pose.slot == "right":
        x = int(W * 0.98) - target_w
    else:
        x = (W - target_w) // 2
    y = int(H * pose.y_bias) - target_h
    y = max(int(H * 0.05), min(y, H - target_h - int(H * 0.02)))
    x = max(0, min(x, W - target_w))

    shadow = make_contact_shadow(cut, (W, H), (x, y))
    comp = Image.alpha_composite(bg, shadow)
    comp.paste(cut, (x, y), cut)
    comp = grade_final(comp)
    return comp.convert("RGB")


def grade_final(img: Image.Image) -> Image.Image:
    """Единый грейд: чуть теплее, лёгкий виньет — фон и кукла «одной сцены»."""
    rgba = img.convert("RGBA")
    rgb = rgba.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(1.06)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.04)
    # тёплый оверлей
    warm = Image.new("RGB", rgb.size, (255, 220, 180))
    rgb = Image.blend(rgb, warm, 0.04)
    # vignette
    w, h = rgb.size
    vig = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse(
        [-int(w * 0.1), -int(h * 0.15), int(w * 1.1), int(h * 1.2)],
        fill=255,
    )
    vig = vig.filter(ImageFilter.GaussianBlur(max(w, h) // 8))
    dark = Image.new("RGB", (w, h), (20, 15, 10))
    rgb = Image.composite(rgb, Image.blend(rgb, dark, 0.35), vig)
    out = Image.new("RGBA", (w, h))
    out.paste(rgb, (0, 0))
    out.putalpha(rgba.split()[-1] if rgba.mode == "RGBA" else Image.new("L", (w, h), 255))
    return out


def draw_greeting(image: Image.Image, text: str = "Happy Birthday!") -> Image.Image:
    img = image.convert("RGBA")
    w, h = img.size
    font_size = int(h * 0.085)
    margin = int(w * 0.06)
    max_w = w - 2 * margin
    font = None
    while font_size > 12:
        font = ImageFont.truetype(FONT_BOLD, font_size)
        bbox = ImageDraw.Draw(img).textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            break
        font_size -= 2
    assert font is not None
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (w - tw) / 2 - bbox[0]
    y = int(h * 0.035) - bbox[1]
    pad_x, pad_y = int(font_size * 0.42), int(font_size * 0.26)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    rect = [
        x + bbox[0] - pad_x,
        y + bbox[1] - pad_y,
        x + bbox[2] + pad_x,
        y + bbox[3] + pad_y,
    ]
    od.rounded_rectangle(rect, radius=max(8, font_size // 5), fill=(20, 12, 8, 165))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 225, 80),
        stroke_width=max(2, font_size // 15),
        stroke_fill=(60, 20, 0),
    )
    return img.convert("RGB")


# Только фон: торт + праздник. Без puppet/felt/lamp — иначе модель рисует кукол и лампы.
SCENE_PROMPT_VARIANTS = {
    "cake": [
        "pink birthday cake with lit candles on wooden table right, red curtains, colorful balloons, confetti, warm party room, empty left side, no people",
        "frosted birthday cake candles on right, streamers balloons gift wrap, velvet curtains, festive warm light, clear left, no people",
        "two tier birthday cake with candles right, paper garlands balloons, checkered tablecloth, bright party room, empty left, no people",
        "chocolate birthday cake lit candles right, hanging balloons confetti, wood table red drapes, festive atmosphere, clear left, no people",
        "white cream birthday cake candles right, burgundy curtains balloons, party decorations wood floor, empty left side, no people",
    ],
    "party": [
        "birthday cake snacks on right table, paper bunting balloons streamers, warm festive room, empty left, no people",
        "cake cookies punch bowl right, colorful flags confetti wallpaper, party lights glow, clear left, no people",
        "presents and birthday cake right, hanging balloons streamers, festive warm room, empty left, no people",
    ],
    "family": [
        "birthday cake on right table, lace curtains balloons, cozy festive room wood floor, empty left, no people",
        "birthday cake near window right, floral wallpaper balloons, warm golden light, clear left, no people",
        "layered cake candles gifts right, patterned wallpaper bunting balloons, festive evening, empty left, no people",
    ],
}


@dataclass
class Framing:
    """Ракурс сцены под вставку выреза — не ломает рабочую схему «пусто слева / торт справа»."""
    id: str
    scrub_frac: float  # сколько слева затереть под персонажа
    cake_x: float      # 0..1 центр торта в макете
    scale_mul: float   # множитель к pose.scale
    prompt_extra: str = ""


# Рабочий формат первым; остальные — вариации того же слота.
FRAMINGS = [
    Framing("cake_right", 0.45, 0.74, 1.00, ""),
    Framing("cake_far_right", 0.48, 0.82, 0.95, ", cake farther right"),
    Framing("wide_left", 0.52, 0.78, 0.88, ", wide empty left floor"),
    Framing("close_left", 0.40, 0.70, 1.08, ", character space larger left"),
    Framing("party_balloons", 0.45, 0.72, 1.00, ", many colorful balloons overhead"),
    Framing("window_light", 0.46, 0.76, 1.00, ", soft window light from right"),
]


def pick_framing(
    rng: random.Random | None = None,
    framing_id: str | None = None,
) -> Framing:
    rng = rng or random
    if framing_id:
        for f in FRAMINGS:
            if f.id == framing_id:
                return f
        known = ", ".join(f.id for f in FRAMINGS)
        raise ValueError(f"неизвестный framing={framing_id!r}; доступны: {known}")
    # чаще рабочий кадр cake_right (~43%)
    weights = [6, 2, 2, 2, 2, 2]
    return rng.choices(FRAMINGS, weights=weights, k=1)[0]


def scene_prompt(
    theme: str,
    pose: Pose,
    rng: random.Random | None = None,
    framing: Framing | None = None,
) -> str:
    rng = rng or random
    base = rng.choice(SCENE_PROMPT_VARIANTS.get(theme, SCENE_PROMPT_VARIANTS["cake"]))
    if framing and framing.prompt_extra:
        # держим короткий промпт — CLIP ~77
        extra = framing.prompt_extra.strip(", ")
        if len(base.split()) < 28:
            base = f"{base}, {extra}"
    return base


def scrub_left_background(bg: Image.Image, frac: float = 0.45) -> Image.Image:
    """Стереть левую часть фона — убрать сгенерированных кукол/фигур перед вставкой выреза."""
    img = bg.convert("RGB")
    w, h = img.size
    cut = int(w * frac)
    wall = img.crop((cut, int(h * 0.15), min(w - 8, cut + 48), int(h * 0.45)))
    floor = img.crop((cut, int(h * 0.70), min(w - 8, cut + 48), int(h * 0.92)))
    wall_c = wall.resize((1, 1), Image.BOX).getpixel((0, 0))
    floor_c = floor.resize((1, 1), Image.BOX).getpixel((0, 0))
    fy = int(h * 0.62)
    left = Image.new("RGB", (cut + 24, h), wall_c)
    ld = ImageDraw.Draw(left)
    ld.rectangle([0, fy, cut + 24, h], fill=floor_c)
    left = left.filter(ImageFilter.GaussianBlur(6))
    out = img.copy()
    mask = Image.new("L", (cut + 24, h), 255)
    md = ImageDraw.Draw(mask)
    for i in range(28):
        md.rectangle([cut + 24 - 28 + i, 0, cut + 24 - 28 + i, h], fill=int(255 * (1 - i / 28)))
    out.paste(left, (0, 0), mask)
    return out


NEGATIVE_BG = (
    "person, people, face, human, character, figure, puppet, doll, "
    "lamp, floor lamp, chandelier, light bulb, glowing orb, pole, "
    "text, watermark, blurry, muddy, horror, ugly, deformed"
)


def make_layout_canvas(
    size: int,
    theme: str,
    rng: random.Random | None = None,
    framing: Framing | None = None,
) -> Image.Image:
    """Мягкий цветовой макет: торт справа, пусто слева."""
    rng = rng or random.Random(0)
    framing = framing or FRAMINGS[0]
    wall = rng.choice([
        (92, 110, 140), (110, 95, 80), (70, 100, 90), (100, 85, 105),
        (85, 95, 115), (120, 100, 85),
    ])
    curtain = rng.choice([
        (140, 35, 45), (40, 75, 130), (45, 110, 65), (150, 95, 40),
        (110, 40, 95), (160, 55, 50),
    ])
    floor = rng.choice([
        (155, 115, 75), (130, 95, 65), (170, 135, 95), (120, 90, 60),
    ])
    table = rng.choice([
        (190, 145, 100), (170, 125, 85), (200, 165, 120),
    ])
    frosting = rng.choice([
        (255, 175, 175), (255, 230, 200), (230, 200, 170), (255, 210, 230),
        (245, 235, 220), (200, 150, 120),
    ])

    canvas = Image.new("RGB", (size, size), wall)
    draw = ImageDraw.Draw(canvas)

    ceil = tuple(max(0, c - 25) for c in wall)
    draw.rectangle([0, 0, size, int(size * 0.18)], fill=ceil)

    ch = int(size * rng.uniform(0.28, 0.40))
    for x in range(0, size, max(10, size // 18)):
        shade = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in curtain)
        draw.rectangle([x, 0, x + size // 20, ch], fill=shade)

    fy = int(size * 0.62)
    draw.rectangle([0, fy, size, size], fill=floor)
    for i in range(6):
        y = fy + int((size - fy) * (i / 6))
        line = tuple(max(0, c - 18) for c in floor)
        draw.line([(0, y), (size, y)], fill=line, width=2)

    tx = int(size * rng.uniform(0.55, 0.68))
    ty = int(size * rng.uniform(0.58, 0.64))
    tw = int(size * rng.uniform(0.38, 0.48))
    th = int(size * rng.uniform(0.12, 0.16))
    draw.ellipse([tx, ty, tx + tw, ty + th], fill=table)

    balloon_n = rng.randint(5, 14)
    if framing.id == "party_balloons":
        balloon_n = rng.randint(12, 20)
    for _ in range(balloon_n):
        x = rng.randint(int(size * 0.35), size - 35)
        y = rng.randint(8, max(20, ch - 10))
        r = rng.randint(16, 34)
        col = rng.choice([
            (255, 90, 130), (80, 200, 255), (255, 220, 60),
            (120, 255, 140), (255, 150, 60), (200, 120, 255),
        ])
        draw.ellipse([x, y, x + r, y + r], fill=col)

    cx = int(size * framing.cake_x)
    cy = int(size * 0.70)
    if theme in ("cake", "family") or (theme == "party" and rng.random() < 0.7):
        cw = int(size * rng.uniform(0.12, 0.16))
        ch_cake = int(size * rng.uniform(0.16, 0.22))
        draw.ellipse([cx - cw - 10, cy - 6, cx + cw + 10, cy + 28], fill=(250, 240, 220))
        draw.rectangle([cx - cw, cy - ch_cake, cx + cw, cy], fill=frosting)
        mw = int(cw * 0.78)
        mh = int(ch_cake * 0.45)
        mid = tuple(max(0, c - 20) for c in frosting)
        draw.rectangle([cx - mw, cy - ch_cake - mh, cx + mw, cy - ch_cake + 4], fill=mid)
        top = tuple(max(0, c - 35) for c in frosting)
        draw.ellipse(
            [cx - mw, cy - ch_cake - mh - 16, cx + mw, cy - ch_cake - mh + 10],
            fill=top,
        )
        for i in range(5):
            bx = cx - mw + 10 + i * max(12, (2 * mw - 16) // 4)
            draw.rectangle(
                [bx - 2, cy - ch_cake - mh - 42, bx + 2, cy - ch_cake - mh - 8],
                fill=(255, 245, 210),
            )
            draw.ellipse(
                [bx - 6, cy - ch_cake - mh - 54, bx + 6, cy - ch_cake - mh - 40],
                fill=(255, 170, 50),
            )
    else:
        for _ in range(3):
            gx = rng.randint(tx + 10, tx + tw - 40)
            gy = rng.randint(ty - 30, ty - 5)
            draw.rectangle(
                [gx, gy, gx + 36, gy + 28],
                fill=rng.choice([(200, 60, 60), (60, 140, 200), (220, 180, 60)]),
            )

    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(8, size // 40)))
    return canvas
