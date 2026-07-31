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


# Комната + торт на столе справа (один проход). Короткие промпты — CLIP 77.
SCENE_PROMPT_VARIANTS = {
    "cake": [
        "pink birthday cake lit candles on wood table right, cozy room red curtains window balloons, empty left, no people",
        "white cream birthday cake lit candles on table right, home room wallpaper drapes balloons, empty left, no people",
        "chocolate birthday cake lit candles on table right, kitchen window chairs balloons, clear left, no people",
        "yellow birthday cake lit candles on table right, apartment curtains balloons, empty left, no people",
        "vanilla birthday cake lit candles on rustic table right, cottage lace curtains window, clear left, no people",
    ],
    "party": [
        "birthday cake lit candles on table right, festive room balloons bunting, empty left, no people",
        "birthday cake lit candles gifts on table right, party flags balloons, clear left, no people",
        "birthday cake lit candles on table right, decorated home streamers, empty left, no people",
    ],
    "family": [
        "birthday cake lit candles on wood table right, cozy room lace curtains balloons, empty left, no people",
        "birthday cake lit candles on table by window right, dining room wallpaper, clear left, no people",
        "birthday cake lit candles gifts on table right, evening apartment bunting, empty left, no people",
    ],
}

# Запасные промпты, если снова включим отдельную генерацию торта.
CAKE_PROMPT_VARIANTS = [
    "homemade two-tier birthday cake, buttercream frosting, five lit candles, white plate, gray background, photo",
    "homemade pink two-tier birthday cake, buttercream, five lit candles, white plate, gray background, photo",
    "homemade chocolate two-tier birthday cake, buttercream, five lit candles, white plate, gray background, photo",
    "homemade vanilla two-tier birthday cake, buttercream, five lit candles, white plate, gray background, photo",
    "homemade yellow two-tier birthday cake, buttercream, five lit candles, white plate, gray background, photo",
]

NEGATIVE_CAKE = (
    "wedding cake, fondant, roses, plastic, melted, goo, horror, creepy, "
    "face, blood, glowing, transparent, four tiers, lightbulbs, "
    "person, room, furniture, text, blurry, multiple cakes, deformed"
)



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
    Framing("cake_right", 0.38, 0.72, 1.00, ""),
    Framing("cake_far_right", 0.40, 0.80, 0.92, ", table farther right"),
    Framing("wide_left", 0.44, 0.76, 0.88, ", wide empty left floor space"),
    Framing("close_left", 0.36, 0.68, 1.05, ", closer view of table"),
    Framing("party_balloons", 0.38, 0.70, 1.00, ", many balloons and garlands"),
    Framing("window_light", 0.38, 0.74, 1.00, ", sunny window behind table"),
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
    # чаще комната+стол справа: cake_right / window_light
    weights = [8, 1, 1, 1, 2, 5]
    return rng.choices(FRAMINGS, weights=weights, k=1)[0]


def pick_cake_prompt(rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(CAKE_PROMPT_VARIANTS)


def make_cake_solo_layout(size: int, rng: random.Random | None = None) -> Image.Image:
    """Чёткий макет торта на сером фоне — форма держится при img2img."""
    rng = rng or random.Random(0)
    # ровный серый — удобно вырезать и не путается с кремом
    bg = (210, 212, 218)
    canvas = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(canvas)
    frost = rng.choice([
        (255, 140, 165), (255, 200, 120), (240, 210, 180),
        (255, 170, 190), (140, 85, 55), (255, 230, 140),
    ])
    frost2 = tuple(max(40, min(255, c - 20)) for c in frost)
    cx, cy = size // 2, int(size * 0.68)
    cw, ch = int(size * 0.30), int(size * 0.38)

    # тарелка
    draw.ellipse([cx - cw - 24, cy - 6, cx + cw + 24, cy + 32], fill=(250, 250, 252))
    draw.ellipse([cx - cw - 14, cy + 2, cx + cw + 14, cy + 22], fill=(235, 235, 240))

    # нижний ярус (цилиндр)
    y1 = cy
    h1 = int(ch * 0.48)
    draw.rounded_rectangle([cx - cw, y1 - h1, cx + cw, y1], radius=8, fill=frost)
    draw.ellipse([cx - cw, y1 - 14, cx + cw, y1 + 10], fill=tuple(min(255, c + 20) for c in frost))
    for i in range(8):
        bx = cx - cw + 8 + i * max(8, (2 * cw - 16) // 7)
        draw.ellipse([bx - 4, y1 - 8, bx + 4, y1 + 4], fill=(255, 255, 255))

    # верхний ярус
    mw = int(cw * 0.70)
    y2 = y1 - h1 + 4
    h2 = int(ch * 0.36)
    draw.rounded_rectangle([cx - mw, y2 - h2, cx + mw, y2], radius=7, fill=frost2)
    draw.ellipse([cx - mw, y2 - 12, cx + mw, y2 + 8], fill=(255, 255, 255))
    for i in range(6):
        bx = cx - mw + 6 + i * max(8, (2 * mw - 12) // 5)
        draw.ellipse([bx - 3, y2 - 6, bx + 3, y2 + 2], fill=(255, 255, 255))

    top = y2 - h2
    draw.ellipse([cx - mw + 4, top - 8, cx + mw - 4, top + 10], fill=(255, 255, 255))
    draw.ellipse([cx - mw + 10, top - 2, cx + mw - 10, top + 8], fill=frost2)

    # ровные свечи
    for i in range(5):
        bx = cx - mw + 14 + i * max(10, (2 * mw - 28) // 4)
        draw.rectangle([bx - 2, top - 34, bx + 2, top + 2], fill=(255, 250, 230))
        draw.ellipse([bx - 5, top - 46, bx + 5, top - 32], fill=(255, 140, 30))
        draw.ellipse([bx - 3, top - 50, bx + 3, top - 38], fill=(255, 245, 130))

    # почти без блюра — форма важнее
    return canvas.filter(ImageFilter.GaussianBlur(radius=1.2))


def cake_looks_ok(img: Image.Image) -> bool:
    """Отсечь светящиеся/пустые/страшные торты."""
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    crop = arr[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5]
    mean = float(crop.mean())
    std = float(crop.std())
    # слишком белый/светящийся или почти плоский
    if mean > 225 or mean < 40:
        return False
    if std < 18:
        return False
    return True


def cutout_cake_rgba(img: Image.Image) -> Image.Image:
    """Вырезать торт с серого фона; жёстче к краям, меньше «каши»."""
    import numpy as np

    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    h, w, _ = arr.shape
    # фон: верхняя полоса + углы (там не должно быть торта)
    top_band = arr[: max(8, h // 10), :, :].reshape(-1, 3)
    corners = np.stack([
        arr[2, 2], arr[2, w - 3], arr[h - 3, 2], arr[h - 3, w - 3],
    ])
    bg = np.concatenate([top_band, corners], axis=0).mean(axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    alpha = np.clip((dist - 22) / 24, 0, 1)
    alpha = (alpha * 255).astype(np.uint8)
    a_img = Image.fromarray(alpha, mode="L")
    a_img = a_img.filter(ImageFilter.MedianFilter(5))
    a_img = a_img.point(lambda p: 255 if p > 100 else (0 if p < 50 else int((p - 50) * 255 / 50)))
    a_img = a_img.filter(ImageFilter.GaussianBlur(0.8))
    out = rgb.convert("RGBA")
    out.putalpha(a_img)
    return trim_transparent(out, pad=4)


def paste_cake_rgba(
    bg: Image.Image,
    cake: Image.Image,
    framing: Framing | None = None,
    scale: float = 0.38,
) -> Image.Image:
    """Вставить вырезанный AI-торт справа на стол."""
    framing = framing or FRAMINGS[0]
    base = bg.convert("RGBA")
    w, h = base.size
    cake = cake.convert("RGBA")
    tw = max(32, int(w * scale))
    th = int(tw * cake.size[1] / max(1, cake.size[0]))
    cake = cake.resize((tw, th), Image.LANCZOS)
    cx = int(w * framing.cake_x)
    # низ торта на линии стола
    x = cx - tw // 2
    y = int(h * 0.64) - th + int(th * 0.08)
    y = max(0, min(h - th, y))

    # всегда стол под тортом — чтобы не «летал»
    table = Image.new("RGBA", base.size, (0, 0, 0, 0))
    td = ImageDraw.Draw(table)
    wood = (168, 120, 75, 245)
    wood2 = (140, 95, 55, 245)
    tx0, ty0 = x - 18, y + th - 18
    tx1, ty1 = x + tw + 18, y + th + 28
    td.rounded_rectangle([tx0, ty0, tx1, ty1], radius=6, fill=wood)
    td.rectangle([tx0 + 10, ty1 - 8, tx0 + 22, min(h - 4, ty1 + 40)], fill=wood2)
    td.rectangle([tx1 - 22, ty1 - 8, tx1 - 10, min(h - 4, ty1 + 40)], fill=wood2)
    td.ellipse([tx0 + 6, ty0 - 4, tx1 - 6, ty0 + 16], fill=(245, 240, 230, 230))
    base = Image.alpha_composite(base, table)

    # тень
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse(
        [x + 8, y + th - 14, x + tw - 8, y + th + 8],
        fill=(0, 0, 0, 75),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))
    out = Image.alpha_composite(base, shadow)
    out.paste(cake, (x, y), cake)
    return out.convert("RGB")


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


def paste_birthday_cake(
    bg: Image.Image,
    framing: Framing | None = None,
    rng: random.Random | None = None,
) -> Image.Image:
    """Гарантированно вставить читаемый торт справа на стол — не надеемся на SD."""
    rng = rng or random
    framing = framing or FRAMINGS[0]
    img = bg.convert("RGBA")
    w, h = img.size
    cx = int(w * framing.cake_x)
    base_y = int(h * 0.60)

    frost = rng.choice([
        (255, 165, 180),
        (255, 225, 200),
        (250, 240, 225),
        (255, 195, 215),
        (200, 145, 115),
    ])
    frost2 = tuple(max(0, c - 28) for c in frost)
    plate_c = (252, 248, 242)
    cream = (255, 255, 255)

    cake_h = int(h * 0.24)
    cake_w = int(w * 0.14)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # подставка
    d.ellipse(
        [cx - 14, base_y + 8, cx + 14, base_y + 22],
        fill=(170, 150, 120, 255),
    )
    d.rectangle([cx - 6, base_y, cx + 6, base_y + 14], fill=(190, 170, 140, 255))
    d.ellipse(
        [cx - cake_w - 16, base_y - 8, cx + cake_w + 16, base_y + 14],
        fill=(*plate_c, 255),
    )
    d.ellipse(
        [cx - cake_w - 10, base_y - 4, cx + cake_w + 10, base_y + 8],
        fill=(235, 228, 218, 255),
    )

    def tier(x0, y0, x1, y1, col, rim=True):
        shade = tuple(max(0, c - 35) for c in col)
        light = tuple(min(255, c + 25) for c in col)
        d.rounded_rectangle([x0, y0, x1, y1], radius=7, fill=(*col, 255))
        # боковой градиент-намёк
        mid = (x0 + x1) // 2
        d.rectangle([x0, y0, mid - 2, y1], fill=(*light, 90))
        d.rectangle([mid + 2, y0, x1, y1], fill=(*shade, 70))
        if rim:
            d.ellipse([x0 + 2, y0 - 6, x1 - 2, y0 + 10], fill=(*cream, 255))
            d.ellipse([x0 + 6, y0 - 2, x1 - 6, y0 + 8], fill=(*col, 255))

    # нижний ярус
    y1 = base_y - 2
    h1 = int(cake_h * 0.46)
    tier(cx - cake_w, y1 - h1, cx + cake_w, y1, frost)
    # крем-точки по низу
    for i in range(7):
        bx = cx - cake_w + 6 + i * max(9, (2 * cake_w - 12) // 6)
        d.ellipse([bx - 5, y1 - 10, bx + 5, y1 + 4], fill=(*cream, 255))
        d.ellipse([bx - 3, y1 - 8, bx + 3, y1], fill=(*frost, 255))

    # верхний ярус
    mw = int(cake_w * 0.70)
    y2 = y1 - h1 + 2
    h2 = int(cake_h * 0.36)
    tier(cx - mw, y2 - h2, cx + mw, y2 + 2, frost2)
    for i in range(5):
        bx = cx - mw + 6 + i * max(8, (2 * mw - 12) // 4)
        d.ellipse([bx - 4, y2 - 8, bx + 4, y2 + 3], fill=(*cream, 255))

    # верхняя шапка крема
    top = y2 - h2
    d.ellipse([cx - mw + 2, top - 8, cx + mw - 2, top + 10], fill=(*cream, 255))
    d.ellipse([cx - mw + 8, top - 4, cx + mw - 8, top + 8], fill=(*frost2, 255))

    # свечи
    n_candles = 5
    for i in range(n_candles):
        bx = cx - mw + 12 + i * max(9, (2 * mw - 24) // (n_candles - 1))
        wax = rng.choice([(255, 250, 230), (255, 220, 220), (220, 240, 255)])
        d.rectangle([bx - 2, top - 30, bx + 2, top + 2], fill=(*wax, 255))
        # ореол
        d.ellipse([bx - 8, top - 48, bx + 8, top - 28], fill=(255, 200, 80, 55))
        d.ellipse([bx - 5, top - 42, bx + 5, top - 28], fill=(255, 140, 30, 255))
        d.ellipse([bx - 3, top - 46, bx + 3, top - 34], fill=(255, 245, 140, 255))

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse(
        [cx - cake_w - 6, base_y + 2, cx + cake_w + 6, base_y + 18],
        fill=(0, 0, 0, 80),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))
    out = Image.alpha_composite(img, shadow)
    out = Image.alpha_composite(out, layer)
    return out.convert("RGB")

def scrub_left_background(bg: Image.Image, frac: float = 0.38) -> Image.Image:
    """Слегка затереть возможные фигуры слева, сохранив обои/мебель комнаты."""
    img = bg.convert("RGB")
    w, h = img.size
    cut = int(w * frac)
    y0, y1 = int(h * 0.18), int(h * 0.82)
    sample = img.crop((cut, y0, min(w - 2, cut + 90), y1))
    fill = sample.resize((cut + 24, y1 - y0), Image.BILINEAR)
    fill = fill.filter(ImageFilter.GaussianBlur(8))
    orig_blur = img.crop((0, y0, cut + 24, y1)).filter(ImageFilter.GaussianBlur(14))
    fill = Image.blend(fill, orig_blur, 0.55)

    out = img.copy()
    mask = Image.new("L", (cut + 24, y1 - y0), 0)
    md = ImageDraw.Draw(mask)
    md.rectangle([0, 16, cut, y1 - y0 - 16], fill=160)
    mask = mask.filter(ImageFilter.GaussianBlur(20))
    out.paste(fill, (0, y0), mask)
    return out


NEGATIVE_BG = (
    "person, people, face, human, character, figure, puppet, doll, "
    "lamp, floor lamp, chandelier, light bulb, glowing orb, pole, "
    "solid color background, blank wall, empty void, flat color, abstract, "
    "melted cake, horror cake, creepy, "
    "bokeh, out of focus, blurry background, shallow depth of field, "
    "text, watermark, muddy, horror, ugly, deformed"
)


def make_layout_canvas(
    size: int,
    theme: str,
    rng: random.Random | None = None,
    framing: Framing | None = None,
) -> Image.Image:
    """Макет комнаты: стена/окно/стол/торт справа, пустой пол слева."""
    rng = rng or random.Random(0)
    framing = framing or FRAMINGS[0]
    wall = rng.choice([
        (168, 150, 130), (145, 155, 165), (160, 140, 125), (150, 145, 155),
        (175, 160, 140), (140, 150, 145),
    ])
    wallpaper_a = tuple(max(0, min(255, c + rng.randint(-18, 12))) for c in wall)
    wallpaper_b = tuple(max(0, min(255, c + rng.randint(-8, 22))) for c in wall)
    curtain = rng.choice([
        (150, 40, 50), (45, 80, 140), (50, 115, 70), (155, 100, 45),
        (120, 45, 100), (165, 60, 55),
    ])
    floor = rng.choice([
        (170, 125, 80), (145, 105, 70), (185, 145, 100), (130, 95, 60),
    ])
    wood = rng.choice([
        (150, 100, 60), (130, 85, 50), (165, 115, 75), (120, 80, 45),
    ])
    tablecloth = rng.choice([
        (245, 235, 220), (220, 60, 70), (240, 240, 245), (90, 130, 90),
        (255, 220, 180), (200, 50, 60),
    ])
    frosting = rng.choice([
        (255, 175, 175), (255, 230, 200), (230, 200, 170), (255, 210, 230),
        (245, 235, 220), (200, 150, 120),
    ])

    canvas = Image.new("RGB", (size, size), wall)
    draw = ImageDraw.Draw(canvas)

    # потолок
    ceil = tuple(max(0, c - 30) for c in wall)
    ceil_y = int(size * 0.14)
    draw.rectangle([0, 0, size, ceil_y], fill=ceil)

    # обои / полосы на стене (даёт модели «комнату», а не плакат)
    pattern = rng.choice(["stripes", "dots", "blocks"])
    if pattern == "stripes":
        step = max(10, size // 22)
        for x in range(0, size, step):
            col = wallpaper_a if (x // step) % 2 == 0 else wallpaper_b
            draw.rectangle([x, ceil_y, x + step // 2, int(size * 0.62)], fill=col)
    elif pattern == "dots":
        for _ in range(80):
            x = rng.randint(0, size)
            y = rng.randint(ceil_y, int(size * 0.58))
            r = rng.randint(3, 8)
            draw.ellipse([x, y, x + r, y + r], fill=wallpaper_b)
    else:
        cell = max(18, size // 16)
        for y in range(ceil_y, int(size * 0.62), cell):
            for x in range(0, size, cell):
                if (x // cell + y // cell) % 2 == 0:
                    draw.rectangle([x, y, x + cell - 2, y + cell - 2], fill=wallpaper_a)

    # окно справа за столом
    wx0 = int(size * rng.uniform(0.58, 0.66))
    wy0 = int(size * 0.18)
    ww, wh = int(size * 0.28), int(size * 0.32)
    sky = rng.choice([(190, 215, 240), (255, 230, 180), (170, 200, 230)])
    draw.rectangle([wx0, wy0, wx0 + ww, wy0 + wh], fill=sky)
    draw.line([wx0 + ww // 2, wy0, wx0 + ww // 2, wy0 + wh], fill=(220, 220, 230), width=3)
    draw.line([wx0, wy0 + wh // 2, wx0 + ww, wy0 + wh // 2], fill=(220, 220, 230), width=3)
    # рама
    frame = tuple(max(0, c - 40) for c in wood)
    draw.rectangle([wx0 - 4, wy0 - 4, wx0 + ww + 4, wy0 + wh + 4], outline=frame, width=5)

    # шторы по бокам окна (не на всю ширину — иначе снова «полосы»)
    ch = int(size * rng.uniform(0.42, 0.52))
    for side, x0 in (("L", wx0 - int(size * 0.08)), ("R", wx0 + ww - 4)):
        for i in range(5):
            shade = tuple(max(0, min(255, c + rng.randint(-20, 15))) for c in curtain)
            draw.rectangle(
                [x0 + i * 7, ceil_y, x0 + i * 7 + 10, ch],
                fill=shade,
            )

    # пол + доски
    fy = int(size * 0.60)
    draw.rectangle([0, fy, size, size], fill=floor)
    plank = max(14, size // 18)
    for i, x in enumerate(range(0, size, plank)):
        shade = tuple(max(0, min(255, c + (-12 if i % 2 else 8))) for c in floor)
        draw.rectangle([x, fy, x + plank - 1, size], fill=shade)
        draw.line([(x, fy), (x, size)], fill=tuple(max(0, c - 25) for c in floor), width=1)
    # плинтус
    draw.rectangle([0, fy - 6, size, fy + 2], fill=tuple(max(0, c - 35) for c in wood))

    # коврик слева-центр (пустое место под персонажа, но не голый пол)
    rug = rng.choice([(160, 50, 55), (50, 90, 140), (60, 120, 80), (180, 140, 70)])
    draw.ellipse(
        [int(size * 0.05), int(size * 0.78), int(size * 0.48), int(size * 0.98)],
        fill=rug,
    )

    # стол справа с ножками
    tx = int(size * (framing.cake_x - 0.18))
    ty = int(size * 0.58)
    tw = int(size * rng.uniform(0.40, 0.50))
    th = int(size * 0.06)
    # ножки
    leg_c = tuple(max(0, c - 25) for c in wood)
    for lx in (tx + int(tw * 0.12), tx + int(tw * 0.78)):
        draw.rectangle([lx, ty + th, lx + int(size * 0.035), int(size * 0.88)], fill=leg_c)
    # столешница
    draw.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=6, fill=wood)
    # скатерть
    draw.ellipse(
        [tx + 8, ty - 4, tx + tw - 8, ty + th + 10],
        fill=tablecloth,
    )

    # картинная рама / полка на стене слева-центр (наполнение, не фигура)
    if rng.random() < 0.7:
        fx0 = int(size * rng.uniform(0.08, 0.22))
        fy0 = int(size * rng.uniform(0.22, 0.32))
        fw, fh = int(size * 0.14), int(size * 0.12)
        draw.rectangle([fx0, fy0, fx0 + fw, fy0 + fh], outline=frame, width=4)
        draw.rectangle(
            [fx0 + 6, fy0 + 6, fx0 + fw - 6, fy0 + fh - 6],
            fill=rng.choice([(90, 120, 150), (160, 100, 80), (100, 140, 100)]),
        )

    # стул у стола (силуэт мебели)
    if rng.random() < 0.65:
        sx = tx - int(size * 0.06)
        sy = ty + int(size * 0.02)
        draw.rectangle([sx, sy, sx + int(size * 0.08), sy + int(size * 0.04)], fill=wood)
        draw.rectangle(
            [sx + 4, sy + int(size * 0.04), sx + 12, int(size * 0.86)],
            fill=leg_c,
        )
        draw.rectangle(
            [sx + int(size * 0.06), sy - int(size * 0.12), sx + int(size * 0.08), sy + 4],
            fill=wood,
        )

    # гирлянда / флажки
    for i in range(8):
        x = int(size * (0.35 + i * 0.07))
        y = int(size * 0.16) + (i % 2) * 10
        flag = rng.choice([
            (255, 90, 130), (80, 200, 255), (255, 220, 60),
            (120, 255, 140), (255, 150, 60),
        ])
        draw.polygon([(x, y), (x + 14, y), (x + 7, y + 18)], fill=flag)

    balloon_n = rng.randint(6, 12)
    if framing.id == "party_balloons":
        balloon_n = rng.randint(12, 18)
    for _ in range(balloon_n):
        x = rng.randint(int(size * 0.40), size - 40)
        y = rng.randint(int(size * 0.08), int(size * 0.28))
        r = rng.randint(14, 28)
        col = rng.choice([
            (255, 90, 130), (80, 200, 255), (255, 220, 60),
            (120, 255, 140), (255, 150, 60), (200, 120, 255),
        ])
        draw.ellipse([x, y, x + r, y + int(r * 1.15)], fill=col)
        draw.line([x + r // 2, y + int(r * 1.15), x + r // 2, y + r + 30], fill=(80, 80, 80), width=1)

    # торт на столе справа — одна узнаваемая форма (2 яруса + свечи)
    cx = int(size * framing.cake_x)
    cy = ty - 2
    frost = rng.choice([
        (255, 150, 170), (255, 220, 160), (245, 235, 220),
        (200, 140, 100), (255, 200, 210),
    ])
    frost2 = tuple(max(40, c - 25) for c in frost)
    cw = int(size * 0.17)
    ch_cake = int(size * 0.26)
    plate = (252, 248, 240)
    draw.ellipse([cx - cw - 14, cy - 4, cx + cw + 14, cy + 16], fill=plate)
    draw.rounded_rectangle([cx - cw, cy - int(ch_cake * 0.5), cx + cw, cy], radius=6, fill=frost)
    mw = int(cw * 0.72)
    draw.rounded_rectangle(
        [cx - mw, cy - int(ch_cake * 0.85), cx + mw, cy - int(ch_cake * 0.45)],
        radius=5,
        fill=frost2,
    )
    top = cy - int(ch_cake * 0.85)
    draw.ellipse([cx - mw, top - 8, cx + mw, top + 8], fill=(255, 255, 255))
    for i in range(5):
        bx = cx - mw + 8 + i * max(9, (2 * mw - 16) // 4)
        draw.rectangle([bx - 2, top - 32, bx + 2, top + 2], fill=(255, 248, 220))
        draw.ellipse([bx - 5, top - 44, bx + 5, top - 30], fill=(255, 150, 40))
        draw.ellipse([bx - 3, top - 48, bx + 3, top - 36], fill=(255, 240, 120))

    # слабый блюр: комната и торт должны читаться
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(2, size // 128)))
    return canvas
