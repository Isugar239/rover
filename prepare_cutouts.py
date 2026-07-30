#!/usr/bin/env python3
"""
Готовит вырезы Шапокляк (прозрачный PNG) через rembg u2netp.

  python3 prepare_cutouts.py
"""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets", "poses")
OUT_JSON = os.path.join(ASSETS, "poses.json")

JOBS = [
    {
        "id": "left_look_right",
        "src": os.path.join(ROOT, "refs", "images3.jpeg"),
        "crop": (0.0, 0.0, 0.68, 1.0),
        "look": "right",
        "slot": "left",
        "scale": 0.80,
        "y_bias": 0.99,
        "light": "left",
        "notes": "Профиль вправо — торт справа",
    },
    {
        "id": "front_gesture",
        "src": os.path.join(ROOT, "refs", "images1.jpeg"),
        "crop": (0.08, 0.0, 0.78, 0.95),
        "look": "front",
        "slot": "left",
        "scale": 0.84,
        "y_bias": 1.0,
        "light": "right",
        "notes": "Жест рукой",
    },
    {
        "id": "waist_up_cake",
        "src": os.path.join(ROOT, "итог", "ok_904.png"),
        "crop": (0.0, 0.14, 0.55, 0.95),
        "look": "right",
        "slot": "left",
        "scale": 0.76,
        "y_bias": 0.98,
        "light": "left",
        "notes": "Смотрит на торт",
    },
]


def trim(im: Image.Image, pad: int = 6) -> Image.Image:
    bb = im.split()[-1].getbbox()
    if not bb:
        return im
    l, t, r, b = bb
    return im.crop(
        (max(0, l - pad), max(0, t - pad), min(im.width, r + pad), min(im.height, b + pad))
    )


def preview(cut: Image.Image, path: str):
    tile = 14
    prev = Image.new("RGBA", cut.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(prev)
    for y in range(0, cut.height, tile):
        for x in range(0, cut.width, tile):
            c = (220, 220, 220, 255) if ((x // tile) + (y // tile)) % 2 == 0 else (170, 170, 170, 255)
            d.rectangle([x, y, x + tile, y + tile], fill=c)
    Image.alpha_composite(prev, cut).convert("RGB").save(path, quality=92)


def kill_blue_fringe(im: Image.Image) -> Image.Image:
    import numpy as np

    arr = np.asarray(im).astype(np.float32)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    blue = (b > r + 15) & (b > g + 5) & (b > 70) & (a > 0)
    a = np.where(blue, 0, a)
    a_img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
    a_img = a_img.filter(ImageFilter.MinFilter(3))
    a_img = a_img.filter(ImageFilter.GaussianBlur(0.8))
    out = im.copy()
    out.putalpha(a_img)
    return out


def main():
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(ROOT, ".numba_cache"))
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    from rembg import remove, new_session

    session = new_session("u2netp")
    os.makedirs(ASSETS, exist_ok=True)
    poses = []
    for job in JOBS:
        src = job["src"]
        if not os.path.isfile(src):
            print(f"[!] нет {src}")
            continue
        print(f"[*] {job['id']}")
        im = Image.open(src).convert("RGB")
        w, h = im.size
        l, t, r, b = job["crop"]
        im = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
        if max(im.size) < 900:
            s = 900 / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        out = remove(im, session=session)
        if not isinstance(out, Image.Image):
            import io

            out = Image.open(io.BytesIO(out)).convert("RGBA")
        out = out.convert("RGBA")
        r, g, b, a = out.split()
        a = a.point(lambda v: 0 if v < 15 else (255 if v > 200 else v))
        a = a.filter(ImageFilter.MedianFilter(3))
        a = a.filter(ImageFilter.GaussianBlur(0.7))
        out = Image.merge("RGBA", (r, g, b, a))
        out = kill_blue_fringe(out)
        out = trim(out)
        path = os.path.join(ASSETS, f"{job['id']}.png")
        out.save(path)
        preview(out, os.path.join(ASSETS, f"{job['id']}_preview.jpg"))
        print(f"[+] {path} {out.size}")
        poses.append(
            {
                "id": job["id"],
                "file": f"{job['id']}.png",
                "look": job["look"],
                "slot": job["slot"],
                "scale": job["scale"],
                "y_bias": job["y_bias"],
                "light": job["light"],
                "notes": job["notes"],
            }
        )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"version": 4, "poses": poses}, f, ensure_ascii=False, indent=2)
    print(f"[+] {OUT_JSON} ({len(poses)} поз)")


if __name__ == "__main__":
    main()
