#!/usr/bin/env python3
"""
Праздничная открытка на FLUX.1-schnell (img2img по коллажу с рефом).

  python3 generate_flux.py
  python3 generate_flux.py --seed 42 --strength 0.65
"""
import argparse
import os
import random

import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "flux-schnell")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "итог")
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BEST_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "images3.jpeg")

PROMPT = (
    "festive birthday party scene, stop-motion puppet of old woman Shapoklyak, "
    "black top hat, long pointed nose, white lace jabot, black dress, "
    "birthday cake with lit candles, colorful balloons, confetti, "
    "cozy room with red curtains, warm cinematic lighting, detailed environment, "
    "medium shot, high quality"
)

NEGATIVE = (
    "ugly, scary, horror, deformed, melted face, photoreal human, "
    "anime, 2d cartoon, watermark, text, signature, blurry"
)


def make_festive_canvas(face_path: str, size: int, rng: random.Random) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (40, 70, 110))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, size, size // 3], fill=(90, 20, 30))
    draw.rectangle([0, int(size * 0.72), size, size], fill=(120, 90, 60))

    for _ in range(8):
        x = rng.randint(10, size - 40)
        y = rng.randint(10, size // 3)
        r = rng.randint(18, 36)
        color = rng.choice(
            [(255, 80, 120), (80, 200, 255), (255, 220, 60), (120, 255, 140), (255, 140, 60)]
        )
        draw.ellipse([x, y, x + r, y + r], fill=color)

    cx, cy = int(size * 0.72), int(size * 0.78)
    draw.ellipse([cx - 70, cy - 25, cx + 70, cy + 35], fill=(255, 230, 200))
    draw.rectangle([cx - 55, cy - 70, cx + 55, cy], fill=(255, 200, 180))
    draw.ellipse([cx - 55, cy - 85, cx + 55, cy - 55], fill=(255, 170, 160))
    for i, col in enumerate([(255, 80, 80), (80, 180, 255), (255, 220, 80)]):
        bx = cx - 30 + i * 30
        draw.rectangle([bx - 3, cy - 110, bx + 3, cy - 80], fill=(255, 240, 200))
        draw.ellipse([bx - 6, cy - 118, bx + 6, cy - 106], fill=col)

    for _ in range(40):
        x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
        draw.rectangle(
            [x, y, x + 4, y + 4],
            fill=rng.choice([(255, 200, 0), (255, 80, 80), (80, 255, 200)]),
        )

    face = Image.open(face_path).convert("RGB")
    fw = int(size * 0.58)
    fh = int(size * 0.72)
    face = face.resize((fw, fh), Image.LANCZOS)
    mask = Image.new("L", (fw, fh), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([0, 0, fw, fh], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(8))
    px, py = int(size * 0.02), int(size * 0.18)
    canvas.paste(face, (px, py), mask)
    return canvas


def draw_greeting(image: Image.Image, text: str) -> Image.Image:
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font_size = int(h * 0.10)
    margin = int(w * 0.05)
    max_w = w - 2 * margin
    while font_size > 10:
        font = ImageFont.truetype(FONT_BOLD, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            break
        font_size -= 2
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (w - tw) / 2 - bbox[0]
    y = int(h * 0.035) - bbox[1]
    pad = int(font_size * 0.35)
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(band)
    bdraw.rectangle(
        [x + bbox[0] - pad, y + bbox[1] - pad, x + bbox[2] + pad, y + bbox[3] + pad],
        fill=(0, 0, 0, 130),
    )
    img = Image.alpha_composite(img.convert("RGBA"), band)
    draw = ImageDraw.Draw(img)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 235, 90),
        stroke_width=max(2, font_size // 18),
        stroke_fill=(60, 20, 0),
    )
    return img.convert("RGB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_DIR)
    p.add_argument("--ref", default=BEST_REF)
    p.add_argument("--strength", type=float, default=0.65,
                   help="img2img: выше = свободнее сцена, ниже = ближе к коллажу")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--text", default="С Днём Рождения!")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--txt2img", action="store_true", help="Без коллажа, только текст")
    p.add_argument("--save-canvas", action="store_true")
    p.add_argument("--cpu", action="store_true", help="Принудительно CPU")
    return p.parse_args()


def pick_device(force_cpu: bool) -> str:
    """FLUX не влезает в 4 ГБ VRAM даже с offload (T5 ~9 ГБ)."""
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram < 12 * (1024**3):
        print(
            f"[*] GPU {vram / (1024**3):.1f} GiB мало для FLUX "
            f"(нужно ≥12 GiB) — CPU"
        )
        return "cpu"
    return "cuda"


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    output = args.output or os.path.join(OUT_DIR, f"flux_{seed}.png")

    if not os.path.isdir(args.model):
        raise SystemExit(
            f"Нет модели: {args.model}\n"
            "Скачай: hf download black-forest-labs/FLUX.1-schnell "
            f"--local-dir {MODEL_DIR}"
        )

    device = pick_device(args.cpu)
    # bf16 экономит RAM на CPU; float32 только если bf16 недоступен
    dtype = torch.bfloat16
    print(f"[*] FLUX.1-schnell | seed={seed} | device={device} | size={args.size}")

    if args.txt2img:
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype)
        gen_kw = dict(
            prompt=PROMPT,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            max_sequence_length=256,
            height=args.size,
            width=args.size,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
    else:
        from diffusers import FluxImg2ImgPipeline

        if not os.path.isfile(args.ref):
            raise SystemExit(f"Нет рефа: {args.ref}")
        canvas = make_festive_canvas(args.ref, args.size, random.Random(seed))
        if args.save_canvas:
            canvas_path = os.path.join(OUT_DIR, f"canvas_flux_{seed}.png")
            canvas.save(canvas_path)
            print(f"[*] Коллаж: {canvas_path}")

        pipe = FluxImg2ImgPipeline.from_pretrained(args.model, torch_dtype=dtype)
        gen_kw = dict(
            prompt=PROMPT,
            image=canvas,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(seed),
        )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        print("[*] CPU mode — обычно 15–45 мин на 512px")
        pipe = pipe.to("cpu")

    print(f"[*] Промпт: {PROMPT}")
    print("[*] Генерация...")
    image = pipe(**gen_kw).images[0]
    if not args.no_text:
        image = draw_greeting(image, args.text)
    image.save(output)
    print(f"[+] {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
