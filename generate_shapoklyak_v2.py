#!/usr/bin/env python3
"""
Праздничная открытка «не стыдно семье»:
  1) собираем коллаж: лицо Шапокляк из рефа + нарисованный торт/шары
  2) img2img смягчает и сливает сцену
  3) текст «С Днём Рождения!»

  python3 generate_shapoklyak_v2.py
  python3 generate_shapoklyak_v2.py --seed 42
"""
import argparse
import os
import random

import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from diffusers import StableDiffusionImg2ImgPipeline

MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRIGGER = "shpklk_character"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "итог")
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BEST_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "images3.jpeg")

PROMPT = (
    f"birthday cake with lit candles balloons confetti, {TRIGGER}, "
    "Shapoklyak stop-motion puppet black top hat long pointed nose "
    "white lace jabot black dress, festive party, medium shot"
)

NEGATIVE = (
    "ugly, scary, horror, deformed, melted face, asymmetric eyes, smear, "
    "2d anime, photoreal human, santa, man, male, train, railway, "
    "watermark, text, signature, extra limbs"
)


def make_festive_canvas(face_path: str, size: int, rng: random.Random) -> Image.Image:
    """Коллаж: реф Шапокляк + торт + шары — якорь для img2img."""
    canvas = Image.new("RGB", (size, size), (40, 70, 110))
    draw = ImageDraw.Draw(canvas)

    # Фон: сцена / шторы
    draw.rectangle([0, 0, size, size // 3], fill=(90, 20, 30))  # красный верх (шторы)
    draw.rectangle([0, int(size * 0.72), size, size], fill=(120, 90, 60))  # пол

    # Шары
    for _ in range(8):
        x = rng.randint(10, size - 40)
        y = rng.randint(10, size // 3)
        r = rng.randint(18, 36)
        color = rng.choice([(255, 80, 120), (80, 200, 255), (255, 220, 60), (120, 255, 140), (255, 140, 60)])
        draw.ellipse([x, y, x + r, y + r], fill=color)

    # Торт на столе справа внизу
    cx, cy = int(size * 0.72), int(size * 0.78)
    draw.ellipse([cx - 70, cy - 25, cx + 70, cy + 35], fill=(255, 230, 200))
    draw.rectangle([cx - 55, cy - 70, cx + 55, cy], fill=(255, 200, 180))
    draw.ellipse([cx - 55, cy - 85, cx + 55, cy - 55], fill=(255, 170, 160))
    for i, col in enumerate([(255, 80, 80), (80, 180, 255), (255, 220, 80)]):
        bx = cx - 30 + i * 30
        draw.rectangle([bx - 3, cy - 110, bx + 3, cy - 80], fill=(255, 240, 200))
        draw.ellipse([bx - 6, cy - 118, bx + 6, cy - 106], fill=col)

    # Конфетти
    for _ in range(40):
        x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
        draw.rectangle([x, y, x + 4, y + 4], fill=rng.choice([(255, 200, 0), (255, 80, 80), (80, 255, 200)]))

    # Лицо/фигура Шапокляк слева-центр
    face = Image.open(face_path).convert("RGB")
    fw = int(size * 0.58)
    fh = int(size * 0.72)
    face = face.resize((fw, fh), Image.LANCZOS)
    # лёгкое овальное «врезание»
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
        (x, y), text, font=font, fill=(255, 235, 90),
        stroke_width=max(2, font_size // 18), stroke_fill=(60, 20, 0),
    )
    return img.convert("RGB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-dir", default="shapoklyak_lora")
    p.add_argument("--lora-scale", type=float, default=0.55,
                   help="Слабее — лицо ближе к рефу")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--ip-scale", type=float, default=0.45)
    p.add_argument("--no-ip-adapter", action="store_true")
    p.add_argument("--ref", default=BEST_REF)
    p.add_argument("--strength", type=float, default=0.48,
                   help="img2img: 0.40-0.55 (ниже = больше похожа на коллаж/реф)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--text", default="С Днём Рождения!")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--save-canvas", action="store_true", help="Сохранить коллаж до img2img")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    rng = random.Random(seed)
    output = args.output or os.path.join(OUT_DIR, f"festive_{seed}.png")

    if not os.path.isfile(args.ref):
        raise SystemExit(f"Нет рефа: {args.ref}")

    canvas = make_festive_canvas(args.ref, args.size, rng)
    if args.save_canvas:
        canvas_path = os.path.join(OUT_DIR, f"canvas_{seed}.png")
        canvas.save(canvas_path)
        print(f"[*] Коллаж: {canvas_path}")

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"[*] Сид {seed} | strength={args.strength} | LoRA={args.lora_scale} | IP={args.ip_scale}")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
    )

    if not args.no_lora and os.path.isdir(args.lora_dir):
        pipe.load_lora_weights(args.lora_dir)
        print(f"[*] LoRA: {args.lora_dir}")
    else:
        print("[*] Без LoRA")

    use_ip = not args.no_ip_adapter
    if use_ip:
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin",
        )
        pipe.set_ip_adapter_scale(args.ip_scale)
        ip_img = Image.open(args.ref).convert("RGB")
        print(f"[*] IP-реф: {args.ref}")
    else:
        ip_img = None
        pipe.enable_attention_slicing()

    if use_cuda:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    print(f"[*] Промпт: {PROMPT}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = dict(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=canvas,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
    )
    if not args.no_lora and os.path.isdir(args.lora_dir):
        kwargs["cross_attention_kwargs"] = {"scale": args.lora_scale}
    if ip_img is not None:
        kwargs["ip_adapter_image"] = ip_img

    print("[*] Генерация...")
    image = pipe(**kwargs).images[0]
    if not args.no_text:
        image = draw_greeting(image, args.text)
    image.save(output)
    print(f"[+] {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
