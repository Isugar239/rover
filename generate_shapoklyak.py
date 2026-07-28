#!/usr/bin/env python3
"""Генератор поздравительной картинки 16:9 со старухой Шапокляк.

Работает локально на видеокарте (Stable Diffusion 1.5) с оптимизациями
под маленький объём VRAM (~4 ГБ). Если CUDA недоступна — падает на CPU.

Режимы:
  1) txt2img  — генерация с нуля по текстовому промпту (по умолчанию).
  2) IP-Adapter — можно дать 1..N скринов Шапокляк из мультика (--ref / --ref-dir),
     и модель возьмёт с них внешность/стиль персонажа.
  3) img2img — перерисовать один исходный кадр (--init).

Примеры:
    # обычная генерация
    python3 generate_shapoklyak.py --seed 42

    # с референсами (скрины из мультика в папке refs/)
    python3 generate_shapoklyak.py --ref-dir refs --ip-scale 0.7 --seed 42

    # перерисовать один кадр
    python3 generate_shapoklyak.py --init refs/kadr.png --strength 0.6
"""

import argparse
import glob
import os
import random

import torch
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Модель по умолчанию — локально скачанный SDXL base (см. download_sdxl.sh).
# Если локальной папки нет — откатываемся на онлайн-идентификатор SDXL.
# Для SD1.5 можно передать --model stable-diffusion-v1-5/stable-diffusion-v1-5
_LOCAL_SDXL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sdxl-base")
DEFAULT_MODEL = _LOCAL_SDXL if os.path.isdir(_LOCAL_SDXL) else "stabilityai/stable-diffusion-xl-base-1.0"
SD15_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# IP-Adapter (репозиторий один, веса/подпапка зависят от модели).
IP_ADAPTER_REPO = "h94/IP-Adapter"
# SD1.5:
IP_ADAPTER_SUBFOLDER_SD15 = "models"
IP_ADAPTER_WEIGHT_SD15 = "ip-adapter_sd15.bin"
# SDXL (вариант vit-h переиспользует уже скачанный ViT-H image encoder):
IP_ADAPTER_SUBFOLDER_SDXL = "sdxl_models"
IP_ADAPTER_WEIGHT_SDXL = "ip-adapter_sdxl_vit-h.bin"


def is_sdxl(model_id: str) -> bool:
    """Грубое определение SDXL по идентификатору модели."""
    return "xl" in model_id.lower()

# Ядро внешности персонажа — не меняется, держит образ Шапокляк.
CHARACTER_PROMPT = (
    "solo single old lady Shapoklyak, one character alone, centered, "
    "big long nose, grey hair bun, "
    "flat-brim black straw hat, white lace collar, dark high-collar coat, "
    "mischievous smile"
)

# Вариативная праздничная сцена — выбирается случайно, чтобы каждый прогон
# давал уникальную картинку.
SCENE_VARIATIONS = [
    "festive birthday scene, colorful balloons and confetti",
    "big birthday cake with lit candles on the table",
    "holding a bunch of colorful balloons",
    "blowing out candles on a cake",
    "party room with paper garlands and gift boxes",
    "surrounded by wrapped presents and streamers",
    "holding a small gift box with a ribbon bow",
    "cozy room with a decorated birthday table and cupcakes",
    "raising a cup for a toast, festive mood",
    "confetti falling, party hats and bright decorations",
]

# Текст поздравления фиксированный, формальный, на английском.
DEFAULT_GREETING = "Happy Birthday"

# Сохраняем совместимость: DEFAULT_PROMPT = базовый образ + первая сцена.
DEFAULT_PROMPT = f"{CHARACTER_PROMPT}, {SCENE_VARIATIONS[0]}, warm cozy lighting, highly detailed"

NEGATIVE_PROMPT = (
    "two people, multiple characters, crowd, duplicate, twins, second person, "
    "lowres, blurry, deformed, disfigured, ugly, extra limbs, extra fingers, "
    "bad anatomy, watermark, text, signature, jpeg artifacts, cropped"
)


def build_random_prompt():
    """Собирает промпт со случайной праздничной сценой."""
    scene = random.choice(SCENE_VARIATIONS)
    return f"{CHARACTER_PROMPT}, {scene}, warm cozy lighting, highly detailed"

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def collect_reference_images(refs, ref_dir):
    """Собирает пути к референсным изображениям из списка и/или папки."""
    paths = list(refs or [])
    if ref_dir:
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(ref_dir, ext)))
    paths = sorted(dict.fromkeys(paths))  # убрать дубли, стабильный порядок
    images = []
    for pth in paths:
        try:
            images.append(Image.open(pth).convert("RGB"))
            print(f"    + референс: {pth}")
        except Exception as e:
            print(f"    ! не смог открыть {pth}: {e}")
    return images


def build_pipeline(model_id: str, use_img2img: bool, use_ip_adapter: bool):
    sdxl = is_sdxl(model_id)

    if sdxl:
        if use_img2img:
            from diffusers import StableDiffusionXLImg2ImgPipeline as PipeCls
        else:
            from diffusers import StableDiffusionXLPipeline as PipeCls
    else:
        if use_img2img:
            from diffusers import StableDiffusionImg2ImgPipeline as PipeCls
        else:
            from diffusers import StableDiffusionPipeline as PipeCls

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    from_kwargs = dict(torch_dtype=dtype)
    if sdxl:
        # Качаем половинные fp16-веса — вдвое меньше трафика и VRAM.
        from_kwargs["variant"] = "fp16"
        from_kwargs["use_safetensors"] = True
    else:
        from_kwargs["safety_checker"] = None   # экономим VRAM и время
        from_kwargs["requires_safety_checker"] = False

    pipe = PipeCls.from_pretrained(model_id, **from_kwargs)

    # Оптимизации памяти — критично для 4 ГБ VRAM.
    # attention slicing несовместим с IP-Adapter (SlicedAttnProcessor ломает
    # загрузку/инференс адаптера), поэтому включаем его только без референсов.
    if not use_ip_adapter:
        pipe.enable_attention_slicing()
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass

    # IP-Adapter грузим до offload.
    if use_ip_adapter:
        print("[*] Загрузка IP-Adapter (референсы)...")
        load_kwargs = dict(
            subfolder=IP_ADAPTER_SUBFOLDER_SDXL if sdxl else IP_ADAPTER_SUBFOLDER_SD15,
            weight_name=IP_ADAPTER_WEIGHT_SDXL if sdxl else IP_ADAPTER_WEIGHT_SD15,
        )
        if sdxl:
            # Вариант sdxl_vit-h обучен на ViT-H энкодере (models/image_encoder),
            # а не на ViT-bigG (sdxl_models/image_encoder, ~3.5 ГБ). Грузим ViT-H
            # энкодер явно из кэша и регистрируем в пайплайне, а авто-загрузку
            # энкодера внутри load_ip_adapter выключаем (image_encoder_folder=None),
            # чтобы diffusers не пытался тянуть bigG из сети.
            from transformers import CLIPVisionModelWithProjection
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                IP_ADAPTER_REPO, subfolder="models/image_encoder", torch_dtype=dtype,
            )
            pipe.register_modules(image_encoder=image_encoder)
            load_kwargs["image_encoder_folder"] = None
        pipe.load_ip_adapter(IP_ADAPTER_REPO, **load_kwargs)

    if use_cuda:
        if sdxl:
            # UNet SDXL (~5 ГБ fp16) целиком в 4 ГБ не влезает, поэтому только
            # посекционная выгрузка на CPU. Медленно, зато не падает по памяти.
            print("[!] SDXL на 4 ГБ VRAM: посекционная выгрузка (генерация будет медленной).")
            pipe.enable_sequential_cpu_offload()
        else:
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to("cuda")
    else:
        print("[!] CUDA недоступна — генерация пойдёт на CPU (медленно).")

    return pipe


def draw_greeting(image: Image.Image, text: str) -> Image.Image:
    """Аккуратно вписывает текст поздравления в верхнюю часть картинки."""
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Подбираем размер шрифта так, чтобы текст влез по ширине.
    font_size = int(h * 0.11)
    margin = int(w * 0.06)
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
    y = int(h * 0.05) - bbox[1]

    # Полупрозрачная подложка для читабельности.
    pad = int(font_size * 0.35)
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(band)
    bdraw.rectangle(
        [x + bbox[0] - pad, y + bbox[1] - pad, x + bbox[2] + pad, y + bbox[3] + pad],
        fill=(0, 0, 0, 110),
    )
    img = Image.alpha_composite(img.convert("RGBA"), band)
    draw = ImageDraw.Draw(img)

    # Текст с контрастной обводкой.
    draw.text(
        (x, y), text, font=font, fill=(255, 235, 90),
        stroke_width=max(2, font_size // 18), stroke_fill=(60, 20, 0),
    )
    return img.convert("RGB")


def main():
    p = argparse.ArgumentParser(description="Поздравление 16:9 с Шапокляк")
    p.add_argument("--text", default=None,
                   help="Текст поздравления (по умолчанию — 'Happy Birthday')")
    p.add_argument("--prompt", default=None,
                   help="Промпт для картинки (по умолчанию — случайная праздничная сцена)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--width", type=int, default=None,
                   help="Ширина генерации (кратно 8; по умолчанию зависит от модели)")
    p.add_argument("--height", type=int, default=None,
                   help="Высота генерации (кратно 8; по умолчанию зависит от модели)")
    p.add_argument("--out-width", type=int, default=1280, help="Ширина финальной картинки")
    p.add_argument("--out-height", type=int, default=720, help="Высота финальной картинки")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=None,
                   help="Сид (по умолчанию — случайный, значение выводится в лог)")
    p.add_argument("--output", default=None,
                   help="Имя файла (по умолчанию — уникальное, с сидом)")

    # Референсы (IP-Adapter).
    p.add_argument("--ref", nargs="+", default=None,
                   help="Пути к референсным скринам Шапокляк")
    p.add_argument("--ref-dir", default=None,
                   help="Папка со скринами Шапокляк (png/jpg/…)")
    p.add_argument("--ip-scale", type=float, default=0.85,
                   help="Насколько сильно опираться на референсы (0..1)")

    # img2img.
    p.add_argument("--init", default=None,
                   help="Исходный кадр для перерисовки (режим img2img)")
    p.add_argument("--strength", type=float, default=0.6,
                   help="Сила перерисовки для img2img (0..1)")
    args = p.parse_args()

    use_img2img = args.init is not None
    ref_images = collect_reference_images(args.ref, args.ref_dir)
    use_ip_adapter = len(ref_images) > 0

    # Разрешение генерации по умолчанию зависит от модели: SDXL любит ~1024,
    # SD1.5 — ~512-768. Кратно 8 и в пропорции 16:9.
    sdxl = is_sdxl(args.model)
    width = args.width if args.width is not None else (1024 if sdxl else 768)
    height = args.height if args.height is not None else (576 if sdxl else 432)

    # Уникальность каждого поздравления: случайные текст/сцена/сид,
    # если пользователь не задал их явно.
    text = args.text if args.text is not None else DEFAULT_GREETING
    prompt = args.prompt if args.prompt is not None else build_random_prompt()
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    output = args.output if args.output is not None else f"shapoklyak_birthday_{seed}.png"

    print(f"[*] Текст: {text!r}")
    print(f"[*] Сид: {seed} (для повтора: --seed {seed})")
    print(f"[*] Промпт: {prompt}")
    print(f"[*] CUDA доступна: {torch.cuda.is_available()}")
    print(f"[*] Режим: {'img2img' if use_img2img else 'txt2img'}"
          f"{' + IP-Adapter' if use_ip_adapter else ''}")
    print("[*] Загрузка модели (первый раз качается из интернета)...")
    pipe = build_pipeline(args.model, use_img2img, use_ip_adapter)

    if use_ip_adapter:
        pipe.set_ip_adapter_scale(args.ip_scale)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    if use_ip_adapter:
        # У нас один IP-Adapter. Если референсов несколько, их нужно передать
        # вложенным списком ([[img1, img2, ...]]) — так diffusers отнесёт все
        # картинки к одному адаптеру, а не к нескольким.
        call_kwargs["ip_adapter_image"] = [ref_images] if len(ref_images) > 1 else ref_images[0]

    if use_img2img:
        init = Image.open(args.init).convert("RGB").resize((width, height), Image.LANCZOS)
        call_kwargs["image"] = init
        call_kwargs["strength"] = args.strength
    else:
        call_kwargs["width"] = width
        call_kwargs["height"] = height

    print("[*] Генерация изображения...")
    result = pipe(**call_kwargs)
    image = result.images[0]

    # Апскейл до финального 16:9.
    image = image.resize((args.out_width, args.out_height), Image.LANCZOS)

    print("[*] Наложение текста поздравления...")
    image = draw_greeting(image, text)

    image.save(output)
    print(f"[+] Готово: {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
