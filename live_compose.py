#!/usr/bin/env python3
"""
Сценическая открытка: один запуск — от загрузки модели до готового PNG.

  python3 live_compose.py

Тема, поза, seed и вид комнаты выбираются сами. Опции только если нужно отладить.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import torch
from PIL import Image

from compose_lib import (
    OUT_DIR,
    draw_greeting,
    load_poses,
    pick_pose,
    pick_framing,
    place_character,
    scene_prompt,
    scrub_left_background,
    NEGATIVE_BG,
    get_pose,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SD15_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/"
    "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)
SD15 = SD15_SNAP if os.path.isdir(SD15_SNAP) else "stable-diffusion-v1-5/stable-diffusion-v1-5"
SDXL_DIR = os.path.join(ROOT, "models", "sdxl-base")
LCM_LORA_LOCAL = os.path.join(ROOT, "models", "lcm-lora-sd15")
LCM_LORA_HUB = "latent-consistency/lcm-lora-sdv1-5"

GREETINGS = [
    "Happy Birthday!",
]


class BackgroundEngine:
    def __init__(
        self,
        backend: str = "auto",
        size: int = 768,
        steps: int = 60,
        strength: float = 0.74,
        device: str = "auto",
    ):
        self.size = size
        self.steps = steps
        self.strength = strength
        self.pipe = None
        self._txt2img = None
        self.backend = backend
        self.device_pref = device  # auto | cuda | cpu
        self.device = "cpu"
        self.dtype = torch.float32
        self.use_lcm = False

    def _pick_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        # SDXL на 4GB + KWS не уживётся; на CPU — очень долго. auto → sd15.
        if (
            self.device_pref != "cpu"
            and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory >= 5.5 * (1024**3)
            and os.path.isdir(SDXL_DIR)
        ):
            return "sdxl"
        return "sd15"

    def _resolve_device(self) -> str:
        if self.device_pref == "cpu":
            return "cpu"
        if self.device_pref == "cuda":
            if not torch.cuda.is_available():
                print("[!] CUDA недоступна, падаем на CPU")
                return "cpu"
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        backend = self._pick_backend()
        self.backend = backend
        use_cuda = self._resolve_device() == "cuda"
        self.device = "cuda" if use_cuda else "cpu"
        self.dtype = torch.float16 if use_cuda else torch.float32
        print(f"[*] backend={backend} device={self.device}")

        if backend == "sdxl":
            from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

            kw = dict(
                torch_dtype=self.dtype,
                use_safetensors=True,
                local_files_only=True,
            )
            if use_cuda:
                kw["variant"] = "fp16"
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(SDXL_DIR, **kw)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.steps = max(self.steps, 30)
            self.guidance = 6.5
            self.use_lcm = False
        else:
            from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                SD15,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.steps = max(self.steps, 60)
            self.guidance = 8.5
            self.use_lcm = False
            self.size = min(self.size, 512)

        if use_cuda:
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                self.pipe = self.pipe.to("cuda")
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass
        else:
            self.pipe = self.pipe.to("cpu")
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass
            # на CPU чуть меньше шагов по умолчанию, если юзер не задал явно
            print("[*] CPU-режим: GPU свободна для KWS")
        print("[+] модель готова")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        theme: str = "cake",
        framing=None,
    ) -> Image.Image:
        assert self.pipe is not None
        from compose_lib import FRAMINGS, make_layout_canvas
        from PIL import ImageFilter, ImageEnhance

        framing = framing or FRAMINGS[0]
        rng = random.Random(seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        layout = make_layout_canvas(self.size, theme, rng, framing=framing)
        strength = self.strength + rng.uniform(-0.01, 0.01)
        strength = max(0.58, min(0.66, strength))
        steps = self.steps
        t0 = time.time()
        image = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_BG,
            image=layout,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=getattr(self, "guidance", 7.5),
            generator=gen,
        ).images[0]

        image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=140, threshold=2))
        image = ImageEnhance.Contrast(image).enhance(1.1)
        image = ImageEnhance.Color(image).enhance(1.08)
        print(
            f"[*] фон за {time.time() - t0:.1f}s  "
            f"steps={steps} strength={strength:.2f} size={self.size}"
        )
        return image


def parse_args():
    p = argparse.ArgumentParser(description="Один запуск: AI фон + Шапокляк → PNG")
    p.add_argument("--theme", choices=["cake", "party", "family"], default=None,
                   help="по умолчанию случайная")
    p.add_argument("--pose", default=None, help="id позы; иначе случайная")
    p.add_argument("--text", default=None, help="текст; иначе случайный из списка")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--strength", type=float, default=0.62)
    p.add_argument("--backend", choices=["auto", "sd15", "sd15_lcm", "sdxl"], default="auto")
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="cpu — GPU свободна для KWS (медленнее); cuda — генерация на видеокарте",
    )
    p.add_argument("--output", default=None)
    p.add_argument("--save-bg", action="store_true")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--bg", default=None, help="готовый фон PNG (без генерации)")
    p.add_argument(
        "--framing",
        default=None,
        help="ракурс фона: cake_right (рабочий), cake_far_right, wide_left, "
             "close_left, party_balloons, window_light; иначе случайный",
    )
    return p.parse_args()


def run_once(
    engine: BackgroundEngine | None,
    theme: str,
    text: str,
    pose_id: str | None,
    seed: int | None,
    save_bg: bool,
    no_text: bool,
    output: str | None,
    bg_path: str | None = None,
    framing_id: str | None = None,
):
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    framing = pick_framing(rng, framing_id=framing_id)
    # всегда нормальный вырез (битый waist_up_cake убран)
    pose = get_pose(pose_id) if pose_id else pick_pose(theme, rng)
    if pose.id == "waist_up_cake":
        pose = get_pose("left_look_right")
    print(f"[*] поза={pose.id} framing={framing.id} theme={theme} seed={seed}")

    if bg_path:
        bg = Image.open(bg_path).convert("RGB")
        print(f"[*] фон из файла: {bg_path}")
    else:
        assert engine is not None
        prompt = scene_prompt(theme, pose, rng, framing=framing)
        print(f"[*] prompt: {prompt}")
        bg = engine.generate(prompt, seed, theme=theme, framing=framing)
        bg = scrub_left_background(bg, frac=framing.scrub_frac)
        os.makedirs(OUT_DIR, exist_ok=True)
        if save_bg:
            bp = os.path.join(OUT_DIR, f"bg_{theme}_{seed}.png")
            bg.save(bp)
            print(f"[*] фон: {bp}")

    scale = pose.scale * framing.scale_mul
    card = place_character(bg, pose, scale=scale)
    if not no_text:
        card = draw_greeting(card, text)

    out = output or os.path.join(OUT_DIR, f"live_{theme}_{pose.id}_{seed}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    card.save(out, quality=95)
    print(f"[+] {os.path.abspath(out)}")
    return out


def main():
    args = parse_args()
    try:
        load_poses()
    except Exception as e:
        raise SystemExit(f"Нет поз: {e}\nСначала: python3 prepare_cutouts.py")

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    text = args.text or "Happy Birthday!"
    # для сцены по умолчанию торт — меньше случайных «ламп/подарков»
    theme = args.theme or "cake"

    engine = None
    if not args.bg:
        engine = BackgroundEngine(
            backend=args.backend,
            size=args.size,
            steps=args.steps,
            strength=args.strength,
            device=args.device,
        )
        print("[*] загрузка модели...")
        t0 = time.time()
        engine.load()
        print(f"[*] модель за {time.time() - t0:.1f}s")

    try:
        run_once(
            engine,
            theme,
            text,
            args.pose,
            seed,
            args.save_bg,
            args.no_text,
            args.output,
            args.bg,
            framing_id=args.framing,
        )
    except Exception as e:
        print(f"[!] ошибка: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
