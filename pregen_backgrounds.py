#!/usr/bin/env python3
"""
Прогон нескольких фонов заранее (если на сцене совсем туго со временем).

  python3 pregen_backgrounds.py
  python3 live_compose.py --theme cake --bg итог/bg_bank/cake_42.png
"""
from __future__ import annotations

import argparse
import os
import random

from live_compose import BackgroundEngine
from compose_lib import OUT_DIR, pick_pose, scene_prompt

BANK = os.path.join(OUT_DIR, "bg_bank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="сколько фонов на тему")
    ap.add_argument("--themes", default="cake,party,family")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(BANK, exist_ok=True)
    engine = BackgroundEngine(backend="sd15_lcm", size=args.size, steps=args.steps)
    engine.load()
    themes = [t.strip() for t in args.themes.split(",") if t.strip()]
    for theme in themes:
        pose = pick_pose(theme)
        for i in range(args.n):
            seed = random.randint(0, 2**31 - 1)
            prompt = scene_prompt(theme, pose)
            print(f"[*] {theme} #{i+1} seed={seed}")
            bg = engine.generate(prompt, seed, theme=theme)
            path = os.path.join(BANK, f"{theme}_{seed}.png")
            bg.save(path)
            print(f"[+] {path}")
    print("[+] банк фонов готов:", BANK)


if __name__ == "__main__":
    main()
