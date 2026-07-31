#!/usr/bin/env python3
"""Пачка тестовых генераций: разные seed / strength / позы. Пишет сводку в итог/test_batch/."""
from __future__ import annotations

import csv
import os
import time

from live_compose import BackgroundEngine, run_once
from compose_lib import OUT_DIR, SCENE_PROMPT_VARIANTS, scene_prompt, get_pose

BANK = os.path.join(OUT_DIR, "test_batch")

# сетка: strength × seeds × позы (часть)
STRENGTHS = [0.72, 0.76, 0.80]
SEEDS = [101, 202, 303, 404, 505, 777, 888, 999, 1234, 2026, 31415, 4242]
POSES = ["left_look_right", "waist_up_cake", "front_gesture"]
# фиксируем cake — ищем стабильный торт без лампы
THEME = "cake"


def main():
    os.makedirs(BANK, exist_ok=True)
    engine = BackgroundEngine(backend="sd15", size=512, steps=60, strength=0.76)
    engine.load()

    rows = []
    n = 0
    total = len(STRENGTHS) * len(SEEDS)  # pose rotates by seed
    t_all = time.time()
    for strength in STRENGTHS:
        engine.strength = strength
        for i, seed in enumerate(SEEDS):
            pose_id = POSES[i % len(POSES)]
            # жёстко один из cake-промптов по кругу (без слова lamp)
            prompts = SCENE_PROMPT_VARIANTS["cake"]
            prompt = prompts[i % len(prompts)]
            n += 1
            out = os.path.join(BANK, f"s{strength:.2f}_{pose_id}_{seed}.png")
            print(f"\n=== [{n}/{total}] strength={strength} pose={pose_id} seed={seed} ===")
            print(f"    prompt: {prompt}")
            t0 = time.time()
            # подменим scene_prompt через прямой generate + compose
            pose = get_pose(pose_id)
            bg = engine.generate(prompt, seed, theme=THEME)
            from compose_lib import place_character, draw_greeting
            card = place_character(bg, pose)
            card = draw_greeting(card, "Happy Birthday!")
            card.save(out)
            dt = time.time() - t0
            print(f"[+] {out} ({dt:.1f}s)")
            rows.append(
                {
                    "file": out,
                    "strength": strength,
                    "pose": pose_id,
                    "seed": seed,
                    "prompt": prompt,
                    "sec": round(dt, 1),
                }
            )

    summary = os.path.join(BANK, "summary.csv")
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[+] {len(rows)} кадров за {time.time()-t_all:.0f}s → {BANK}")
    print(f"[+] сводка: {summary}")


if __name__ == "__main__":
    main()
