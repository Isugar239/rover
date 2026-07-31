#!/usr/bin/env python3
"""Капшены для LoRA. Короткие — CLIP учит триггер, не хвост."""
import argparse
import json
import os

TRIGGER = "shpklk_character"
BASE_CAPTION = (
    f"{TRIGGER}, Starukha Shapoklyak stop-motion puppet, "
    "black top hat, long pointed nose, white lace jabot, black dress, sly smile"
)
VARIANTS = [
    "front view",
    "side view",
    "three quarter view",
    "close up portrait",
    "holding object",
    "full body",
    "smiling",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="dataset/images")
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(args.dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not files:
        print(f"В {args.dir} нет картинок")
        return

    meta_rows = []
    for i, fname in enumerate(files):
        stem = os.path.splitext(fname)[0]
        caption = f"{BASE_CAPTION}, {VARIANTS[i % len(VARIANTS)]}"
        with open(os.path.join(args.dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(caption)
        meta_rows.append({"file_name": fname, "text": caption})
        print(f"{fname} -> {caption}")

    meta_path = os.path.join(args.dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nГотово: {len(files)} капшенов")


if __name__ == "__main__":
    main()
