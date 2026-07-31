#!/usr/bin/env bash
# Лёгкая докачка только скрипта обучения LoRA (не весь репозиторий).
# Аналог download_sdxl.sh: curl с докачкой, без git clone (~50 КБ вместо сотен МБ).
#
# Запускай, когда трафик/вайтлист позволят. Сейчас НЕ запускать без нужды.
#
# Использование:
#   chmod +x download_diffusers_examples.sh
#   ./download_diffusers_examples.sh
set -u

# Коммит/ветка raw.githubusercontent — main достаточно стабилен для examples.
BASE="https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image"
DIR="${HOME}/diffusers/examples/text_to_image"

# Нужен только train-скрипт; pip-пакет diffusers у тебя уже стоит.
# requirements.txt — справочно (зависимости уже могут быть установлены).
FILES=(
  "train_text_to_image_lora.py"
  "requirements.txt"
)

download_one() {
  local rel="$1"
  local url="${BASE}/${rel}"
  local out="${DIR}/${rel}"
  mkdir -p "$(dirname "$out")"
  for attempt in $(seq 1 30); do
    curl -fL -C - --retry 5 --retry-delay 2 --speed-limit 1024 --speed-time 20 \
         -o "$out" "$url" && return 0
    echo "  [retry $attempt] $rel (size=$(stat -c%s "$out" 2>/dev/null || echo 0))"
    sleep 2
  done
  echo "  !! не удалось скачать $rel"
  return 1
}

echo "[*] Каталог: $DIR"
echo "[*] Это НЕ веса модели — только example-скрипт обучения (~50 КБ)."
fail=0
for f in "${FILES[@]}"; do
  echo "=== $f ($(date +%H:%M:%S)) ==="
  download_one "$f" || fail=1
done

if [ "$fail" -eq 0 ]; then
  echo "DIFFUSERS_EXAMPLES_DOWNLOAD_DONE"
  echo "Дальше: ./train_lora.sh"
else
  echo "DIFFUSERS_EXAMPLES_DOWNLOAD_FAILED"
  exit 1
fi
