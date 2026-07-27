#!/usr/bin/env bash
# Надёжная загрузка весов SDXL base (fp16) в локальную папку через curl с
# докачкой (-C -) и авто-обрывом на зависании (--speed-time). Пережёвывает
# нестабильный xet-CDN HuggingFace, на котором виснет snapshot_download.
set -u

REPO="stabilityai/stable-diffusion-xl-base-1.0"
BASE="https://huggingface.co/${REPO}/resolve/main"
DIR="$HOME/dostavshik/models/sdxl-base"

FILES=(
  "model_index.json"
  "scheduler/scheduler_config.json"
  "text_encoder/config.json"
  "text_encoder/model.fp16.safetensors"
  "text_encoder_2/config.json"
  "text_encoder_2/model.fp16.safetensors"
  "tokenizer/merges.txt"
  "tokenizer/special_tokens_map.json"
  "tokenizer/tokenizer_config.json"
  "tokenizer/vocab.json"
  "tokenizer_2/merges.txt"
  "tokenizer_2/special_tokens_map.json"
  "tokenizer_2/tokenizer_config.json"
  "tokenizer_2/vocab.json"
  "unet/config.json"
  "unet/diffusion_pytorch_model.fp16.safetensors"
  "vae/config.json"
  "vae/diffusion_pytorch_model.fp16.safetensors"
)

download_one() {
  local rel="$1"
  local url="${BASE}/${rel}"
  local out="${DIR}/${rel}"
  mkdir -p "$(dirname "$out")"
  for attempt in $(seq 1 60); do
    # -f: падать на HTTP-ошибке; -C -: докачка; --speed-time: обрыв при зависании
    curl -fL -C - --retry 8 --retry-delay 3 --speed-limit 2048 --speed-time 25 \
         -o "$out" "$url" && return 0
    echo "  [retry $attempt] $rel (size=$(stat -c%s "$out" 2>/dev/null || echo 0))"
    sleep 2
  done
  echo "  !! не удалось скачать $rel"
  return 1
}

echo "[*] Каталог: $DIR"
fail=0
for f in "${FILES[@]}"; do
  echo "=== $f ($(date +%H:%M:%S)) ==="
  download_one "$f" || fail=1
done

if [ "$fail" -eq 0 ]; then
  echo "SDXL_DOWNLOAD_DONE"
else
  echo "SDXL_DOWNLOAD_FAILED"
fi
