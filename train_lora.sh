#!/usr/bin/env bash
set -e

DIFFUSERS_EXAMPLES_DIR="$HOME/diffusers/examples/text_to_image"
DATASET_DIR="$(pwd)/dataset/images"
OUTPUT_DIR="$(pwd)/shapoklyak_lora"
MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -d "$DATASET_DIR" ] || [ -z "$(find "$DATASET_DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | head -1)" ]; then
  echo "Нет картинок в $DATASET_DIR"
  exit 1
fi

python3 "$(pwd)/prepare_captions.py" --dir "$DATASET_DIR"

TRAIN_SCRIPT="$DIFFUSERS_EXAMPLES_DIR/train_text_to_image_lora.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Не найден $TRAIN_SCRIPT — ./download_diffusers_examples.sh"
  exit 1
fi

INSTALLED_DIFFUSERS="$(python3 -c 'import diffusers; print(diffusers.__version__)')"
sed -i "s/check_min_version(\"[^\"]*\")/check_min_version(\"${INSTALLED_DIFFUSERS}\")/" "$TRAIN_SCRIPT"

# Чистим старую LoRA, чтобы не смешать веса
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$DIFFUSERS_EXAMPLES_DIR"

# 4 ГБ: 384 + checkpointing. rank=8 сильнее держит персонажа.
accelerate launch --mixed_precision=fp16 --num_processes=1 --num_machines=1 --dynamo_backend=no \
  train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_DIR" \
  --resolution=384 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --rank=8 \
  --dataloader_num_workers=0 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="fp16" \
  --seed=42 \
  --checkpointing_steps=150 \
  --report_to="tensorboard"

# Если финальный save упал — подхватить последний чекпоинт
if [ ! -f "$OUTPUT_DIR/pytorch_lora_weights.safetensors" ]; then
  LAST="$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  if [ -n "$LAST" ] && [ -f "$LAST/pytorch_lora_weights.safetensors" ]; then
    cp "$LAST/pytorch_lora_weights.safetensors" "$OUTPUT_DIR/"
    echo "[*] Взял веса из $LAST"
  fi
fi

echo ""
echo "LoRA: $OUTPUT_DIR"
echo "Дальше: python3 generate_shapoklyak_v2.py --festive"
