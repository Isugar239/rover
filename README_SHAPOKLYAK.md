# Шапокляк LoRA — быстрый пайплайн

## 1. Собери датасет
```bash
mkdir -p dataset/images
# закинь туда 15-20 картинок (свои 3 + доп. кадры из мультфильма)
python3 prepare_captions.py --dir dataset/images
```

## 2. Обучи LoRA (~15-20 мин на 3060 Ti)
```bash
chmod +x train_lora.sh
./train_lora.sh
```

## 3. Сгенерируй картинку
Без IP-Adapter (просто LoRA):
```bash
pip install diffusers transformers accelerate safetensors peft pillow
python3 generate_shapoklyak_v2.py
```

С IP-Adapter поверх (LoRA + твои 3 референса):
```bash
mkdir -p refs
# закинь свои 3 фото Шапокляк в refs/
python3 generate_shapoklyak_v2.py --ip-adapter --ref-dir refs --ip-scale 0.6
```

Результаты сохраняются в папку `итог/`.

## Если результат кривой — крути эти параметры
- `--lora-scale` 0.6-1.0 (выше = сильнее сходство, но риск переобучения/артефактов)
- `--ip-scale` 0.4-0.7 (выше = сильнее копирует референс, но больше шума)
- `--steps` 30-45 (выше = чище, но дольше)
- `--guidance` 5.5-7.5 (выше = больше следует промпту, но может "передавить")
- `--seed` — поменяй, если конкретный результат не понравился, а параметры ок
