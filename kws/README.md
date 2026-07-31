# Keyword Spotting (four / five)

Распознавание слов **four** и **five** с микрофона в реальном времени. Модель обучена на [Google Speech Commands v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands) с классами:

| Класс | Назначение |
|-------|------------|
| `background` | тишина и фоновый шум |
| `unknown` | любая другая речь из датасета |
| `four` | целевое слово |
| `five` | целевое слово |

Актуальный чекпоинт: `models/kws_best_hard.pt`.

Старые эксперименты лежат в `old/kws/`.

## Зависимости

```bash
pip install torch torchaudio sounddevice numpy
```

Датасет для дообучения: каталог `data/` в корне репозитория (Speech Commands v0.02).

## Быстрый старт

Список микрофонов:

```bash
python3 kws/run_best_kws.py --list-devices
```

Запуск с логом вероятностей (рекомендуется `--device 0` — физический вход, не pulse):

```bash
python3 kws/run_best_kws.py --device 0 --log-probs
```

Один триггер и выход:

```bash
python3 kws/run_best_kws.py --device 0 --once
```

## Настройка микрофона

На ноутбуке часто включены **Capture +30 dB** и **Internal Mic Boost +30 dB** — сигнал клиппит и модель «не слышит» нормально. Перед тестом:

```bash
amixer -c 0 set Capture 35%
amixer -c 0 set 'Internal Mic Boost' 0
```

В рантайме при клиппинге (`peak ≈ 1.0`) скрипт пишет предупреждение.

## Архитектура

```
kws/
├── kws_best.py          # модель, log-mel frontend, датасет с аугментациями
├── train_best_kws.py    # обучение + калибровка порогов
├── run_best_kws.py      # микрофон → детект → таймер
└── models/
    └── kws_best_hard.pt
```

**Frontend** — один и тот же `LogMelFrontend` (torchaudio) при обучении и inference: 16 kHz, окно 1 с, 40 mel-полос.

**Модель** — `RobustKWSNet` (depthwise-separable CNN, ~45k параметров).

**Аугментации при обучении:** шум до −5 dB SNR, реверб, сдвиг, громкость, клиппинг; hard negatives (`forward`, `follow`, `right`, …).

**Runtime:** EMA по вероятностям, порог + margin над вторым классом, 3 подряд попадания, cooldown 2 с.

## Обучение

Из корня репозитория:

```bash
python3 kws/train_best_kws.py \
  --data-root data \
  --output kws/models/kws_best_hard.pt \
  --epochs 15 \
  --batch-size 128
```

По умолчанию `--data-root` указывает на `../data`, `--output` — на `kws/models/kws_best_hard.pt`.

## Параметры runtime

| Флаг | По умолчанию | Описание |
|------|--------------|----------|
| `--model` | `kws/models/kws_best_hard.pt` | чекпоинт v2 |
| `--device` | системный default | индекс микрофона |
| `--compute-device` | `cpu` | где считать модель (KWS лучше на CPU, GPU — для SD) |
| `--threshold` | из чекпоинта | переопределить порог |
| `--margin` | `0.18` | отрыв от второго класса |
| `--consecutive-hits` | `3` | подряд уверенных окон |
| `--min-rms` | `0.003` | отсечка тишины |
| `--log-probs` | — | печать вероятностей |

Калиброванные пороги в чекпоинте (validation): `four ≈ 0.85`, `five ≈ 0.80`.

## Метрики (test, hard-negative модель)

- **four:** recall ~83%, sample FPR ~0.06%
- **five:** recall ~93%, sample FPR ~0.2%
- **unknown / background:** recall ~97–100%

На реальном микрофоне качество сильно зависит от уровня входа и фона.

## Что добавить для ещё лучшего качества

1. Записи с **вашего** микрофона: 100–200 раз `four`/`five` + фон (TV, разговоры).
2. Шумовые датасеты: [MUSAN](https://www.openslr.org/17/), [DEMAND](https://zenodo.org/record/1227121).
3. Доп. negative speech: [Common Voice](https://commonvoice.mozilla.org/) (ru/en).
4. Дообучение: положить свои wav в структуру `data/four/`, `data/five/` и снова `train_best_kws.py`.

## Интеграция с live_compose

Генерация открытки (`live_compose.py`) и KWS конкурируют за GPU на 4 GB. Рекомендуемый режим:

```bash
python3 live_compose.py --device cpu   # SD на CPU, GPU свободна
python3 kws/run_best_kws.py --compute-device cuda --device 0
```

или KWS на CPU, SD на CUDA — как удобнее по задержкам.
