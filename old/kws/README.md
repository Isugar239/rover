# Архив: старые KWS-эксперименты

Ранние версии keyword spotting (MFCC/librosa, five-only, GUI). Оставлены для истории; **не использовать в проде**.

Актуальная версия: [`kws/README.md`](../../kws/README.md).

| Файл | Описание |
|------|----------|
| `kws_model.py` | маленькая CNN (24k params) |
| `kws_data.py` | датасет four/five/background |
| `train_kws.py`, `train_five_only.py`, `train_kws_librosa_unknown.py` | обучение |
| `test_kws_runtime.py`, `kws_runtime_librosa.py`, `kws_gui.py` | runtime / GUI |
| `models/*.pt` | старые чекпоинты |

Запуск (из корня репозитория):

```bash
python3 old/kws/kws_gui.py
python3 old/kws/test_kws_runtime.py --log-probs
```

Пути к моделям и `data/` прописаны относительно расположения скриптов.
