import argparse
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from kws_model import KWSConvNet


def load_model(model_path: str, device: torch.device) -> Tuple[KWSConvNet, list[str]]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    labels = checkpoint.get("labels", ["background", "four", "five"])
    model = KWSConvNet(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, labels


def start_countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"таймер: {remaining} сек")
        time.sleep(1)
    print("таймер: 0 сек")


def main() -> None:
    old_kws = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Real-time KWS test script.")
    parser.add_argument("--model", default=str(old_kws / "models" / "kws_cnn.pt"))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--threshold-four", type=float, default=None)
    parser.add_argument("--threshold-five", type=float, default=None)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--top2-margin", type=float, default=0.15)
    parser.add_argument("--avg", type=int, default=4)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--consec-hits", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--blocksize", type=int, default=0)
    parser.add_argument("--latency", default="high")
    parser.add_argument("--log-probs", action="store_true")
    parser.add_argument("--save-trigger", default=str(old_kws / "trigger.wav"))
    parser.add_argument("--min-rms", type=float, default=0.005)
    parser.add_argument("--buffer-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.25)
    args = parser.parse_args()

    if args.list_devices:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"[{i}] {dev['name']} (in={dev['max_input_channels']})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels = load_model(args.model, device)
    label_to_timer = {"four": 4, "five": 5}

    device_sample_rate = args.sample_rate
    if device_sample_rate is None and args.device is not None:
        device_info = sd.query_devices(args.device, "input")
        device_sample_rate = int(device_info["default_samplerate"])
        print(f"sample_rate из устройства: {device_sample_rate} Hz")
    if device_sample_rate is None:
        device_sample_rate = 16000
    model_sample_rate = 16000
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=model_sample_rate,
        n_mfcc=40,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 64,
            "center": False,
        },
    ).to(device)

    buffer_len = int(device_sample_rate * args.buffer_sec)
    hop_len = int(device_sample_rate * args.hop_sec)
    audio_buffer: Deque[float] = deque(maxlen=buffer_len)
    work_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    locked = threading.Event()
    stop_event = threading.Event()
    hits = {"four": 0, "five": 0}
    prob_history: dict[str, Deque[float]] = {
        "four": deque(maxlen=args.avg),
        "five": deque(maxlen=args.avg),
        "background": deque(maxlen=args.avg),
    }
    pending_samples = 0

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"audio status: {status}")
        if locked.is_set():
            return
        nonlocal pending_samples
        mono = indata[:, 0].astype(np.float32)
        audio_buffer.extend(mono.tolist())
        pending_samples += frames
        while len(audio_buffer) == buffer_len and pending_samples >= hop_len:
            chunk = np.array(audio_buffer, dtype=np.float32)
            work_queue.put(chunk)
            pending_samples -= hop_len

    def worker() -> None:
        while not stop_event.is_set():
            try:
                chunk = work_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if locked.is_set():
                continue
            rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-8)
            if rms < args.min_rms:
                hits["four"] = 0
                hits["five"] = 0
                if args.log_probs:
                    print(f"rms: {rms:.6f} < min_rms={args.min_rms:.6f} -> skip")
                continue
            waveform = torch.tensor(chunk).unsqueeze(0).to(device)
            if device_sample_rate != model_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, device_sample_rate, model_sample_rate
                )
            with torch.no_grad():
                features = mfcc(waveform).unsqueeze(1)
                logits = model(features)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            best_idx = int(np.argmax(probs))
            best_label = labels[best_idx]
            best_prob = float(probs[best_idx])
            sorted_probs = sorted([float(p) for p in probs], reverse=True)
            second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            if args.log_probs:
                parts = [f"{label}:{float(prob):.3f}" for label, prob in zip(labels, probs)]
                print(f"probs: {' '.join(parts)} | rms={rms:.6f}")

            for label, prob in zip(labels, probs):
                if label in prob_history:
                    prob_history[label].append(float(prob))

            background_prob = (
                float(np.mean(prob_history["background"]))
                if prob_history["background"]
                else 0.0
            )

            if best_label in hits:
                avg_prob = float(np.mean(prob_history[best_label])) if prob_history[best_label] else 0.0
                if best_label == "four" and args.threshold_four is not None:
                    thr = args.threshold_four
                elif best_label == "five" and args.threshold_five is not None:
                    thr = args.threshold_five
                else:
                    thr = args.threshold
                confident = (
                    avg_prob >= thr
                    and (avg_prob - background_prob) >= args.margin
                    and (avg_prob - second_prob) >= args.top2_margin
                )
                if confident:
                    hits[best_label] += 1
                else:
                    hits[best_label] = 0

                if hits[best_label] >= args.consec_hits:
                    locked.set()
                    seconds = label_to_timer.get(best_label, 4)
                    try:
                        wave = torch.tensor(chunk).unsqueeze(0)
                        torchaudio.save(args.save_trigger, wave, device_sample_rate)
                        print(f"сохранил триггер в {args.save_trigger}")
                    except Exception as exc:
                        print(f"не удалось сохранить триггер: {exc}")
                    print(f"детект: {best_label} ({avg_prob:.3f}), таймер {seconds} сек")
                    start_countdown(seconds)
                    stop_event.set()
            else:
                hits["four"] = 0
                hits["five"] = 0

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    print("старт записи. скажи 'four' или 'five'...")
    stream_kwargs = {}
    if args.device is not None:
        stream_kwargs["device"] = args.device
        print(f"использую входное устройство: {args.device}")

    with sd.InputStream(
        samplerate=device_sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=args.blocksize if args.blocksize > 0 else 0,
        latency=args.latency,
        **stream_kwargs,
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

    print("готово, выхожу.")


if __name__ == "__main__":
    main()
