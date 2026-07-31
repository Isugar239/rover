import argparse
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import librosa
import numpy as np
import sounddevice as sd
import torch

from kws_model import KWSConvNet


SAMPLE_RATE = 16000


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    labels = checkpoint.get("labels", ["background", "four", "five"])
    model = KWSConvNet(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, labels


def start_timer(seconds: int, fire: bool) -> threading.Event:
    locked = threading.Event()
    locked.set()

    def worker():
        time.sleep(seconds)
        if fire:
            print(f"Fired after {seconds}s")
        locked.clear()

    threading.Thread(target=worker, daemon=True).start()
    return locked


def main() -> None:
    old_kws = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="KWS runtime (librosa MFCC).")
    parser.add_argument("--model", default=str(old_kws / "models" / "kws_librosa_unknown.pt"))
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--consec-hits", type=int, default=2)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.25)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--min-rms", type=float, default=0.01)
    parser.add_argument("--log-probs", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels = load_model(args.model, device)
    label_to_timer = {"four": 4, "five": 5}

    window_len = int(SAMPLE_RATE * args.window_sec)
    hop_len = int(SAMPLE_RATE * args.hop_sec)

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    ring_buffer = np.zeros(window_len, dtype=np.float32)
    locked: Optional[threading.Event] = None
    last_pred = None
    consec_count = 0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(indata[:, 0].copy())

    def classify(window: np.ndarray):
        mfcc = librosa.feature.mfcc(y=window, sr=SAMPLE_RATE, n_mfcc=args.n_mfcc)
        x = torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        cls = probs.argmax(1).item()
        conf = probs[0, cls].item()
        return cls, conf, probs.squeeze(0).cpu().numpy()

    def processing_loop():
        nonlocal ring_buffer, locked, last_pred, consec_count
        pending = np.zeros(0, dtype=np.float32)

        while True:
            chunk = audio_q.get()
            pending = np.concatenate([pending, chunk])

            while len(pending) >= hop_len:
                new_part = pending[:hop_len]
                pending = pending[hop_len:]
                ring_buffer = np.concatenate([ring_buffer, new_part])[-window_len:]

                if locked is not None and locked.is_set():
                    continue

                rms = float(np.sqrt(np.mean(ring_buffer ** 2)) + 1e-8)
                if rms < args.min_rms:
                    consec_count = 0
                    last_pred = None
                    if args.log_probs:
                        print(f"rms={rms:.6f} < min_rms={args.min_rms:.6f}")
                    continue

                cls, conf, probs = classify(ring_buffer)
                if args.log_probs:
                    parts = [f"{label}:{prob:.3f}" for label, prob in zip(labels, probs)]
                    print(f"probs: {' '.join(parts)} | rms={rms:.6f}")

                label = labels[cls]
                if label not in label_to_timer:
                    consec_count = 0
                    last_pred = None
                    continue

                if conf >= args.threshold:
                    if label == last_pred:
                        consec_count += 1
                    else:
                        consec_count = 1
                    last_pred = label

                    if consec_count >= args.consec_hits:
                        seconds = label_to_timer[label]
                        print(f"Detected '{label}' (conf={conf:.2f}) -> timer {seconds}s")
                        locked = start_timer(seconds, fire=True)
                        ring_buffer[:] = 0
                        consec_count = 0
                        last_pred = None
                else:
                    consec_count = 0
                    last_pred = None

    threading.Thread(target=processing_loop, daemon=True).start()

    stream_kwargs = {}
    if args.device is not None:
        stream_kwargs["device"] = args.device
        print(f"использую входное устройство: {args.device}")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=hop_len,
        callback=audio_callback,
        **stream_kwargs,
    ):
        print("Listening...")
        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
