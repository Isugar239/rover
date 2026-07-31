#!/usr/bin/env python3
"""Real-time runner for the robust KWS v2 checkpoint."""
from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from kws_best import SAMPLE_RATE, WINDOW_SAMPLES, load_checkpoint

KWS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = KWS_DIR / "models" / "kws_best_hard.pt"


def list_devices() -> None:
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(
                f"[{index}] {device['name']} "
                f"channels={device['max_input_channels']} "
                f"rate={device['default_samplerate']:.0f}"
            )


def start_timer(seconds: int) -> None:
    def worker() -> None:
        for remaining in range(seconds, 0, -1):
            print(f"timer: {remaining}")
            time.sleep(1)
        print("timer: FIRE")

    threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", type=int, default=None, help="microphone device index")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--compute-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--consecutive-hits", type=int, default=None)
    parser.add_argument("--hop-sec", type=float, default=None)
    parser.add_argument("--cooldown-sec", type=float, default=None)
    parser.add_argument("--min-rms", type=float, default=0.003)
    parser.add_argument("--log-probs", action="store_true")
    parser.add_argument("--save-trigger", default="trigger_best.wav")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    compute_device = torch.device(args.compute_device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    model, frontend, checkpoint = load_checkpoint(args.model, compute_device)
    labels: list[str] = checkpoint["labels"]
    targets: list[str] = checkpoint["target_labels"]
    thresholds: dict[str, float] = checkpoint["thresholds"]
    runtime = checkpoint["runtime"]
    margin = args.margin if args.margin is not None else float(runtime["margin"])
    consecutive_hits = (
        args.consecutive_hits
        if args.consecutive_hits is not None
        else int(runtime["consecutive_hits"])
    )
    hop_sec = args.hop_sec if args.hop_sec is not None else float(runtime["hop_seconds"])
    cooldown_sec = (
        args.cooldown_sec
        if args.cooldown_sec is not None
        else float(runtime["cooldown_seconds"])
    )
    ema_weight = float(runtime["score_ema"])

    device_info = sd.query_devices(args.device, "input")
    microphone_rate = int(device_info["default_samplerate"])
    input_window = int(microphone_rate)
    input_hop = max(1, int(microphone_rate * hop_sec))
    ring: deque[float] = deque(maxlen=input_window)
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=12)
    stop_event = threading.Event()

    smoothed = np.zeros(len(labels), dtype=np.float32)
    initialized = False
    hits = {target: 0 for target in targets}
    cooldown_until = 0.0
    pending_samples = 0
    last_clip_warning = 0.0

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"audio status: {status}")
        mono = indata[:, 0].astype(np.float32, copy=True)
        try:
            audio_queue.put_nowait(mono)
        except queue.Full:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            audio_queue.put_nowait(mono)

    print(
        f"listening on [{args.device if args.device is not None else 'default'}] "
        f"{device_info['name']} at {microphone_rate} Hz; targets={targets}"
    )
    print(
        "thresholds="
        + ", ".join(
            f"{target}:{args.threshold if args.threshold is not None else thresholds[target]:.3f}"
            for target in targets
        )
        + f" margin={margin:.3f} hits={consecutive_hits}"
    )

    with sd.InputStream(
        device=args.device,
        samplerate=microphone_rate,
        channels=1,
        dtype="float32",
        latency="low",
        callback=callback,
    ):
        while not stop_event.is_set():
            try:
                samples = audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            ring.extend(samples.tolist())
            pending_samples += samples.size
            if len(ring) < input_window:
                continue

            while pending_samples >= input_hop:
                pending_samples -= input_hop
                chunk = np.asarray(ring, dtype=np.float32)
                rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-10))
                peak = float(np.max(np.abs(chunk)))
                now = time.monotonic()
                if peak >= 0.995 and now - last_clip_warning > 2.0:
                    print("warning: microphone is clipping; lower its input gain")
                    last_clip_warning = now
                if rms < args.min_rms or now < cooldown_until:
                    for target in targets:
                        hits[target] = 0
                    continue

                waveform = torch.from_numpy(chunk.copy()).unsqueeze(0)
                if microphone_rate != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(
                        waveform, microphone_rate, SAMPLE_RATE
                    )
                if waveform.shape[-1] != WINDOW_SAMPLES:
                    waveform = waveform[..., :WINDOW_SAMPLES]
                    if waveform.shape[-1] < WINDOW_SAMPLES:
                        waveform = torch.nn.functional.pad(
                            waveform, (0, WINDOW_SAMPLES - waveform.shape[-1])
                        )
                waveform = waveform.to(compute_device)
                with torch.inference_mode():
                    probabilities = (
                        model(frontend(waveform))
                        .softmax(dim=1)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

                if not initialized:
                    smoothed[:] = probabilities
                    initialized = True
                else:
                    smoothed[:] = ema_weight * probabilities + (1.0 - ema_weight) * smoothed
                best_index = int(np.argmax(smoothed))
                best_label = labels[best_index]

                if args.log_probs:
                    values = " ".join(
                        f"{label}:{score:.3f}" for label, score in zip(labels, smoothed)
                    )
                    print(f"{values} rms={rms:.4f} peak={peak:.3f}")

                for target in targets:
                    target_index = labels.index(target)
                    target_score = float(smoothed[target_index])
                    other_score = float(
                        np.max(np.delete(smoothed, target_index))
                    )
                    threshold = (
                        args.threshold
                        if args.threshold is not None
                        else float(thresholds[target])
                    )
                    confident = (
                        best_label == target
                        and target_score >= threshold
                        and target_score - other_score >= margin
                    )
                    hits[target] = hits[target] + 1 if confident else 0
                    if hits[target] < consecutive_hits:
                        continue

                    print(
                        f"DETECTED {target}: score={target_score:.3f} "
                        f"margin={target_score - other_score:.3f}"
                    )
                    torchaudio.save(
                        args.save_trigger,
                        waveform.detach().cpu(),
                        SAMPLE_RATE,
                    )
                    start_timer(4 if target == "four" else 5)
                    cooldown_until = now + cooldown_sec
                    smoothed.fill(0)
                    initialized = False
                    for name in targets:
                        hits[name] = 0
                    if args.once:
                        stop_event.set()
                    break


if __name__ == "__main__":
    main()
