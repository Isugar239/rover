import argparse
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import sounddevice as sd
import torch
import torchaudio
import tkinter as tk
from tkinter import ttk

from kws_model import KWSConvNet


def load_model(model_path: str, device: torch.device) -> tuple[KWSConvNet, list[str]]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    labels = checkpoint.get("labels", ["background", "four", "five"])
    model = KWSConvNet(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, labels


class KWSApp:
    def __init__(self, root: tk.Tk, model_path: str) -> None:
        self.root = root
        self.root.title("KWS GUI")
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.labels = load_model(model_path, self.device)

        self.sample_sec = tk.DoubleVar(value=1.5)
        self.hop_sec = tk.DoubleVar(value=0.25)
        self.min_rms = tk.DoubleVar(value=0.01)
        self.threshold = tk.DoubleVar(value=0.8)
        self.running = False

        self.device_var = tk.StringVar()
        self.device_index: Optional[int] = None
        self.devices = self._list_devices()

        self._build_ui()

        self.audio_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.work_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.latest_probs: dict[str, float] = {label: 0.0 for label in self.labels}
        self.latest_rms: float = 0.0

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 64,
                "center": False,
            },
        ).to(self.device)

        self._ui_tick()

    def _list_devices(self) -> list[tuple[int, str, float]]:
        items = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                items.append((i, dev["name"], float(dev["default_samplerate"])))
        return items

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        device_frame = ttk.LabelFrame(self.root, text="Устройство ввода")
        device_frame.pack(fill="x", **pad)

        device_names = [f"[{i}] {name} ({sr:.0f} Hz)" for i, name, sr in self.devices]
        self.device_var.set(device_names[0] if device_names else "")
        device_menu = ttk.OptionMenu(device_frame, self.device_var, self.device_var.get(), *device_names)
        device_menu.pack(fill="x", **pad)

        params_frame = ttk.LabelFrame(self.root, text="Параметры")
        params_frame.pack(fill="x", **pad)

        self._add_slider(params_frame, "Длина окна, сек", self.sample_sec, 0.5, 2.5)
        self._add_slider(params_frame, "Шаг, сек", self.hop_sec, 0.1, 1.0)
        self._add_slider(params_frame, "min RMS", self.min_rms, 0.000, 0.05)
        self._add_slider(params_frame, "threshold", self.threshold, 0.3, 0.99)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", **pad)

        self.start_button = ttk.Button(control_frame, text="Старт", command=self.start)
        self.start_button.pack(side="left", **pad)
        self.stop_button = ttk.Button(control_frame, text="Стоп", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", **pad)

        self.status_label = ttk.Label(self.root, text="Статус: остановлено")
        self.status_label.pack(fill="x", **pad)

        self.prob_text = tk.Text(self.root, height=6, width=60, state="disabled")
        self.prob_text.pack(fill="both", expand=True, **pad)

    def _add_slider(
        self,
        frame: ttk.LabelFrame,
        label: str,
        variable: tk.DoubleVar,
        min_val: float,
        max_val: float,
    ) -> None:
        row = ttk.Frame(frame)
        row.pack(fill="x", padx=6, pady=2)
        ttk.Label(row, text=label).pack(side="left")
        scale = ttk.Scale(row, from_=min_val, to=max_val, variable=variable)
        scale.pack(side="left", fill="x", expand=True, padx=6)
        value_label = ttk.Label(row, textvariable=variable)
        value_label.pack(side="right")

    def _parse_device(self) -> Optional[int]:
        if not self.device_var.get():
            return None
        label = self.device_var.get()
        try:
            idx = int(label.split("]")[0].strip("[").strip())
        except ValueError:
            return None
        return idx

    def start(self) -> None:
        if self.running:
            return
        self.device_index = self._parse_device()
        if self.device_index is None:
            self.status_label.config(text="Статус: нет устройства")
            return
        self.running = True
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Статус: слушаю")

        self.audio_thread = threading.Thread(target=self._run_audio_loop, daemon=True)
        self.audio_thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Статус: остановлено")

    def _run_audio_loop(self) -> None:
        device_info = sd.query_devices(self.device_index, "input")
        device_sr = int(device_info["default_samplerate"])

        buffer_len = int(device_sr * self.sample_sec.get())
        hop_len = int(device_sr * self.hop_sec.get())
        audio_buffer: Deque[float] = deque(maxlen=buffer_len)
        pending_samples = 0

        def callback(indata, frames, time_info, status) -> None:
            nonlocal pending_samples
            if status:
                self.status_label.config(text=f"Статус: {status}")
            mono = indata[:, 0].astype(np.float32)
            audio_buffer.extend(mono.tolist())
            pending_samples += frames
            while len(audio_buffer) == buffer_len and pending_samples >= hop_len:
                chunk = np.array(audio_buffer, dtype=np.float32)
                self.work_queue.put(chunk)
                pending_samples -= hop_len

        with sd.InputStream(
            samplerate=device_sr,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=0,
            latency="high",
            device=self.device_index,
        ):
            while not self.stop_event.is_set():
                try:
                    chunk = self.work_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-8)
                self.latest_rms = rms
                if rms < self.min_rms.get():
                    self.latest_probs = {label: 0.0 for label in self.labels}
                    continue
                waveform = torch.tensor(chunk).unsqueeze(0).to(self.device)
                if device_sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, device_sr, 16000)
                with torch.no_grad():
                    feats = self.mfcc(waveform).unsqueeze(1)
                    logits = self.model(feats)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                self.latest_probs = {label: float(prob) for label, prob in zip(self.labels, probs)}

        self.stop_event.set()
        self.running = False

    def _ui_tick(self) -> None:
        if self.running:
            parts = [f"{label}: {self.latest_probs.get(label, 0.0):.3f}" for label in self.labels]
            text = f"RMS={self.latest_rms:.6f} | " + " ".join(parts)
        else:
            text = "RMS=0.000000 | " + " ".join([f"{label}: 0.000" for label in self.labels])
        self.prob_text.config(state="normal")
        self.prob_text.delete("1.0", tk.END)
        self.prob_text.insert(tk.END, text)
        self.prob_text.config(state="disabled")
        self.root.after(200, self._ui_tick)


def main() -> None:
    old_kws = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="KWS GUI")
    parser.add_argument("--model", default=str(old_kws / "models" / "kws_five_only.pt"))
    args = parser.parse_args()

    root = tk.Tk()
    app = KWSApp(root, args.model)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()


if __name__ == "__main__":
    main()
