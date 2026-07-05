import os
import random
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


LABELS = ["background", "four", "five"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}


class SubsetSpeechCommands:
    def __init__(self, subset: str, root: str):
        self._path = root

        def load_list(filename: str) -> List[str]:
            filepath = os.path.join(self._path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                return [os.path.join(self._path, line.strip()) for line in file]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            all_wavs: List[str] = []
            for dirpath, _, filenames in os.walk(self._path):
                if os.path.basename(dirpath) == "_background_noise_":
                    continue
                for filename in filenames:
                    if filename.endswith(".wav"):
                        all_wavs.append(os.path.join(dirpath, filename))
            self._walker = [w for w in all_wavs if w not in excludes]
        else:
            raise ValueError(f"Unknown subset: {subset}")


class KWSDataset(Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        sample_rate: int = 16000,
        background_ratio: float = 1.0,
        seed: int = 1337,
    ):
        super().__init__()
        random.seed(seed)
        self.sample_rate = sample_rate
        self.speech = SubsetSpeechCommands(subset=subset, root=root)

        self.examples: List[Tuple[str, str, int]] = []
        for path in self.speech._walker:
            label = os.path.basename(os.path.dirname(path))
            if label in ("four", "five"):
                self.examples.append((path, label, -1))

        self.background_samples = self._build_background_samples(background_ratio)

    def _build_background_samples(self, background_ratio: float) -> List[Tuple[str, str, int]]:
        background_dir = os.path.join(self.speech._path, "_background_noise_")
        if not os.path.isdir(background_dir):
            return []

        noise_files = [
            os.path.join(background_dir, filename)
            for filename in os.listdir(background_dir)
            if filename.endswith(".wav")
        ]
        if not noise_files:
            return []

        target_count = int(len(self.examples) * background_ratio / 2)
        samples = []
        for _ in range(max(target_count, len(noise_files))):
            noise_path = random.choice(noise_files)
            samples.append((noise_path, "background", -1))
        return samples

    def __len__(self) -> int:
        return len(self.examples) + len(self.background_samples)

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        target_len = self.sample_rate
        if waveform.size(1) > target_len:
            start = random.randint(0, waveform.size(1) - target_len)
            waveform = waveform[:, start : start + target_len]
        elif waveform.size(1) < target_len:
            pad = target_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < len(self.examples):
            path, label, _ = self.examples[idx]
            waveform = self._load_audio(path)
        else:
            path, label, _ = self.background_samples[idx - len(self.examples)]
            waveform = self._load_audio(path)
        waveform = self._crop_or_pad(waveform)
        return waveform, LABEL_TO_INDEX[label]
