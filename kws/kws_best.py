"""Shared model, audio frontend, and dataset for robust keyword spotting."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE
LABELS = ["background", "unknown", "four", "five"]
TARGET_LABELS = ("four", "five")
HARD_NEGATIVE_WEIGHTS = {
    "forward": 8.0,
    "follow": 5.0,
    "right": 5.0,
    "nine": 3.0,
    "three": 2.0,
    "off": 2.0,
}

FRONTEND_CONFIG = {
    "sample_rate": SAMPLE_RATE,
    "n_fft": 400,
    "win_length": 400,
    "hop_length": 160,
    "n_mels": 40,
    "center": False,
}


class LogMelFrontend(nn.Module):
    """The exact same normalized log-mel frontend for training and runtime."""

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=FRONTEND_CONFIG["sample_rate"],
            n_fft=FRONTEND_CONFIG["n_fft"],
            win_length=FRONTEND_CONFIG["win_length"],
            hop_length=FRONTEND_CONFIG["hop_length"],
            n_mels=FRONTEND_CONFIG["n_mels"],
            center=FRONTEND_CONFIG["center"],
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = torch.log(self.mel(waveform).clamp_min(1e-6))
        mean = features.mean(dim=(-2, -1), keepdim=True)
        std = features.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        features = (features - mean) / std
        if features.ndim == 3:
            features = features.unsqueeze(1)
        return features


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RobustKWSNet(nn.Module):
    """Compact DS-CNN, substantially stronger than the original 24k model."""

    def __init__(self, num_classes: int = len(LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, 64),
            DepthwiseSeparableBlock(64, 96, stride=2),
            DepthwiseSeparableBlock(96, 128),
            DepthwiseSeparableBlock(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.20), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[RobustKWSNet, LogMelFrontend, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError(f"{path} is not a robust KWS v2 checkpoint")
    labels = checkpoint["labels"]
    model = RobustKWSNet(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frontend = LogMelFrontend().to(device).eval()
    return model, frontend, checkpoint


def _load_split(root: Path, filename: str) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in (root / filename).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


class RobustSpeechCommands(Dataset):
    """Speech Commands v0.02 with unknown speech and realistic online augmentation."""

    def __init__(
        self,
        root: str | Path,
        subset: str,
        background_examples: int = 4_000,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.subset = subset
        self.augment = augment
        validation = _load_split(self.root, "validation_list.txt")
        testing = _load_split(self.root, "testing_list.txt")

        wavs = sorted(
            p
            for p in self.root.glob("*/*.wav")
            if p.parent.name != "_background_noise_"
        )
        if subset == "training":
            selected = [
                p
                for p in wavs
                if p.relative_to(self.root).as_posix() not in validation
                and p.relative_to(self.root).as_posix() not in testing
            ]
        elif subset == "validation":
            selected = [p for p in wavs if p.relative_to(self.root).as_posix() in validation]
        elif subset == "testing":
            selected = [p for p in wavs if p.relative_to(self.root).as_posix() in testing]
        else:
            raise ValueError(f"unknown subset: {subset}")

        self.examples: list[tuple[Optional[Path], int]] = []
        for path in selected:
            word = path.parent.name
            label = LABELS.index(word) if word in TARGET_LABELS else LABELS.index("unknown")
            self.examples.append((path, label))

        self.examples.extend([(None, LABELS.index("background"))] * background_examples)
        self.noise_files = sorted((self.root / "_background_noise_").glob("*.wav"))
        self._noise_cache: dict[Path, torch.Tensor] = {}
        self.class_counts = torch.bincount(
            torch.tensor([label for _, label in self.examples]), minlength=len(LABELS)
        ).tolist()

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def sample_weights(self) -> torch.Tensor:
        # Real audio is mostly non-keyword speech. Keep that prior while still
        # showing both targets often, and mine phonetically confusing words.
        desired_class_mass = [0.15, 0.45, 0.20, 0.20]
        multipliers = []
        for path, label in self.examples:
            if label == LABELS.index("unknown") and path is not None:
                multipliers.append(HARD_NEGATIVE_WEIGHTS.get(path.parent.name, 1.0))
            else:
                multipliers.append(1.0)
        multiplier_sums = [0.0] * len(LABELS)
        for (_, label), multiplier in zip(self.examples, multipliers):
            multiplier_sums[label] += multiplier
        return torch.tensor(
            [
                desired_class_mass[label] * multiplier / max(multiplier_sums[label], 1.0)
                for (_, label), multiplier in zip(self.examples, multipliers)
            ],
            dtype=torch.double,
        )

    @staticmethod
    def _mono_resampled(path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        return waveform

    def _fixed_window(self, waveform: torch.Tensor, random_position: bool) -> torch.Tensor:
        length = waveform.numel()
        if length > WINDOW_SAMPLES:
            start = (
                random.randint(0, length - WINDOW_SAMPLES)
                if random_position
                else (length - WINDOW_SAMPLES) // 2
            )
            return waveform[start : start + WINDOW_SAMPLES]
        if length < WINDOW_SAMPLES:
            total_pad = WINDOW_SAMPLES - length
            left = random.randint(0, total_pad) if random_position else total_pad // 2
            return F.pad(waveform, (left, total_pad - left))
        return waveform

    def _noise(self) -> torch.Tensor:
        if not self.noise_files:
            return torch.randn(WINDOW_SAMPLES) * 0.02
        path = random.choice(self.noise_files)
        if path not in self._noise_cache:
            self._noise_cache[path] = self._mono_resampled(path)
        noise = self._noise_cache[path]
        if noise.numel() < WINDOW_SAMPLES:
            repeats = math.ceil(WINDOW_SAMPLES / max(noise.numel(), 1))
            noise = noise.repeat(repeats)
        start = random.randint(0, noise.numel() - WINDOW_SAMPLES)
        return noise[start : start + WINDOW_SAMPLES].clone()

    def _background(self, index: int) -> torch.Tensor:
        if not self.augment:
            if index % 5 == 0 or not self.noise_files:
                return torch.zeros(WINDOW_SAMPLES)
            path = self.noise_files[index % len(self.noise_files)]
            noise = self._mono_resampled(path)
            if noise.numel() < WINDOW_SAMPLES:
                noise = noise.repeat(math.ceil(WINDOW_SAMPLES / max(noise.numel(), 1)))
            start = (index * 7_919) % (noise.numel() - WINDOW_SAMPLES + 1)
            return (noise[start : start + WINDOW_SAMPLES] * 0.20).clamp(-1.0, 1.0)
        if random.random() < 0.18:
            return torch.zeros(WINDOW_SAMPLES)
        noise = self._noise()
        gain = 10 ** random.uniform(-1.3, -0.15)
        return (noise * gain).clamp(-1.0, 1.0)

    @staticmethod
    def _reverb(waveform: torch.Tensor) -> torch.Tensor:
        result = waveform.clone()
        for _ in range(random.randint(2, 5)):
            delay = random.randint(160, 2_400)
            gain = random.uniform(0.05, 0.28) * math.exp(-delay / 2_000)
            result[delay:] += waveform[:-delay] * gain
        return result / result.abs().max().clamp_min(1.0)

    def _augment_speech(self, waveform: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-1_600, 1_600)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[:shift] = 0
        elif shift < 0:
            waveform[shift:] = 0

        waveform = waveform * (10 ** random.uniform(-0.45, 0.20))
        if random.random() < 0.25:
            waveform = self._reverb(waveform)

        if random.random() < 0.82:
            noise = self._noise()
            speech_rms = waveform.square().mean().sqrt().clamp_min(1e-4)
            noise_rms = noise.square().mean().sqrt().clamp_min(1e-4)
            snr_db = random.uniform(-5.0, 20.0)
            noise_gain = speech_rms / (noise_rms * (10 ** (snr_db / 20)))
            waveform = waveform + noise * noise_gain

        if random.random() < 0.12:
            clip_level = random.uniform(0.35, 0.90)
            waveform = waveform.clamp(-clip_level, clip_level) / clip_level
        return waveform.clamp(-1.0, 1.0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.examples[index]
        if path is None:
            waveform = self._background(index)
        else:
            waveform = self._mono_resampled(path)
            waveform = self._fixed_window(waveform, random_position=self.augment)
            if self.augment:
                waveform = self._augment_speech(waveform)
        return waveform, label
