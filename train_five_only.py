import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kws_model import KWSConvNet


LABELS = ["background", "five"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_list(root: str, filename: str) -> List[str]:
    filepath = os.path.join(root, filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return [os.path.join(root, line.strip()) for line in file]


def build_walker(root: str, subset: str) -> List[str]:
    if subset == "validation":
        return load_list(root, "validation_list.txt")
    if subset == "testing":
        return load_list(root, "testing_list.txt")
    if subset == "training":
        excludes = set(load_list(root, "validation_list.txt") + load_list(root, "testing_list.txt"))
        wavs: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            if os.path.basename(dirpath) == "_background_noise_":
                continue
            for filename in filenames:
                if filename.endswith(".wav"):
                    path = os.path.join(dirpath, filename)
                    if path not in excludes:
                        wavs.append(path)
        return wavs
    raise ValueError(f"Unknown subset: {subset}")


class FiveOnlyDataset(Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        sample_rate: int = 16000,
        background_ratio: float = 1.0,
        silence_ratio: float = 0.5,
        seed: int = 1337,
    ):
        super().__init__()
        random.seed(seed)
        self.sample_rate = sample_rate
        self._path = root
        self._walker = build_walker(root, subset)

        self.examples: List[Tuple[str, int]] = []
        for path in self._walker:
            label = os.path.basename(os.path.dirname(path))
            if label == "five":
                self.examples.append((path, 1))

        self.background_samples = self._build_background_samples(background_ratio)
        self.silence_samples = self._build_silence_samples(silence_ratio)

    def _build_background_samples(self, background_ratio: float) -> List[Tuple[str, int]]:
        background_dir = os.path.join(self._path, "_background_noise_")
        if not os.path.isdir(background_dir):
            return []

        noise_files = [
            os.path.join(background_dir, filename)
            for filename in os.listdir(background_dir)
            if filename.endswith(".wav")
        ]
        if not noise_files:
            return []

        target_count = int(len(self.examples) * background_ratio)
        samples = []
        for _ in range(max(target_count, len(noise_files))):
            noise_path = random.choice(noise_files)
            samples.append((noise_path, 0))
        return samples

    def _build_silence_samples(self, silence_ratio: float) -> List[Tuple[None, int]]:
        if silence_ratio <= 0:
            return []
        target_count = int(len(self.examples) * silence_ratio)
        return [(None, 0) for _ in range(max(target_count, 1))]

    def __len__(self) -> int:
        return len(self.examples) + len(self.background_samples) + len(self.silence_samples)

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
            path, label = self.examples[idx]
            waveform = self._load_audio(path)
        elif idx < len(self.examples) + len(self.background_samples):
            path, label = self.background_samples[idx - len(self.examples)]
            waveform = self._load_audio(path)
        else:
            _, label = self.silence_samples[idx - len(self.examples) - len(self.background_samples)]
            waveform = torch.zeros(1, self.sample_rate)
        waveform = self._crop_or_pad(waveform)
        return waveform, label


def train(
    data_root: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    silence_ratio: float,
) -> None:
    train_dataset = FiveOnlyDataset(
        root=data_root, subset="training", silence_ratio=silence_ratio
    )
    val_dataset = FiveOnlyDataset(root=data_root, subset="validation", silence_ratio=0.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 64,
            "center": False,
        },
    ).to(device)

    model = KWSConvNet(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def to_mfcc(waveforms: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = mfcc(waveforms)
        if features.dim() == 3:
            features = features.unsqueeze(1)
        return features

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            features = to_mfcc(waveforms)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)

        model.eval()
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                features = to_mfcc(waveforms)
                outputs = model(features)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_count += labels.size(0)

        val_acc = val_correct / max(val_count, 1)
        print(
            f"epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": LABELS,
        },
        output_path,
    )
    print(f"saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KWS model for five vs background.")
    parser.add_argument("--data-root", default="data", help="Dataset root directory.")
    parser.add_argument("--output", default="models/kws_five_only.pt", help="Output path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--silence-ratio", type=float, default=0.8)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    train(
        data_root=args.data_root,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        silence_ratio=args.silence_ratio,
    )


if __name__ == "__main__":
    main()
