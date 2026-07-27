import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kws_model import KWSConvNet


LABELS = ["background", "unknown", "four", "five"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


class LibrosaKWSDataset(Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        sample_rate: int = 16000,
        window_sec: float = 1.0,
        n_mfcc: int = 13,
        unknown_ratio: float = 1.0,
        background_ratio: float = 1.0,
        silence_ratio: float = 0.5,
        seed: int = 1337,
    ):
        super().__init__()
        random.seed(seed)
        self._path = root
        self.sample_rate = sample_rate
        self.window_len = int(sample_rate * window_sec)
        self.n_mfcc = n_mfcc
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64, "center": False},
        )

        walker = build_walker(root, subset)
        self.four_files = [p for p in walker if os.path.basename(os.path.dirname(p)) == "four"]
        self.five_files = [p for p in walker if os.path.basename(os.path.dirname(p)) == "five"]
        self.unknown_files = [
            p
            for p in walker
            if os.path.basename(os.path.dirname(p)) not in ("four", "five")
        ]

        self.background_files = self._list_background()
        self.samples: List[Tuple[Optional[str], str]] = []

        self.samples += [(p, "four") for p in self.four_files]
        self.samples += [(p, "five") for p in self.five_files]

        unknown_target = int(len(self.samples) * unknown_ratio)
        if self.unknown_files:
            self.samples += [(random.choice(self.unknown_files), "unknown") for _ in range(unknown_target)]

        background_target = int(len(self.samples) * background_ratio)
        if self.background_files:
            self.samples += [(random.choice(self.background_files), "background") for _ in range(background_target)]

        silence_target = int(len(self.samples) * silence_ratio)
        self.samples += [(None, "background") for _ in range(max(silence_target, 1))]

        random.shuffle(self.samples)

    def _list_background(self) -> List[str]:
        background_dir = os.path.join(self._path, "_background_noise_")
        if not os.path.isdir(background_dir):
            return []
        return [
            os.path.join(background_dir, filename)
            for filename in os.listdir(background_dir)
            if filename.endswith(".wav")
        ]

    def _load_audio(self, path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        audio = waveform.mean(dim=0).numpy()
        if audio.size > self.window_len:
            start = random.randint(0, audio.size - self.window_len)
            audio = audio[start : start + self.window_len]
        elif audio.size < self.window_len:
            audio = np.pad(audio, (0, self.window_len - audio.size))
        return audio.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        if path is None:
            audio = np.zeros(self.window_len, dtype=np.float32)
        else:
            audio = self._load_audio(path)
        features = self.mfcc(torch.tensor(audio).unsqueeze(0))
        return features, LABELS.index(label)


def train(
    data_root: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    unknown_ratio: float,
    silence_ratio: float,
    num_workers: int,
) -> None:
    train_dataset = LibrosaKWSDataset(
        root=data_root,
        subset="training",
        unknown_ratio=unknown_ratio,
        silence_ratio=silence_ratio,
    )
    val_dataset = LibrosaKWSDataset(
        root=data_root,
        subset="validation",
        unknown_ratio=0.0,
        silence_ratio=0.0,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = KWSConvNet(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

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
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
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
    parser = argparse.ArgumentParser(description="Train KWS model with unknown class (librosa MFCC).")
    parser.add_argument("--data-root", default="data", help="Dataset root directory.")
    parser.add_argument("--output", default="models/kws_librosa_unknown.pt", help="Output path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--unknown-ratio", type=float, default=1.0)
    parser.add_argument("--silence-ratio", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
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
        unknown_ratio=args.unknown_ratio,
        silence_ratio=args.silence_ratio,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
