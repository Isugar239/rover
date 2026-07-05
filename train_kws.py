import argparse
import os
import random
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from kws_data import KWSDataset, LABELS
from kws_model import KWSConvNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_root: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    train_dataset = KWSDataset(root=data_root, subset="training")
    val_dataset = KWSDataset(root=data_root, subset="validation")

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
    parser = argparse.ArgumentParser(description="Train KWS model for four/five/background.")
    parser.add_argument("--data-root", default="data", help="Dataset root directory.")
    parser.add_argument("--output", default="models/kws_cnn.pt", help="Output path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
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
    )


if __name__ == "__main__":
    main()
