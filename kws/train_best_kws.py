#!/usr/bin/env python3
"""Train a robust four/five KWS model on Speech Commands v0.02."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from kws_best import (
    FRONTEND_CONFIG,
    LABELS,
    TARGET_LABELS,
    LogMelFrontend,
    RobustKWSNet,
    RobustSpeechCommands,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    frontend: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    confusion = torch.zeros(len(LABELS), len(LABELS), dtype=torch.long)
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(frontend(waveforms))
        total_loss += criterion(logits, labels_device).item() * labels.numel()
        total += labels.numel()
        probs = logits.softmax(dim=1).cpu()
        predictions = probs.argmax(dim=1)
        for actual, predicted in zip(labels, predictions):
            confusion[int(actual), int(predicted)] += 1
        all_probs.append(probs)
        all_labels.append(labels)

    return (
        total_loss / max(total, 1),
        confusion,
        torch.cat(all_probs),
        torch.cat(all_labels),
    )


def print_metrics(name: str, loss: float, confusion: torch.Tensor) -> float:
    print(f"\n{name}: loss={loss:.4f}")
    print("actual\\pred " + " ".join(f"{label:>10}" for label in LABELS))
    for index, label in enumerate(LABELS):
        values = " ".join(f"{int(value):10d}" for value in confusion[index])
        print(f"{label:>11} {values}")

    recalls = []
    for index, label in enumerate(LABELS):
        true_positive = confusion[index, index].item()
        recall = true_positive / max(confusion[index].sum().item(), 1)
        precision = true_positive / max(confusion[:, index].sum().item(), 1)
        recalls.append(recall)
        print(f"  {label:>10}: precision={precision:.4f} recall={recall:.4f}")
    macro_recall = float(np.mean(recalls))
    print(f"  macro recall={macro_recall:.4f}")
    return macro_recall


def calibrate_thresholds(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    max_false_positive_rate: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    print("\nThreshold calibration:")
    for target in TARGET_LABELS:
        index = LABELS.index(target)
        scores = probabilities[:, index]
        positive = labels == index
        negative = ~positive
        selected: tuple[float, float, float] | None = None
        fallback: tuple[float, float, float, float] | None = None

        for threshold in torch.linspace(0.30, 0.995, 140):
            detected = scores >= threshold
            true_positive = int((detected & positive).sum())
            false_positive = int((detected & negative).sum())
            false_negative = int((~detected & positive).sum())
            recall = true_positive / max(int(positive.sum()), 1)
            false_positive_rate = false_positive / max(int(negative.sum()), 1)
            precision = true_positive / max(true_positive + false_positive, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            candidate = (f1, float(threshold), recall, false_positive_rate)
            if fallback is None or candidate[0] > fallback[0]:
                fallback = candidate
            if false_positive_rate <= max_false_positive_rate:
                if selected is None or recall > selected[1]:
                    selected = (float(threshold), recall, false_positive_rate)

        if selected is None:
            assert fallback is not None
            _, threshold, recall, false_positive_rate = fallback
        else:
            threshold, recall, false_positive_rate = selected
        thresholds[target] = round(threshold, 4)
        print(
            f"  {target}: threshold={threshold:.3f} "
            f"recall={recall:.4f} sample_fpr={false_positive_rate:.6f}"
        )
    return thresholds


KWS_DIR = Path(__file__).resolve().parent
REPO_ROOT = KWS_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--output", default=str(KWS_DIR / "models" / "kws_best_hard.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epoch-samples", type=int, default=40_000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-sample-fpr", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    print(f"device={device}")

    train_set = RobustSpeechCommands(
        args.data_root, "training", background_examples=4_000, augment=True
    )
    validation_set = RobustSpeechCommands(
        args.data_root, "validation", background_examples=1_500, augment=False
    )
    test_set = RobustSpeechCommands(
        args.data_root, "testing", background_examples=1_500, augment=False
    )
    print(f"labels={LABELS}")
    print(f"train class counts={train_set.class_counts}")
    print(f"validation class counts={validation_set.class_counts}")

    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_set.sample_weights,
        num_samples=args.epoch_samples,
        replacement=True,
        generator=generator,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_seed,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_set, sampler=sampler, **loader_kwargs)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    frontend = LogMelFrontend().to(device)
    model = RobustKWSNet(len(LABELS)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model parameters={parameter_count:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(frontend(waveforms))
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += labels.numel()

        validation_loss, confusion, _, _ = evaluate(
            model, frontend, validation_loader, device
        )
        score = print_metrics(f"epoch {epoch}/{args.epochs} validation", validation_loss, confusion)
        print(
            f"train loss={running_loss / max(seen, 1):.4f} "
            f"accuracy={correct / max(seen, 1):.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_loss, validation_confusion, validation_probs, validation_labels = evaluate(
        model, frontend, validation_loader, device
    )
    print_metrics("best validation", validation_loss, validation_confusion)
    thresholds = calibrate_thresholds(
        validation_probs, validation_labels, args.max_sample_fpr
    )
    test_loss, test_confusion, _, _ = evaluate(model, frontend, test_loader, device)
    test_macro_recall = print_metrics("test", test_loss, test_confusion)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_name": "RobustKWSNet",
            "model_state": best_state,
            "labels": LABELS,
            "target_labels": list(TARGET_LABELS),
            "frontend": FRONTEND_CONFIG,
            "thresholds": thresholds,
            "runtime": {
                "score_ema": 0.55,
                "margin": 0.18,
                "consecutive_hits": 3,
                "hop_seconds": 0.10,
                "cooldown_seconds": 2.0,
            },
            "training": {
                "dataset": "Google Speech Commands v0.02",
                "epochs": args.epochs,
                "epoch_samples": args.epoch_samples,
                "seed": args.seed,
                "best_validation_macro_recall": best_score,
                "test_macro_recall": test_macro_recall,
            },
        },
        output,
    )
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
