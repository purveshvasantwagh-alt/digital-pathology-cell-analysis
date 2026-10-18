import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cell_analysis.data import make_dataloaders
from cell_analysis.model import build_model


def set_seed(seed: int) -> None:
    """Set random seeds for more repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or validation epoch and return loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("The data loader produced no images.")

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 blood-cell image classifier."
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Folder containing TRAIN and TEST class subfolders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable data splitting and training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data-loading worker processes.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/resnet18_best.pt"),
        help="Where to save the best validation checkpoint.",
    )
    parser.add_argument(
        "--fine-tune-layer4",
        action="store_true",
        help="Train ResNet18 layer4 as well as the final classifier layer.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    train_dir = args.images_root / "TRAIN"
    test_dir = args.images_root / "TEST"

    if not train_dir.is_dir():
        parser.error(f"Training folder not found: {train_dir}")
    if not test_dir.is_dir():
        parser.error(f"Test folder not found: {test_dir}")

    set_seed(args.seed)

    train_loader, val_loader, _test_loader, class_names = make_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names))

    if args.fine_tune_layer4:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.fine_tune_layer4:
        optimizer = torch.optim.Adam(
            [
                {"params": model.layer4.parameters(), "lr": 0.0001},
                {"params": model.fc.parameters(), "lr": 0.001},
            ]
        )
    else:
        optimizer = torch.optim.Adam(
            model.fc.parameters(),
            lr=0.001,
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = -1.0

    print("Device:", device)
    print("Classes:", class_names)
    print("Fine-tuning layer4:", args.fine_tune_layer4)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f}, "
            f"train accuracy={train_accuracy:.4f} | "
            f"validation loss={val_loss:.4f}, "
            f"validation accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "validation_accuracy": val_accuracy,
                "fine_tune_layer4": args.fine_tune_layer4,
                "seed": args.seed,
                "batch_size": args.batch_size,
            }
            torch.save(checkpoint_data, args.checkpoint)

            print("Saved best checkpoint to:", args.checkpoint)

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()