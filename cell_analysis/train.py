import argparse
from pathlib import Path

import torch
from torch import nn

from cell_analysis.data import make_dataloaders
from cell_analysis.model import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
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

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/resnet18_best.pt"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, _test_loader, class_names = make_dataloaders(
        train_dir=args.images_root / "TRAIN",
        test_dir=args.images_root / "TEST",
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=0.001)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = -1.0

    print("Device:", device)
    print("Classes:", class_names)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f}, train accuracy={train_accuracy:.4f} | "
            f"validation loss={val_loss:.4f}, "
            f"validation accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "validation_accuracy": val_accuracy,
                },
                args.checkpoint,
            )
            print("Saved best checkpoint to:", args.checkpoint)


if __name__ == "__main__":
    main()