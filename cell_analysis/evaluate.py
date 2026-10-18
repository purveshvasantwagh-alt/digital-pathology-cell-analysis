import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from cell_analysis.data import make_dataloaders
from cell_analysis.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/resnet18_best.pt"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    _, _, test_loader, class_names = make_dataloaders(
        train_dir=args.images_root / "TRAIN",
        test_dir=args.images_root / "TEST",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if checkpoint["class_names"] != class_names:
        raise ValueError("Checkpoint class names do not match the dataset.")

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

    label_ids = list(range(len(class_names)))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_ids,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_data = classification_report(
        true_labels,
        predicted_labels,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_ids,
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    (args.report_dir / "classification_report.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    with (args.report_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])

    print(f"Test images evaluated: {len(true_labels)}")
    print(report)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)
    print("Reports saved in:", args.report_dir)


if __name__ == "__main__":
    main()