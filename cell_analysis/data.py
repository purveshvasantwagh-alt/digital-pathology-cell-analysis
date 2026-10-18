from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights


def make_dataloaders(
    train_dir: str | Path,
    test_dir: str | Path,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
):
    """Create stratified train/validation loaders and a held-out test loader."""

    weights = ResNet18_Weights.DEFAULT

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = weights.transforms()

    train_augmented = datasets.ImageFolder(
        str(train_dir), transform=train_transform
    )
    train_for_validation = datasets.ImageFolder(
        str(train_dir), transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        str(test_dir), transform=eval_transform
    )

    if train_augmented.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train and test folders have different class mappings.")

    indices = list(range(len(train_augmented)))
    labels = train_augmented.targets

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        Subset(train_augmented, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(train_for_validation, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_names = train_augmented.classes
    return train_loader, val_loader, test_loader, class_names