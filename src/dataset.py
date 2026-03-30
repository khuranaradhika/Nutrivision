"""
dataset.py — Data loading, transforms, and Food-101 dataset utilities.
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import Food101

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(img_size=224):
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transform(img_size=224):
    """Test/validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_food101(data_dir='./data', batch_size=32, num_workers=2,
                 val_split=0.1, seed=42):
    """
    Download and prepare Food-101 dataset with train/val/test splits.

    Args:
        data_dir: root directory for dataset download
        batch_size: batch size for data loaders
        num_workers: number of worker processes for data loading
        val_split: fraction of training data to use for validation
        seed: random seed for reproducible val split

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    # Download datasets
    train_dataset_full = Food101(
        root=data_dir, split='train', transform=train_transform, download=True
    )
    test_dataset = Food101(
        root=data_dir, split='test', transform=test_transform, download=True
    )

    class_names = train_dataset_full.classes
    num_classes = len(class_names)

    # Train/val split
    train_size = int((1 - val_split) * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f'Food-101 loaded: {num_classes} classes')
    print(f'  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')

    return train_loader, val_loader, test_loader, class_names


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization for visualization."""
    t = tensor.clone()
    for i in range(3):
        t[i] = t[i] * std[i] + mean[i]
    return torch.clamp(t, 0, 1)
