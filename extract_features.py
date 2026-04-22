# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:52:45 2026

@author: admin
"""

import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    val_size: int = 5000,
):
    # No augmentation for feature extraction
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    full_train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=test_transform,
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # validation/test 都不增强
    val_subset.dataset = copy.deepcopy(full_train_dataset)
    val_subset.dataset.transform = test_transform

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_subset,
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

    return train_loader, val_loader, test_loader


def build_model(model_name: str, num_classes: int = 100, pretrained: bool = False):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # original resnet18, no stem changes

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

        # IMPORTANT: must match the improved training-time architecture
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

    else:
        raise ValueError("model_name must be 'resnet18' or 'resnet50'")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        x = self.features(x)              # [B, C, 1, 1]
        x = torch.flatten(x, 1)          # [B, C]
        return x


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = model(images)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def load_trained_backbone(model_name: str, checkpoint_path: str, device):
    model = build_model(model_name=model_name, num_classes=100)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    feature_extractor = ResNetFeatureExtractor(model).to(device)
    return feature_extractor


def save_features(save_dir, prefix, split_name, features, labels):
    os.makedirs(save_dir, exist_ok=True)

    feat_path = os.path.join(save_dir, f"{prefix}_{split_name}_features.npy")
    label_path = os.path.join(save_dir, f"{prefix}_{split_name}_labels.npy")

    np.save(feat_path, features)
    np.save(label_path, labels)

    print(f"Saved features: {feat_path} | shape={features.shape}")
    print(f"Saved labels:   {label_path} | shape={labels.shape}")


def process_one_model(
    model_name: str,
    checkpoint_path: str,
    train_loader,
    val_loader,
    test_loader,
    device,
    save_dir: str = "./features",
):
    print(f"\n========== Extracting features for {model_name} ==========")
    print(f"Loading checkpoint: {checkpoint_path}")

    feature_extractor = load_trained_backbone(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    val_features, val_labels = extract_features(feature_extractor, val_loader, device)
    test_features, test_labels = extract_features(feature_extractor, test_loader, device)

    save_features(save_dir, model_name, "train", train_features, train_labels)
    save_features(save_dir, model_name, "val", val_features, val_labels)
    save_features(save_dir, model_name, "test", test_features, test_labels)

    print(f"{model_name} done.")
    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape:   {val_features.shape}")
    print(f"Test features shape:  {test_features.shape}")


def main():
    seed = 42
    data_dir = "./data"
    batch_size = 256
    num_workers = 2
    val_size = 5000

    checkpoints = {
        "resnet18": "./checkpoints/best_resnet18.pth",
        "resnet50": "./checkpoints/best_resnet50.pth",
    }

    set_seed(seed)
    device = get_device()
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_size=val_size,
    )

    for model_name, checkpoint_path in checkpoints.items():
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found, skipping {model_name}: {checkpoint_path}")
            continue

        process_one_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            save_dir="./features",
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()