import os
import copy
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import sys

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

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
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

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
        transform=train_transform,
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

    # validation set should not use augmentation
    val_subset.dataset = copy.deepcopy(full_train_dataset)
    val_subset.dataset.transform = test_transform

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
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

        # keep original stem for resnet18
        # no changes here

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

        # improved CIFAR-style stem for 32x32 images
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

    else:
        raise ValueError("model_name must be 'resnet18' or 'resnet50'")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    return run_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
    )


def get_training_config(model_name: str):
    """
    resnet18 -> original config
    resnet50 -> improved config
    """
    if model_name == "resnet18":
        return {
            "epochs": 30,
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "milestones": [15, 22, 27],
            "gamma": 0.2,
            "label_smoothing": 0.0,
        }

    elif model_name == "resnet50":
        return {
            "epochs": 50,
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "milestones": [25, 38, 45],
            "gamma": 0.2,
            "label_smoothing": 0.1,
        }

    else:
        raise ValueError("model_name must be 'resnet18' or 'resnet50'")


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    device,
    save_dir: str = "./checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_{model_name}.pth")

    cfg = get_training_config(model_name)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg["gamma"]
    )

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"\n========== Training {model_name} ==========")
    print(
        f"Config | epochs={cfg['epochs']} | lr={cfg['lr']} | "
        f"milestones={cfg['milestones']} | label_smoothing={cfg['label_smoothing']}"
    )

    for epoch in range(1, cfg["epochs"] + 1):
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{model_name} | Epoch [{epoch:02d}/{cfg['epochs']}] | "
            f"LR: {current_lr:.5f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    model.load_state_dict(best_state)
    return model, history, best_val_acc, save_path


@torch.no_grad()
def test_model(model, test_loader, device, model_name: str):
    cfg = get_training_config(model_name)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    return test_loss, test_acc


def train_and_evaluate_one_model(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    device,
    pretrained: bool = False,
):
    model = build_model(
        model_name=model_name,
        num_classes=100,
        pretrained=pretrained,
    ).to(device)

    print(f"\nModel: {model_name}, pretrained={pretrained}")

    model, history, best_val_acc, save_path = train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir="./checkpoints",
    )

    test_loss, test_acc = test_model(model, test_loader, device, model_name)

    print(f"{model_name} | Best Val Acc: {best_val_acc:.4f}")
    print(f"{model_name} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"{model_name} | Best model saved to: {save_path}")

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "save_path": save_path,
        "history": history,
    }


def main():
    # ===== shared config =====
    log_filename = "training_log.txt"
    sys.stdout = Tee(log_filename)
    seed = 42
    data_dir = "./data"
    batch_size = 128
    num_workers = 2
    val_size = 5000
    pretrained = False

    set_seed(seed)
    torch.backends.cudnn.benchmark = True

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

    results = []

    # run original resnet18 + improved resnet50
    for model_name in ["resnet18", "resnet50"]:
        result = train_and_evaluate_one_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            pretrained=pretrained,
        )
        results.append(result)

        torch.cuda.empty_cache()

    print("\n========== Final Summary ==========")
    for result in results:
        print(
            f"{result['model_name']} | "
            f"Best Val Acc: {result['best_val_acc']:.4f} | "
            f"Test Acc: {result['test_acc']:.4f} | "
            f"Saved: {result['save_path']}"
        )


if __name__ == "__main__":
    main()