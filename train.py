from HybridCNNViT import HybridCNNViT
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
import random
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(model_name, num_classes):
    if model_name == "resnet50_scratch":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50_pretrained":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Freeze early layers if you want, but standard fine-tuning updates all or last few.
        # We will fine-tune the whole model for simplicity and better performance.
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit_small_scratch":
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)
    elif model_name == "vit_small_pretrained":
        model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes)
    elif model_name == "hybrid_cnn_vit":
        model = HybridCNNViT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def save_loss_curve(history, save_path, model_name):
    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training/Validation Loss - {model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def save_history_csv(history, save_path):
    with open(save_path, "w") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
        for epoch, train_loss, val_loss, train_acc, val_acc in zip(
            history["epoch"],
            history["train_loss"],
            history["val_loss"],
            history["train_acc"],
            history["val_acc"],
        ):
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.6f},{val_acc:.6f}\n")

def measure_sample_inference_time(model, data_loader, device, warmup_batches=2):
    model.eval()
    total_samples = 0
    total_time = 0.0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)

            if batch_idx < warmup_batches:
                _ = model(images)
                continue

            sync_device(device)
            start_time = time.perf_counter()
            _ = model(images)
            sync_device(device)
            elapsed = time.perf_counter() - start_time

            total_time += elapsed
            total_samples += images.size(0)

    if total_samples == 0:
        return None

    return (total_time / total_samples) * 1000.0

def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor MRI Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to base dataset directory (should contain train/ and test/ directories inside Brain_Tumor_MRI_Dataset)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory to save the trained model checkpoint")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50_scratch", "resnet50_pretrained", "vit_small_scratch", "vit_small_pretrained", "hybrid_cnn_vit"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    # Assuming dataset was extracted to args.data_dir/Brain_Tumor_MRI_Dataset
    dataset_base = os.path.join(args.data_dir, "Epic and CSCR hospital Dataset")
    train_dir = os.path.join(dataset_base, "Train")
    test_dir = os.path.join(dataset_base, "Test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory not found at {test_dir}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}")

    model = get_model(args.model, num_classes).to(device)

    if args.model == 'hybrid_cnn_vit':
        args.epochs = 50

    # --- Optimizer: differential LR for hybrid, flat LR for everything else ---
    if args.model == "hybrid_cnn_vit":
        optimizer = optim.AdamW([
            {"params": model.layer3.parameters(),          "lr": 1e-5},
            {"params": model.layer4.parameters(),          "lr": 1e-5},
            {"params": model.proj.parameters(),            "lr": 3e-4},
            {"params": model.proj_norm.parameters(),       "lr": 3e-4},
            {"params": model.cnn_proj.parameters(),        "lr": 3e-4},
            {"params": model.transformer.parameters(),     "lr": 3e-4},
            {"params": model.head.parameters(),            "lr": 3e-4},
            {"params": [model.cls_token, model.pos_embed], "lr": 3e-4},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # --- Label smoothing loss (0.1 for hybrid, standard CE for others) ---
    if args.model == "hybrid_cnn_vit":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.model == "hybrid_cnn_vit":
        def warmup_cosine_lambda(current_epoch):
            if current_epoch < args.warmup_epochs:
                return float(current_epoch + 1) / float(args.warmup_epochs)
            progress = (current_epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
    else:
        scheduler = None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_acc = -1.0
    best_epoch = 0
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.model}_best.pt")
    loss_curve_path = os.path.join(args.ckpt_dir, f"{args.model}_loss_curve.png")
    history_csv_path = os.path.join(args.ckpt_dir, f"{args.model}_history.csv")
    summary_path = os.path.join(args.ckpt_dir, f"{args.model}_training_summary.txt")
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    print(f"\n[INFO] Starting training for {args.epochs} epochs..."
            f"(warmup: {args.warmup_epochs} epochs)...")

    sync_device(device)
    training_start_time = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step()
            # Current LR for logging (first param group is representative)
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else 1e-4
            print(f"Epoch [{epoch+1:>3}/{args.epochs}] "
                f"lr: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path}")

    sync_device(device)
    training_time_sec = time.perf_counter() - training_start_time

    save_loss_curve(history, loss_curve_path, args.model)
    save_history_csv(history, history_csv_path)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    sample_time_ms = measure_sample_inference_time(model, test_loader, device)

    with open(summary_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
        f.write(f"Training Time Seconds: {training_time_sec:.2f}\n")
        f.write(f"Training Time Minutes: {training_time_sec / 60.0:.2f}\n")
        if sample_time_ms is not None:
            f.write(f"Sample Test Time Milliseconds: {sample_time_ms:.4f}\n")
        else:
            f.write("Sample Test Time Milliseconds: unavailable\n")
        f.write(f"Loss Curve Path: {loss_curve_path}\n")
        f.write(f"History CSV Path: {history_csv_path}\n")
        f.write(f"Checkpoint Path: {ckpt_path}\n")

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.4f}")
    print(f"Training time: {training_time_sec:.2f} sec ({training_time_sec / 60.0:.2f} min)")
    if sample_time_ms is not None:
        print(f"Sample test inference time: {sample_time_ms:.4f} ms/image")
    else:
        print("Sample test inference time: unavailable (not enough batches after warmup)")
    print(f"Saved loss curve to: {loss_curve_path}")
    print(f"Saved training history to: {history_csv_path}")
    print(f"Saved training summary to: {summary_path}")

if __name__ == "__main__":
    main()
