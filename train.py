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
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor MRI Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to base dataset directory (should contain train/ and test/ directories inside Brain_Tumor_MRI_Dataset)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory to save the trained model checkpoint")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50_scratch", "resnet50_pretrained", "vit_small_scratch", "vit_small_pretrained"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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

    # Model
    model = get_model(args.model, num_classes)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_acc = 0.0
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.model}_best.pt")

    # --- ADDED IF CONDITION BEFORE TRAINING ---
    if os.path.exists(ckpt_path):
        print(f"\n[INFO] Checkpoint found at '{ckpt_path}'. Skipping training.")
    else:
        print(f"\n[INFO] No existing checkpoint found. Starting training for {args.epochs} epochs...")
        # Training Loop
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

            print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save Best Model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                print(f"  --> Saved new best checkpoint to {ckpt_path}")

        print(f"Training Complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
