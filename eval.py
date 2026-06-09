import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import get_model, set_seed

def main():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor MRI Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to base dataset directory")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the trained model checkpoint")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50_scratch", "resnet50_pretrained", "vit_small_scratch", "vit_small_pretrained"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_base = os.path.join(args.data_dir, "Brain_Tumor_MRI_Dataset")
    test_dir = os.path.join(dataset_base, "test")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory not found at {test_dir}")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    model = get_model(args.model, num_classes)
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.model}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"[{args.model}] Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save metrics to text file
    metrics_path = os.path.join(args.results_dir, f"{args.model}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {args.model}')
    cm_path = os.path.join(args.results_dir, f"{args.model}_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics and confusion matrix to {args.results_dir}")

if __name__ == "__main__":
    main()
