import os
import argparse
from Explainability import GradCAM, ViTAttentionVisualizer, save_explanation_plot
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
    parser.add_argument("--model", type=str, required=True, choices=["resnet50_scratch", "resnet50_pretrained", "vit_small_scratch", "vit_small_pretrained", "hybrid_cnn_vit"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_base = os.path.join(args.data_dir, "Epic and CSCR hospital Dataset")
    test_dir = os.path.join(dataset_base, "Test")

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


    # =====================================================================
    # Generating Visual Model Explanations (New Section)
    # =====================================================================
    print("\nGenerating visual model explanations...")
    
    # Create subfolder inside your output path to keep them organized
    exp_dir = os.path.join(args.results_dir, f"{args.model}_explanations")
    os.makedirs(exp_dir, exist_ok=True)

    # Enable gradients so we can backpropagate for Grad-CAM logic
    torch.set_grad_enabled(True)
    model.train(False) # Ensure normalization layers are in eval mode

    # Determine visualizer type and initialize
    explainer = None
    explainer_type = None

    if args.model in ["resnet50_scratch", "resnet50_pretrained", "hybrid_cnn_vit"]:
        # Both ResNet and your hybrid CNN-ViT have model.layer4 containing target convolutional outputs.
        target_layer = model.layer4
        explainer = GradCAM(model, target_layer)
        explainer_type = "Grad-CAM"
    elif args.model in ["vit_small_scratch", "vit_small_pretrained"]:
        explainer = ViTAttentionVisualizer(model)
        explainer_type = "Attention Map"

    if explainer is not None:
        # Step 1: Collect one correct prediction per category as representative explanation samples
        explained_classes = set()
        samples_to_explain = []

        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            img_batch = img.unsqueeze(0).to(device)
            
            # Predict
            outputs = model(img_batch)
            pred = outputs.argmax(dim=1).item()
            
            if pred == label and label not in explained_classes:
                explained_classes.add(label)
                # Store (image_tensor, actual_label_idx, predicted_label_idx)
                samples_to_explain.append((img, label, pred))
                
            if len(explained_classes) == num_classes:
                break

        # Step 2: Generate visual heatmaps and save
        for img, label, pred in samples_to_explain:
            img_batch = img.unsqueeze(0).to(device)
            true_name = class_names[label]
            pred_name = class_names[pred]
            
            heatmap = None
            if explainer_type == "Grad-CAM":
                heatmap, _ = explainer.generate_cam(img_batch, target_class=label)
            elif explainer_type == "Attention Map":
                heatmap = explainer.generate_attention_map(img_batch)

            if heatmap is not None:
                # Save visual explanation plot
                safe_name = true_name.lower().replace(' ', '_')
                save_path = os.path.join(exp_dir, f"explain_{safe_name}.png")
                save_explanation_plot(img, heatmap, true_name, pred_name, save_path)
                print(f"Successfully saved {explainer_type} map for '{true_name}' class to: {save_path}")
            else:
                print(f"[Warning] Failed to generate {explainer_type} map for class '{true_name}'.")

        # Cleanup hooks
        explainer.remove_hooks()
    else:
        print("[Warning] No valid explainability tool initialized.")

    # Reset gradient behavior back to default
    torch.set_grad_enabled(False)
    print("Pipeline evaluation completed successfully.")

if __name__ == "__main__":
    main()
    