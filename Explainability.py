import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # Required for generating and overlaying heatmaps

class GradCAM:
    """
    Computes Grad-CAM for models containing convolutional feature maps
    (Works for ResNet50 models and the Hybrid CNN-ViT backbone).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # Capture gradient with respect to output feature map
            self.gradients = grad_output[0]

        # Register forward and full backward hooks
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_image, target_class=None):
        self.model.zero_grad()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients.detach()      # B x C x H x W
        activations = self.activations.detach()  # B x C x H x W

        # Global average pool the gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # B x C x 1 x 1
        
        # Weighted linear combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze(0)  # H x W
        cam = F.relu(cam) # Relu keeps only features that positively correlate with the target class
        
        # Normalize heatmap to [0, 1]
        cam -= cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam /= cam_max
            
        return cam.cpu().numpy(), target_class

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


class ViTAttentionVisualizer:
    """
    Extracts self-attention weights from the final Transformer block of a ViT
    and builds an attention map relative to the [CLS] token.
    """
    def __init__(self, model):
        self.model = model
        self.attention_weights = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        target_layer = None
        
        # Attempt to find standard timm ViT structure
        if hasattr(self.model, 'blocks'):
            target_layer = self.model.blocks[-1].attn.attn_drop
        # Alternate standard pytorch transformer stack structure
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            target_layer = self.model.transformer.layers[-1].self_attn
            
        if target_layer is not None:
            def hook(module, input, output):
                # Standard timm model passes self-attention weights directly into the dropout layer
                self.attention_weights = input[0].detach()
            self.hook_handle = target_layer.register_forward_hook(hook)
        else:
            print("[Warning] Could not automatically locate the final self-attention layer for ViT.")

    def generate_attention_map(self, input_image):
        self.model.zero_grad()
        _ = self.model(input_image)
        if self.attention_weights is None:
            return None
        
        # Attention shape: [batch, num_heads, sequence_len, sequence_len]
        # In ViT, sequence_len = 1 (CLS token) + num_patches
        attn = self.attention_weights[0]  # Take first batch sample [num_heads, seq_len, seq_len]
        
        # Average attention across all attention heads
        attn_mean = attn.mean(dim=0)      # [seq_len, seq_len]
        
        # Extract how the [CLS] token (index 0) attends to all spatial patch tokens (indices 1 to end)
        cls_attn = attn_mean[0, 1:]       # [num_patches]
        
        # Determine the square grid dimension (e.g., 14x14 if 196 patches)
        num_patches = cls_attn.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        if grid_size * grid_size == num_patches:
            cls_attn_2d = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
            # Normalize map to [0, 1]
            cls_attn_2d -= cls_attn_2d.min()
            if cls_attn_2d.max() > 0:
                cls_attn_2d /= cls_attn_2d.max()
            return cls_attn_2d
        else:
            return None

    def remove_hooks(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()


# Helper function to superimpose heatmap on image and plot
def save_explanation_plot(image_tensor, heatmap, true_class, pred_class, save_path):
    # 1. Denormalize the image tensor to standard 0-255 RGB image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * std + mean) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    # 2. Resize heatmap to match image size (224 x 224)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # 3. Apply color mapping
    color_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    # 4. Superimpose heatmap with original image (alpha = opacity)
    overlayed = cv2.addWeighted(img_np, 0.6, color_heatmap, 0.4, 0)
    
    # 5. Build dual-pane figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original MRI\nTrue: {true_class}")
    axes[0].axis('off')
    
    axes[1].imshow(overlayed)
    axes[1].set_title(f"Model Explanation\nPred: {pred_class}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
