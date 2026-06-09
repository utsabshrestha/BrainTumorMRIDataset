1. Dataset Information
• Dataset Name: Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No
Tumor)
• Source URL: Brain Tumor MRI Dataset (Mendeley Dataset)
• Dataset Version: Version 6
• Imaging Modality: T1-weighted contrast-enhanced MRI (2D slices)
• Total Number of Images: 11,148 images
• Data Split Ratio: Fixed 80% Training / 20% Testing split
Pathology Class Training Set (80%) Testing Set (20%) Total Images
Glioma 3,018 603 3,621
Meningioma 2,183 436 2,619
Pituitary Tumor 2,145 429 2,574
No Tumor
(Healthy)
1,945 389 2,334
TOTAL 8,991 1,857 11,148
2. Project Task
• Main Task: The main task of this project is Multi-Class Image Classification
combined with Explainable AI (XAI).
• Project Input and Output:
− Input: Single channel (grayscale) 2D T1-weighted contrast-enhanced Magnetic
Resonance Imaging (MRI) brain scans, resized to a standardized spatial
resolution of 224 x 224 pixels and mapped to 3 channels (RGB) to
accommodate pre-trained network requirements.
− Output: A categorical probability distribution across four distinct diagnostic
classes: Glioma, Meningioma, Pituitary Tumor, and No Tumor (Healthy),
supplemented by visual interpretability maps (heatmaps) highlighting the
pathological regions driving the classification.
3. Deep Learning Models
We will employ a comparative approach utilizing two distinct paradigms of computer
vision models:
1. Baseline CNN (DenseNet121 or ResNet50): We will utilize a standard
Convolutional Neural Network backbone. Suitability: CNNs possess a strong
inductive bias for spatial locality and translation invariance, making them
exceptional at extracting fine-grained edge, texture, and structural details from
medical scans with low computational overhead.
2. Advanced Structure (Swin Transformer or Small Vision Transformer - ViT): We will
deploy a lightweight Vision Transformer (such as vit_small_patch16_224 or Swin-
T). Suitability: Transformers leverage multi-head self-attention mechanisms to
model global dependencies and long-range contextual relationships across the
entire image. This allows the model to capture macro-level structural anomalies
and complex tumor boundaries that traditional CNNs might struggle to integrate
seamlessly.
4. Novelty
While brain tumor classification using vanilla CNNs is heavily documented in existing
literature, this project introduces novelty through a controlled comparative
benchmarking of localized CNN feature extraction against global Vision Transformer
attention mechanisms on the same diagnostic dataset. Furthermore, the project moves
beyond a standard "black-box" numerical evaluation by layering Explainable AI
diagnostics (Grad-CAM for CNNs and Attention Rollout for ViTs) to qualitatively
evaluate whether advanced attention layers align more accurately with real clinical
radiology annotations than traditional convolutional layers.
5. Data Processing
• Preprocessing Methods & Rationale:
− Resizing to 224 x 224: Mandated by the mathematical patch-splitting
architecture of pre-trained Vision Transformers (16 x16 pixel patches).
− Channel Triplication: Converting 1-channel grayscale MRIs to 3-channel RGB
tensors so they match the input dimensional requirements of models pre-
trained on ImageNet.
− ImageNet Standard Normalization: Normalizing pixel values using ImageNet's
mean and standard deviation to ensure the pre-trained weights transfer
smoothly and converge faster.
− Mild Data Augmentation (Random Horizontal Flips/Rotations): Applied strictly
to the training split to enhance model generalization and mitigate spatial bias
without distorting the clinical features of the tumors.
• Postprocessing Methods & Rationale:
− Softmax Activation: Applied to the raw logits of the models to convert output
vectors into a highly interpretable probability distribution summing to 1.
− Argmax Mapping: Mapping the maximum probability index back to its
corresponding clinical class string for deployment formatting.
− Activation Mapping Generation: Running Grad-CAM and Attention Rollout
algorithms post-inference to map internal gradients and attention weights
back onto the original visual space of the input MRI.
6. Research Challenges
• Transfer Learning: Yes. Training advanced Vision Transformers from scratch
requires millions of images to overcome their lack of inherent inductive bias. We
address this by utilizing weights pre-trained on the massive ImageNet dataset,
freezing the foundational feature extraction layers, and fine-tuning only the deep
clinical classification layers.
• Interpretability / Explainable AI (XAI): Yes. AI tools in healthcare cannot be
deployed blindly. This project actively addresses the "black box" challenge by
integrating Grad-CAM (for the CNN) and Attention Maps (for the ViT) to visually
justify the model's classifications, allowing a clinician to verify if the network is
focusing on actual tumor mass or random image artifacts.
7. Evaluation
• Evaluation Metrics: Accuracy: For overall correctness.
− Precision: To evaluate the false positive rate.
− Recall (Sensitivity): Crucial for clinical diagnostics. Measures the model’s ability
to catch true positive tumors, minimizing dangerous False Negatives (missing
a real tumor).
− F1-Score: To measure the harmonic balance between Precision and Recall.
• Visualizations to Present:
− Training & Validation Loss/Accuracy Curves: To trace convergence patterns and
evaluate potential overfitting.
− Confusion Matrices (Side-by-Side): One for the CNN and one for the ViT to
explicitly see which classes (e.g., Meningioma vs. Glioma) are most frequently
misclassified by each architecture.
− XAI Heatmaps Overlay: A multi-panel visual comparison displaying: Original MRI
Image -> CNN Grad-CAM Overlay -> ViT Attention Map Overlay to visually audit
model logic.

📂 Folder Structure:
------------------------
/Brain_Tumor_MRI_Dataset/
├── train/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Pituitary/
│   └── No_Tumor/
├── test/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Pituitary/
│   └── No_Tumor/

📷 Image Details:
------------------------
Format: JPEG/PNG
Modality: T1-weighted contrast-enhanced MRI
Color: Grayscale or RGB (depending on scan)
Pre-processed and organized into labeled folders