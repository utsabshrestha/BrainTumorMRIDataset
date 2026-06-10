import torch
import torch.nn as nn
import torchvision.models as models
import math

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4, d_model=384, nhead=6, num_transformer_layers=4, dropout=0.1):
        super().__init__()

        # --- CNN Backbone (ResNet50 pretrained) ---
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Extract layer groups; freeze early layers
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # will be frozen
        self.layer2 = backbone.layer2  # will be frozen
        self.layer3 = backbone.layer3  # fine-tuned
        self.layer4 = backbone.layer4  # fine-tuned; output: B x 2048 x 7 x 7

        # Freeze stem + layer1 + layer2
        for param in list(self.stem.parameters()) + \
                     list(self.layer1.parameters()) + \
                     list(self.layer2.parameters()):
            param.requires_grad = False

        # --- Projection: 2048 -> d_model ---
        self.proj = nn.Linear(2048, d_model)

        # --- CLS token + positional embedding ---
        # 7x7 = 49 spatial tokens + 1 CLS = 50
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, d_model))  # 49 + 1
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- Classifier head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # B x 2048 x 7 x 7

        # Flatten spatial dims -> token sequence
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # B x 49 x 2048
        x = self.proj(x)                                   # B x 49 x d_model

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                    # B x 50 x d_model
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)                            # B x 50 x d_model

        # Classify from CLS token
        cls_out = x[:, 0]                                  # B x d_model
        return self.head(cls_out)