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

        # --- Projection: CNN feature tokens -> transformer width ---
        self.proj = nn.Linear(2048, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)

        # Keep a direct global CNN path. This preserves the strong pretrained
        # ResNet signal while the transformer learns spatial token interactions.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_proj = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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
        # Fuse CLS attention, mean spatial attention, and direct CNN features.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Dropout(0.3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # B x 2048 x 7 x 7

        cnn_global = self.global_pool(x).flatten(1)        # B x 2048
        cnn_global = self.cnn_proj(cnn_global)             # B x d_model

        # Flatten spatial dims -> token sequence
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # B x 49 x 2048
        x = self.proj(x)                                   # B x 49 x d_model
        x = self.proj_norm(x)
        x = self.proj_drop(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                    # B x 50 x d_model
        x = x + self._pos_embed(H, W)

        # Transformer
        x = self.transformer(x)                            # B x 50 x d_model

        # Classify from fused transformer and CNN features
        cls_out = x[:, 0]                                  # B x d_model
        token_mean = x[:, 1:].mean(dim=1)                  # B x d_model
        fused = torch.cat([cls_out, token_mean, cnn_global], dim=1)
        return self.head(fused)

    def _pos_embed(self, height, width):
        if height * width + 1 == self.pos_embed.shape[1]:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        base_size = int(math.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.transpose(1, 2).reshape(1, -1, base_size, base_size)
        patch_pos = nn.functional.interpolate(
            patch_pos,
            size=(height, width),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return torch.cat([cls_pos, patch_pos], dim=1)
