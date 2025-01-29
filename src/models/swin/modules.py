import torch.nn as nn
import timm

class SwinClassifier(nn.Module):
    def __init__(
        self,
        dropout_rate=0.3
    ):
        super().__init__()
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in22k',
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        num_features = self.backbone.num_features
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1)  # Binary classification
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)