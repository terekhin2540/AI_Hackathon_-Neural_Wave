import math

from torchvision import models
import torch.nn as nn
import torch


class EfficientNetV2SBinary(nn.Module):
    def __init__(self):
        super().__init__()
        # Load base model

        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.dropout_classifier = nn.Dropout(p=0.3)

        # self.mlp = nn.Sequential(nn.Linear(1280, 2500), nn.SiLU(), nn.Linear(2500, 1280), nn.SiLU())
        self.mlp = nn.Identity()

        # Modify classifier for binary output
        self.model.classifier = nn.Linear(1280, 1)  # Binary output

        # Initialize new layer
        init_range = 1.0 / math.sqrt(self.model.classifier.out_features)
        nn.init.uniform_(self.model.classifier.weight, -init_range, init_range)
        nn.init.zeros_(self.model.classifier.bias)

        # Freeze feature extraction layers
        # self._freeze_layers()

    def _freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model.features(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.model.classifier(self.dropout_classifier(x))

        return x
