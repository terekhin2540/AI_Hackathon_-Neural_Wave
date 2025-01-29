import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics import Accuracy
from torchvision.models import resnet50, ResNet50_Weights

class PretrainedResNet50(nn.Module):
    def __init__(self, freeze_backbone=True, dropout_rate=0.5):
        super().__init__()
        
        # Load pre-trained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        if freeze_backbone:
            # Freeze all layers except the final ones
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1)  # Single output for binary classification
        )
        
    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze(self, num_layers=0):
        """
        Unfreeze the last n layers of the backbone
        If num_layers = 0, unfreeze all layers
        """
        if num_layers == 0:
            # Unfreeze all layers
            for param in self.resnet.parameters():
                param.requires_grad = True
        else:
            # Get list of all layers
            layers = list(self.resnet.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True