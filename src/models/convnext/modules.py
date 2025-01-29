import timm
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy

class ConvNextClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        # Load pretrained model
        self.backbone = timm.create_model(
            'convnext_tiny.fb_in22k',
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # New classification head
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# class TimmClassifier(pl.LightningModule):
#     def __init__(
#         self,
#         model_name='convnext_tiny.fb_in22k',
#         learning_rate=1e-4,
#         weight_decay=1e-4
#     ):
#         super().__init__()
#         self.model = ConvNextClassifier(model_name)
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
        
#         # Loss and metrics
#         self.criterion = nn.BCEWithLogitsLoss()
#         self.train_accuracy = Accuracy(task='binary')
#         self.val_accuracy = Accuracy(task='binary')
        
#     def forward(self, x):
#         return self.model(x)
    
#     def configure_optimizers(self):
#         # Separate parameter groups for backbone and classifier
#         backbone_params = self.model.backbone.parameters()
#         classifier_params = self.model.classifier.parameters()
        
#         param_groups = [
#             {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # Lower LR for backbone
#             {'params': classifier_params, 'lr': self.learning_rate}       # Higher LR for classifier
#         ]
        
#         optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=10,  # Number of epochs
#             eta_min=self.learning_rate * 0.01
#         )
        
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_loss"
#             }
#         }