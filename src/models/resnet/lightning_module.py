import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

from models.resnet.modules import PretrainedResNet50
from utils.metrics import accuracy


class PretrainedResNetLightning(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        unfreeze_epoch=5,
        unfreeze_layers=0,
        weight_decay=1e-4
    ):
        super().__init__()
        self.model = PretrainedResNet50(freeze_backbone=True, dropout_rate=0.5)
        self.learning_rate = learning_rate
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_layers = unfreeze_layers
        self.weight_decay = weight_decay
        
        # Loss and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float().view(-1, 1))
        
        # Calculate accuracy
        preds = torch.sigmoid(logits).round()
        acc = accuracy(preds, y)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float().view(-1, 1))
        
        # Calculate accuracy
        preds = torch.sigmoid(logits).round()
        acc = accuracy(preds, y)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        # Use AdamW with weight decay for better regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
    
    def on_epoch_start(self):
        # Unfreeze layers after specified epoch
        if self.current_epoch == self.unfreeze_epoch:
            self.model.unfreeze(self.unfreeze_layers)
            self.log('unfroze_layers', self.unfreeze_layers)