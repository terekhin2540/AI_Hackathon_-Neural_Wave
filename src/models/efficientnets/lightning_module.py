import torch
from torch import nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from models.efficientnets.modules import EfficientNetV2SBinary
from utils.metrics import accuracy
from sklearn.metrics import f1_score

class EfficientNetV2SBinaryLightningModule(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = EfficientNetV2SBinary()  # Images must be 384 x 384
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = F.sigmoid(logits).round()
        loss = self.criterion(logits, y.float().view(-1, 1))

        preds = preds.view(-1)
        y = y.view(-1)

        acc = accuracy(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = F.sigmoid(logits).round()
        loss = self.criterion(logits, y.float().view(-1, 1))

        preds = preds.view(-1)
        y = y.view(-1)

        acc = accuracy(preds, y)
        f_beta = self.fbeta_score(preds, y, beta=0.5)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/pred_proportion", preds.mean(), on_epoch=True, on_step=False)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/f1", f_beta, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = F.sigmoid(logits).round()
        loss = self.criterion(logits, y.float().view(-1, 1))

        preds = preds.view(-1)
        y = y.view(-1)
        
        acc = accuracy(preds, y)
        f_beta = self.fbeta_score(preds, y, beta=0.5)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test/pred_proportion", preds.mean(), on_epoch=True, on_step=False)
        self.log("test/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/f1", f_beta, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def fbeta_score(self, preds, targets, beta=0.5):
        # Ensure predictions and targets are integers
        preds = preds.int()
        targets = targets.int()

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()

        # Calculate Precision and Recall
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        # Calculate F-beta score
        beta_squared = beta ** 2
        fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + 1e-7)

        return fbeta

    # def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # Define the scheduler - here we reduce the learning rate every X epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # Adjust step_size and gamma as needed
        
        # Return a dict so Lightning knows to use the scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Use "step" if you want to change after each batch
                "frequency": 1  # Apply scheduler every epoch
            }
        }
