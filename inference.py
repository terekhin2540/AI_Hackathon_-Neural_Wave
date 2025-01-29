import argparse
from pathlib import Path
import timeit
import time

from tqdm import tqdm

import torch
import torch.nn.functional as F

from data_wrappers.dataset import BinaryImageDataset, transform_test, DataLoader
from models.efficientnets import EfficientNetV2SBinaryLightningModule
from models.swin import SwinBinaryLightningModule
from models.convnext import ConvnextBinaryLightningModule
from utils.metrics import accuracy

checkpoint_paths = {
    "efficientnet": '/teamspace/studios/this_studio/lightning_logs/efficientnetv2_edges_masked_dropout03_even_more_data_lr_sched/checkpoints/step=1374-val_loss=0.159-val_acc=0.963-val_f1=0.823.ckpt',
    "swin": '/teamspace/studios/this_studio/lightning_logs/swin_edges_masked_dropout03_more_data/checkpoints/step=394-val_loss=0.179-val_acc=0.963-val_f1=0.845.ckpt',
    "convnext": '/teamspace/studios/this_studio/lightning_logs/convnext_edges_masked_dropout04_more_data/checkpoints/step=1182-val_loss=0.120-val_acc=0.959-val_f1=0.834.ckpt'
}

target_sizes = {
    "efficientnet": 384,
    "swin": 224,
    "convnext": 384
}


def fbeta_score(preds, targets, beta=0.5):
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


def main(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == "efficientnet":
        model = EfficientNetV2SBinaryLightningModule()
        checkpoint = torch.load(checkpoint_paths[args.model], map_location=dev)
        model.load_state_dict(checkpoint['state_dict'])
    if args.model == "swin":
        model = SwinBinaryLightningModule()
        checkpoint = torch.load(checkpoint_paths[args.model], map_location=dev)
        model.load_state_dict(checkpoint['state_dict'])
    if args.model == "convnext":
        model = ConvnextBinaryLightningModule()
        checkpoint = torch.load(checkpoint_paths[args.model], map_location=dev)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"Using model: {args.model} on {dev}")

    base_path = Path(args.data)
    ds = BinaryImageDataset(
        positive_dir=base_path / "aligned",
        negative_dir=base_path / "not_aligned",
        target_size=target_sizes[args.model],
        preprocess="edges",
        mask_left=True,
        transform=transform_test,
    )
    dl = DataLoader(ds, batch_size=args.batchsize, num_workers=args.njobs)

    dataset_predictions = []
    dataset_targets = []
    dataset_time = []
    print("Processing data...")
    with torch.no_grad():
        for batch in tqdm(dl):
            x, y = batch
            start = time.time()
            logits = model(x)
            preds = F.sigmoid(logits).round()
            preds = preds.view(-1)
            end = time.time()
            elapsed = end - start
            dataset_time.append(elapsed)
            y = y.view(-1)
            dataset_predictions.append(preds)
            dataset_targets.append(y)

    dataset_avg_time = sum(dataset_time) / len(dataset_time)
    dataset_targets = torch.cat(dataset_targets, dim=-1)
    dataset_predictions = torch.cat(dataset_predictions, dim=-1)
    acc = accuracy(dataset_predictions, dataset_targets)
    f_beta = fbeta_score(dataset_predictions, dataset_targets, beta=0.5)
    print(f"Accuracy: {acc * 100:.1f}%")
    print(f"F-score: {f_beta * 100:.1f}%")
    print(f"Average prediction time: {round(dataset_avg_time, 5)} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--data", type=str,
                        help="Path to dataset. Must be a directory that contains two subdirectories: 'aligned' and 'not aligned' both of which have to contain JPEGs.")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "swin", "convnext"],
                        help="The model to use. Options are 'efficientnet', 'swin', 'convnext'")
    parser.add_argument("--batchsize", type=int, default=1,
                        help="How many examples to send to the model simultaneously. Defaults to 1.")
    parser.add_argument("--njobs", default=4, type=int,
                        help="How many processes to use for data uploading. Defaults to 1.")
    args = parser.parse_args()
    main(args)
