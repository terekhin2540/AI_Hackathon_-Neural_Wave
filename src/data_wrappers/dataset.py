from pathlib import Path
from typing import List, Tuple, Callable
import random


from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule

import albumentations as A
from albumentations.core.composition import BboxParams, KeypointParams
from albumentations.pytorch import ToTensorV2
import cv2


def preprocess_image(image, target_size: int, preprocess: str = "grayscale"):
    """
    Args:
        image: PIL Image
        target_size: the number to make the image have the size target_size X target_size
    Returns:
        np.array: the image as a numpy array
    """
    # Convert the image to a square by cropping from the right
    assert preprocess in ("no", "grayscale", "edges")
    width, height = image.size
    crop = width - height
    cropped_image = image.crop((crop, 0, width, height))
    cropped_width, cropped_height = cropped_image.size
    assert cropped_width == cropped_height

    # Resize the image to the target size
    resized_image = cropped_image.resize((target_size, target_size))

    # Convert to grayscale if requested
    if preprocess == "grayscale":
        resized_image = resized_image.convert("L")

    image_array = np.array(resized_image)
    if preprocess == "edges":
        image_array = preprocess_dark(image_array)

    return image_array.astype(np.float32)


def preprocess_dark(image):
    # image: np.array
    # Convert the image from BGR to RGB
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Check if the image is dark or light
    if np.mean(gray_image) < 55: # Dark
        brightness = 155
        contrast = 1.5
    else: # Light
        brightness = 75
        contrast = 1.35

    # Adjust brightness and contrast on the grayscale image
    adjusted_image = cv2.convertScaleAbs(gray_image, alpha=contrast, beta=brightness)

    # Create a new layer (copy of the adjusted grayscale image)
    layer = adjusted_image.copy()

    # Apply Gaussian blur to the new layer
    blurred_layer = cv2.GaussianBlur(layer, (31, 31), 0)

    # Set the blend mod
    blended_image = cv2.subtract(adjusted_image, blurred_layer) # subtract
    return blended_image


def mask_from_the_left(image_tensor):
    # image: torch.tensor
    _, w, h = image_tensor.shape
    image_tensor[:, :, :w//2] = 0
    return image_tensor


class BinaryImageDataset(Dataset):
    def __init__(
        self,
        positive_dir: Path,
        negative_dir: Path,
        target_size: int,
        preprocess: str,
        mask_left: bool,
        transform: Callable,
    ):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.target_size = target_size
        self.prepr = preprocess
        self.mask_left = mask_left
        self.transform = transform
        print(self.target_size, self.prepr)
        assert self.transform is not None
        self.samples: List[Tuple[str, int]] = []

        # Load positive samples
        for img_path in self.positive_dir.rglob("*.jpg"):
            if img_path.is_file():
                self.samples.append((str(img_path), 1))

        # Load negative samples
        for img_path in self.negative_dir.rglob("*.jpg"):
            if img_path.is_file():
                self.samples.append((str(img_path), 0))

        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image_array = preprocess_image(image, self.target_size, self.prepr)
        image_array = image_array / 255
        if image_array.ndim == 2:
            image_array = image_array[:, :, np.newaxis].repeat(3, axis=2)

        image = self.transform(image=image_array)["image"]
        if self.mask_left:
            image = mask_from_the_left(image)

        return image, label


class SteelBinaryImageDataModule(LightningDataModule):

    def __init__(
        self,
        train_positive_dir: Path | None = None,
        train_negative_dir: Path | None = None,
        val_positive_dir: Path | None = None,
        val_negative_dir: Path | None = None,
        test_positive_dir: Path | None = None,
        test_negative_dir: Path | None = None,
        target_size: int = 384,
        preprocess: str = "no",
        mask_left: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_positive_dir = train_positive_dir
        self.train_negative_dir = train_negative_dir
        self.val_positive_dir = val_positive_dir
        self.val_negative_dir = val_negative_dir
        self.test_positive_dir = test_positive_dir
        self.test_negative_dir = test_negative_dir
        self.target_size = target_size
        self.prepr = preprocess
        self.mask_left = mask_left
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = BinaryImageDataset(
                positive_dir=self.train_positive_dir,
                negative_dir=self.train_negative_dir,
                target_size=self.target_size,
                preprocess=self.prepr,
                mask_left=self.mask_left,
                transform=transform_train,
            )
            self.val_dataset = BinaryImageDataset(
                positive_dir=self.val_positive_dir,
                negative_dir=self.val_negative_dir,
                target_size=self.target_size,
                preprocess=self.prepr,
                mask_left=self.mask_left,
                transform=transform_test,
            )
        if stage == "validate":
            self.val_dataset = BinaryImageDataset(
                positive_dir=self.val_positive_dir,
                negative_dir=self.val_negative_dir,
                target_size=self.target_size,
                preprocess=self.prepr,
                mask_left=self.mask_left,
                transform=transform_test,
            )
        if stage == "test":
            self.test_dataset = BinaryImageDataset(
                positive_dir=self.test_positive_dir,
                negative_dir=self.test_negative_dir,
                target_size=self.target_size,
                preprocess=self.prepr,
                mask_left=self.mask_left,
                transform=transform_test,
            )
        if stage == "predict":
            self.predict_dataset = BinaryImageDataset(
                positive_dir=self.test_negative_dir,
                negative_dir=self.test_negative_dir,
                target_size=self.target_size,
                preprocess=self.prepr,
                mask_left=self.mask_left,
                transform=transform_test,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


transform_train = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        # A.Affine(scale=(1.1, 1.2), p=0.5),  # Scales the image by 10-30%
        A.Affine(rotate=(-10, 10), scale=(1.0, 1.0), p=0.5),
        ToTensorV2(),
    ]
)


transform_test = A.Compose(
    [
        ToTensorV2()
    ]
)

if __name__ == "__main__":
    root = Path(
        "/teamspace/s3_connections/dtp-sbm-segmentation-video-tasks-bars-stopper-alignment-images-hackaton-usi"
    )
    train_path = root / "example_set"
    train_negative = train_path / "aligned"
    train_positive = train_path / "not_aligned"
    ds = BinaryImageDataset(
        positive_dir=train_positive,
        negative_dir=train_negative,
        target_size=384,
        preprocess="grayscale",
        mask_left=False,
        transform=transform_train,
    )
    # print(len(ds))
    print(ds.__getitem__(1))
    i, la = ds[1]
    print(la)
    print(i.shape)
