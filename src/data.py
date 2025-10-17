import os
import torch
import albumentations as A
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_strong_transforms(img_size: int = 128):
    """
    Applies a series of strong augmentations for data generation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.6
        ),
        
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.6),

        A.OneOf([
            A.CoarseDropout(
                num_holes_range=(3,9),
                hole_height_range=(0.0, 0.15),
                hole_width_range=(0.0, 0.15),
                fill=0,
                p=0.5
            ),
            A.Superpixels(
                p_replace=0.1, n_segments=150, p=0.8
            )
        ], p=0.5),

      A.OneOf([
            A.ElasticTransform(alpha=24, sigma=8, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.35, p=1.0),
        ], p=0.7),

        A.RandomSizedCrop(
            min_max_height=(int(img_size * 0.6), img_size),
            size=(img_size, img_size),
            p=0.4
        ),
    ])


def get_datasets(root="augmented_images", img_size=128):
    """
    Creates and returns the train, validation, and test datasets.
    This function is separate from DataLoader creation to allow for
    flexible batch sizes during hyperparameter tuning.
    """
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=tfm)
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=tfm)
    test_ds  = datasets.ImageFolder(os.path.join(root, "test"),  transform=tfm)

    return {"train": train_ds, "val": val_ds, "test": test_ds}
