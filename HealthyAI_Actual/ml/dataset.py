from __future__ import annotations

from pathlib import Path

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms


CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


IMAGE_SIZE = 224


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]


IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def build_image_index(
    dataset_dir: Path,
) -> dict[str, Path]:

    index = {}

    folders = [
        dataset_dir
        / "HAM10000_images_part_1",

        dataset_dir
        / "HAM10000_images_part_2",
    ]

    for folder in folders:

        if not folder.exists():
            continue

        for image_path in folder.glob(
            "*.jpg"
        ):

            index[
                image_path.stem
            ] = image_path

    return index


def load_metadata(
    dataset_dir: Path,
) -> pd.DataFrame:

    metadata_path = (
        dataset_dir
        / "HAM10000_metadata.csv"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Missing {metadata_path}. "
            "Run download_dataset.py first."
        )

    dataframe = pd.read_csv(
        metadata_path
    )

    required_columns = {
        "lesion_id",
        "image_id",
        "dx",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:

        raise ValueError(
            "HAM10000 metadata is missing "
            f"columns: {sorted(missing)}"
        )

    return dataframe


class HAM10000Dataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_index,
        transform=None,
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

        self.image_index = image_index

        self.transform = transform

        self.class_to_index = {
            class_name: index
            for index, class_name
            in enumerate(CLASSES)
        }

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(self, index):

        row = self.dataframe.iloc[
            index
        ]

        image_path = self.image_index.get(
            row.image_id
        )

        if image_path is None:

            raise FileNotFoundError(
                "Image not found for "
                f"image_id={row.image_id}"
            )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

        if self.transform:

            image = self.transform(
                image
            )

        label = self.class_to_index[
            row.dx
        ]

        return image, label


def train_transform():

    return transforms.Compose(
        [

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.15
            ),

            transforms.RandomRotation(
                degrees=18
            ),

            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(
                    0.82,
                    1.0,
                ),
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


def eval_transform():

    return transforms.Compose(
        [

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )