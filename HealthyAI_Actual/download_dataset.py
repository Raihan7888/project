from __future__ import annotations

from pathlib import Path
import sys

from kaggle.api.kaggle_api_extended import (
    KaggleApi,
)


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

KAGGLE_DATASET = (
    "kmader/skin-cancer-mnist-ham10000"
)


def main():

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = (
        DATASET_DIR
        / "HAM10000_metadata.csv"
    )

    part1 = (
        DATASET_DIR
        / "HAM10000_images_part_1"
    )

    part2 = (
        DATASET_DIR
        / "HAM10000_images_part_2"
    )

    if (
        metadata.exists()
        and part1.exists()
        and part2.exists()
    ):

        image_count = (
            len(list(part1.glob("*.jpg")))
            +
            len(list(part2.glob("*.jpg")))
        )

        print()
        print(
            f"HAM10000 already exists."
        )
        print(
            f"Images found: {image_count}"
        )
        return

    print()
    print("=" * 60)
    print("Healthy AI — HAM10000 Downloader")
    print("=" * 60)
    print()
    print(
        "Dataset:"
    )
    print(
        KAGGLE_DATASET
    )
    print()

    try:

        api = KaggleApi()
        api.authenticate()

    except Exception as exc:

        print()
        print(
            "Kaggle authentication failed."
        )
        print()
        print(
            "Place your Kaggle API file here:"
        )
        print(
            r"%USERPROFILE%\.kaggle\kaggle.json"
        )
        print()
        print(
            f"Details: {exc}"
        )

        sys.exit(1)

    print(
        "Downloading HAM10000..."
    )

    try:

        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(
                DATASET_DIR
            ),
            unzip=True,
        )

    except Exception as exc:

        print()
        print(
            "Dataset download failed."
        )
        print(
            f"Details: {exc}"
        )

        sys.exit(1)

    if not metadata.exists():

        print(
            "ERROR: HAM10000_metadata.csv "
            "was not found after download."
        )

        sys.exit(1)

    image_count = 0

    if part1.exists():

        image_count += len(
            list(
                part1.glob("*.jpg")
            )
        )

    if part2.exists():

        image_count += len(
            list(
                part2.glob("*.jpg")
            )
        )

    print()
    print("=" * 60)
    print("Download complete")
    print("=" * 60)
    print(
        f"Images found: {image_count}"
    )
    print(
        f"Metadata: {metadata}"
    )
    print()
    print(
        "Next command:"
    )
    print(
        "python train_model.py"
    )


if __name__ == "__main__":
    main()