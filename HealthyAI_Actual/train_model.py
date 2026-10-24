from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import (
    GroupShuffleSplit,
)

from sklearn.utils.class_weight import (
    compute_class_weight,
)

from torch import nn

from torch.utils.data import (
    DataLoader,
)

from torchvision.models import (
    MobileNet_V2_Weights,
    mobilenet_v2,
)

from ml.dataset import (
    CLASSES,
    HAM10000Dataset,
    build_image_index,
    eval_transform,
    load_metadata,
    train_transform,
)


BASE_DIR = Path(
    __file__
).resolve().parent

DATASET_DIR = (
    BASE_DIR / "dataset"
)

MODEL_DIR = (
    BASE_DIR / "model"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA SPLIT
# ============================================================

def split_dataframe(
    dataframe,
    seed,
):

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_indices, temp_indices = next(
        first_split.split(
            dataframe,
            groups=dataframe[
                "lesion_id"
            ],
        )
    )

    train_dataframe = (
        dataframe
        .iloc[train_indices]
        .copy()
    )

    temporary_dataframe = (
        dataframe
        .iloc[temp_indices]
        .copy()
    )

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed,
    )

    validation_indices, test_indices = next(
        second_split.split(
            temporary_dataframe,
            groups=temporary_dataframe[
                "lesion_id"
            ],
        )
    )

    validation_dataframe = (
        temporary_dataframe
        .iloc[validation_indices]
        .copy()
    )

    test_dataframe = (
        temporary_dataframe
        .iloc[test_indices]
        .copy()
    )

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(CLASSES),
    )

    return model


# ============================================================
# TRAIN/EVALUATE ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training=True,
):

    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            if training:

                loss.backward()

                optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    average_loss = (
        total_loss
        / max(total, 1)
    )

    accuracy = (
        correct
        / max(total, 1)
    )

    return (
        average_loss,
        accuracy,
    )


# ============================================================
# TEST PREDICTIONS
# ============================================================

def predict(
    model,
    loader,
    device,
):

    model.eval()

    true_values = []
    predicted_values = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            logits = model(
                images
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            predicted_values.extend(
                predictions
            )

            true_values.extend(
                labels.numpy()
            )

    return (
        np.asarray(true_values),
        np.asarray(predicted_values),
    )


# ============================================================
# ONNX EXPORT
# ============================================================

def export_onnx(
    model,
    output_path,
):

    model = model.cpu()
    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=[
            "input"
        ],
        output_names=[
            "logits"
        ],
        opset_version=18,
        dynamo=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs-head",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--epochs-fine",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Optional number of images "
            "for a quick pipeline test."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Healthy AI training"
    )
    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    dataframe = load_metadata(
        DATASET_DIR
    )

    image_index = build_image_index(
        DATASET_DIR
    )

    dataframe = dataframe[
        dataframe.image_id.isin(
            image_index
        )
    ].copy()

    if args.limit:

        limit = min(
            args.limit,
            len(dataframe),
        )

        dataframe = (
            dataframe
            .sample(
                limit,
                random_state=args.seed,
            )
            .reset_index(drop=True)
        )

    missing_classes = sorted(
        set(CLASSES)
        - set(
            dataframe.dx.unique()
        )
    )

    if missing_classes:

        raise RuntimeError(
            "Dataset is missing classes: "
            f"{missing_classes}"
        )

    print(
        "Images:",
        len(dataframe),
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = split_dataframe(
        dataframe,
        args.seed,
    )

    print(
        "Train:",
        len(train_dataframe),
    )

    print(
        "Validation:",
        len(validation_dataframe),
    )

    print(
        "Test:",
        len(test_dataframe),
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = HAM10000Dataset(
        train_dataframe,
        image_index,
        train_transform(),
    )

    validation_dataset = HAM10000Dataset(
        validation_dataframe,
        image_index,
        eval_transform(),
    )

    test_dataset = HAM10000Dataset(
        test_dataframe,
        image_index,
        eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_mapping = {
        class_name: index
        for index, class_name
        in enumerate(CLASSES)
    }

    training_labels = (
        train_dataframe.dx
        .map(class_mapping)
        .to_numpy()
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(
            len(CLASSES)
        ),
        y=training_labels,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model()
    model = model.to(device)

    # Freeze MobileNet backbone
    for parameter in (
        model.features.parameters()
    ):

        parameter.requires_grad = False

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_validation_accuracy = 0.0

    history = []

    # --------------------------------------------------------
    # HEAD TRAINING
    # --------------------------------------------------------

    print()
    print(
        "Training classifier head..."
    )

    for epoch in range(
        args.epochs_head
    ):

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )

        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer,
            device,
            training=False,
        )

        epoch_number = epoch + 1

        history.append(
            [
                epoch_number,
                train_loss,
                train_accuracy,
                validation_loss,
                validation_accuracy,
            ]
        )

        print(
            f"Epoch {epoch_number}: "
            f"train={train_accuracy:.4f} "
            f"validation={validation_accuracy:.4f}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                model.state_dict(),
                MODEL_DIR
                / "healthy_ai_best.pth",
            )

    # --------------------------------------------------------
    # FINE TUNING
    # --------------------------------------------------------

    print()
    print(
        "Fine-tuning MobileNetV2..."
    )

    for parameter in (
        model.features.parameters()
    ):

        parameter.requires_grad = True

    for parameter in (
        model.features[:-30].parameters()
    ):

        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters(),
        ),
        lr=1e-5,
        weight_decay=1e-4,
    )

    for fine_epoch in range(
        args.epochs_fine
    ):

        epoch_number = (
            args.epochs_head
            + fine_epoch
            + 1
        )

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )

        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer,
            device,
            training=False,
        )

        history.append(
            [
                epoch_number,
                train_loss,
                train_accuracy,
                validation_loss,
                validation_accuracy,
            ]
        )

        print(
            f"Epoch {epoch_number}: "
            f"train={train_accuracy:.4f} "
            f"validation={validation_accuracy:.4f}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                model.state_dict(),
                MODEL_DIR
                / "healthy_ai_best.pth",
            )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    best_model_path = (
        MODEL_DIR
        / "healthy_ai_best.pth"
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location="cpu",
        )
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print()
    print(
        "Evaluating test set..."
    )

    true_values, predicted_values = predict(
        model,
        test_loader,
        device,
    )

    report = classification_report(
        true_values,
        predicted_values,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    (
        MODEL_DIR
        / "metrics.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        MODEL_DIR
        / "class_names.json"
    ).write_text(
        json.dumps(
            CLASSES,
            indent=2,
        ),
        encoding="utf-8",
    )

    split_information = {

        "total":
            len(dataframe),

        "train":
            len(train_dataframe),

        "validation":
            len(validation_dataframe),

        "test":
            len(test_dataframe),

        "grouped_by":
            "lesion_id",

        "seed":
            args.seed,
    }

    (
        MODEL_DIR
        / "dataset_split.json"
    ).write_text(
        json.dumps(
            split_information,
            indent=2,
        ),
        encoding="utf-8",
    )

    history_dataframe = pd.DataFrame(
        history,
        columns=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ],
    )

    history_dataframe.to_csv(
        MODEL_DIR
        / "training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(
        true_values,
        predicted_values,
        labels=np.arange(
            len(CLASSES)
        ),
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        matrix
    )

    plt.title(
        "Healthy AI Confusion Matrix"
    )

    plt.xticks(
        range(len(CLASSES)),
        CLASSES,
        rotation=45,
    )

    plt.yticks(
        range(len(CLASSES)),
        CLASSES,
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        MODEL_DIR
        / "confusion_matrix.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # EXPORT ONNX
    # --------------------------------------------------------

    float_model_path = (
        MODEL_DIR
        / "healthy_ai_float32.onnx"
    )

    print()
    print(
        "Exporting ONNX model..."
    )

    export_onnx(
        model,
        float_model_path,
    )

    print()
    print(
        "=" * 60
    )

    print(
        "Training completed."
    )

    print(
        "Test accuracy:",
        round(
            report["accuracy"],
            4,
        ),
    )

    print(
        "ONNX:",
        float_model_path,
    )

    print(
        "Next:"
    )

    print(
        "python quantize_model.py"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()