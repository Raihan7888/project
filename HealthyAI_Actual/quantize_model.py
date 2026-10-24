from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from PIL import Image

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


BASE_DIR = Path(
    __file__
).resolve().parent

MODEL_DIR = (
    BASE_DIR / "model"
)

DATASET_DIR = (
    BASE_DIR / "dataset"
)


class HAMCalibrationReader(
    CalibrationDataReader
):

    def __init__(
        self,
        image_paths,
        input_name,
    ):

        self.image_paths = image_paths
        self.input_name = input_name
        self.index = 0

    def _prepare(
        self,
        image_path,
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (
                224,
                224,
            )
        )

        array = (
            np.asarray(
                image,
                dtype=np.float32,
            )
            / 255.0
        )

        mean = np.array(
            [
                0.485,
                0.456,
                0.406,
            ],
            dtype=np.float32,
        )

        std = np.array(
            [
                0.229,
                0.224,
                0.225,
            ],
            dtype=np.float32,
        )

        array = (
            array - mean
        ) / std

        array = np.transpose(
            array,
            (
                2,
                0,
                1,
            ),
        )

        array = array[
            None,
            ...
        ]

        return {
            self.input_name:
                array
        }

    def get_next(self):

        if self.index >= len(
            self.image_paths
        ):

            return None

        image_path = (
            self.image_paths[
                self.index
            ]
        )

        self.index += 1

        return self._prepare(
            image_path
        )


def find_calibration_images():

    images = []

    for folder_name in [
        "HAM10000_images_part_1",
        "HAM10000_images_part_2",
    ]:

        folder = (
            DATASET_DIR
            / folder_name
        )

        if not folder.exists():
            continue

        images.extend(
            folder.glob(
                "*.jpg"
            )
        )

    return images[:128]


def main():

    source_model = (
        MODEL_DIR
        / "healthy_ai_float32.onnx"
    )

    output_model = (
        MODEL_DIR
        / "healthy_ai_int8.onnx"
    )

    if not source_model.exists():

        raise SystemExit(
            "Float32 ONNX model not found. "
            "Run train_model.py first."
        )

    images = (
        find_calibration_images()
    )

    if not images:

        raise SystemExit(
            "HAM10000 images were not found."
        )

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(source_model),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_name = (
        session.get_inputs()[0].name
    )

    reader = HAMCalibrationReader(
        images,
        input_name,
    )

    print()
    print(
        "Healthy AI INT8 quantization"
    )

    print(
        f"Calibration images: {len(images)}"
    )

    quantize_static(
        model_input=str(
            source_model
        ),
        model_output=str(
            output_model
        ),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    information = {

        "source":
            source_model.name,

        "output":
            output_model.name,

        "format":
            "ONNX QDQ",

        "weights":
            "INT8",

        "activations":
            "calibrated",

        "calibration_images":
            len(images),
    }

    (
        MODEL_DIR
        / "quantization.json"
    ).write_text(
        json.dumps(
            information,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "INT8 model created:"
    )

    print(
        output_model
    )


if __name__ == "__main__":
    main()