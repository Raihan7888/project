from __future__ import annotations

import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from PIL import Image, UnidentifiedImageError

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"

DB_PATH = DATA_DIR / "healthy_ai.db"

MODEL_PATH = MODEL_DIR / "healthy_ai_int8.onnx"
CLASSES_PATH = MODEL_DIR / "class_names.json"


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# DATABASE
# ============================================================

def get_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def init_database():
    with get_db() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                screening_band TEXT NOT NULL,
                top_predictions TEXT NOT NULL,
                image_width INTEGER,
                image_height INTEGER,
                image_format TEXT,
                model_version TEXT NOT NULL
            )
            """
        )

        connection.commit()


# ============================================================
# MODEL INFORMATION
# ============================================================

def load_classes():

    if not CLASSES_PATH.exists():
        return []

    try:
        return json.loads(
            CLASSES_PATH.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return []


def get_model_status():

    ready = (
        MODEL_PATH.exists()
        and CLASSES_PATH.exists()
    )

    return {
        "ready": ready,
        "model": MODEL_PATH.name,
        "classes": load_classes(),
        "size_mb": (
            round(
                MODEL_PATH.stat().st_size
                / (1024 * 1024),
                2,
            )
            if MODEL_PATH.exists()
            else None
        ),
        "inference": "browser-local ONNX Runtime Web",
        "input_size": "224x224 RGB",
    }


# ============================================================
# SCREENING BAND
# ============================================================

def screening_band(label: str, confidence: float) -> str:

    """
    Project-level UI category only.

    This is NOT a clinical risk score.
    """

    if confidence < 0.60:
        return "UNCERTAIN"

    if label in {
        "mel",
        "bcc",
        "akiec",
    }:
        return "REVIEW"

    return "LOWER-CONCERN"


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(file_storage):

    if (
        file_storage is None
        or not file_storage.filename
    ):
        raise ValueError(
            "Please choose an image first."
        )

    allowed_extensions = {
        "jpg",
        "jpeg",
        "png",
        "webp",
    }

    filename = file_storage.filename

    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if extension not in allowed_extensions:
        raise ValueError(
            "Supported formats are JPG, JPEG, PNG and WEBP."
        )

    raw_data = file_storage.read()

    if not raw_data:
        raise ValueError(
            "The uploaded file is empty."
        )

    try:

        image = Image.open(
            io.BytesIO(raw_data)
        )

        image.verify()

        image = Image.open(
            io.BytesIO(raw_data)
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:

        raise ValueError(
            "The selected file is not a valid image."
        ) from exc

    if image.width < 64 or image.height < 64:

        raise ValueError(
            "Please use an image at least 64×64 pixels."
        )

    return image


# ============================================================
# DATABASE INSERT
# ============================================================

def save_prediction(payload):

    prediction_id = uuid.uuid4().hex[:12]

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with get_db() as connection:

        connection.execute(
            """
            INSERT INTO predictions (
                id,
                created_at,
                filename,
                label,
                confidence,
                screening_band,
                top_predictions,
                image_width,
                image_height,
                image_format,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                timestamp,
                payload["filename"],
                payload["label"],
                payload["confidence"],
                payload["screening_band"],
                json.dumps(
                    payload["top_predictions"]
                ),
                payload.get("image_width"),
                payload.get("image_height"),
                payload.get("image_format"),
                "HealthyAI-MobileNetV2-ONNX-INT8-v1",
            ),
        )

        connection.commit()

    return prediction_id


# ============================================================
# MAIN PAGE
# ============================================================

@app.get("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# MODEL FILE
# ============================================================

@app.get("/model/<path:filename>")
def model_file(filename):

    return send_from_directory(
        MODEL_DIR,
        filename,
        as_attachment=False,
    )


# ============================================================
# HEALTH API
# ============================================================

@app.get("/api/health")
def health():

    return jsonify(
        {
            "ok": True,
            "model": get_model_status(),
        }
    )


# ============================================================
# SAVE HISTORY
# ============================================================

@app.post("/api/history")
def create_history():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    required_fields = [
        "filename",
        "label",
        "confidence",
        "top_predictions",
    ]

    missing = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing:

        return (
            jsonify(
                {
                    "error": (
                        "Missing fields: "
                        + ", ".join(missing)
                    )
                }
            ),
            400,
        )

    try:

        confidence = float(
            data["confidence"]
        )

        if not 0 <= confidence <= 1:
            raise ValueError

        label = str(
            data["label"]
        )

        classes = load_classes()

        if classes and label not in classes:

            return (
                jsonify(
                    {
                        "error":
                        "Unknown model class."
                    }
                ),
                400,
            )

        top_predictions = (
            data["top_predictions"]
        )

        if not isinstance(
            top_predictions,
            list,
        ):
            raise ValueError

        payload = {

            "filename":
                str(
                    data["filename"]
                )[:255],

            "label":
                label,

            "confidence":
                confidence,

            "screening_band":
                screening_band(
                    label,
                    confidence,
                ),

            "top_predictions":
                top_predictions[:3],

            "image_width":
                int(
                    data.get(
                        "image_width",
                        0,
                    )
                    or 0
                ),

            "image_height":
                int(
                    data.get(
                        "image_height",
                        0,
                    )
                    or 0
                ),

            "image_format":
                str(
                    data.get(
                        "image_format",
                        "",
                    )
                )[:20],
        }

        prediction_id = save_prediction(
            payload
        )

        return jsonify(
            {
                "ok": True,
                "id": prediction_id,
                "screening_band":
                    payload[
                        "screening_band"
                    ],
            }
        )

    except (
        TypeError,
        ValueError,
        IndexError,
    ):

        return (
            jsonify(
                {
                    "error":
                    "Invalid prediction data."
                }
            ),
            400,
        )


# ============================================================
# HISTORY
# ============================================================

@app.get("/api/history")
def history_list():

    try:

        limit = int(
            request.args.get(
                "limit",
                50,
            )
        )

    except ValueError:

        limit = 50

    limit = min(
        max(limit, 1),
        200,
    )

    with get_db() as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results = []

    for row in rows:

        item = dict(row)

        item[
            "confidence_percent"
        ] = round(
            item["confidence"] * 100,
            1,
        )

        try:

            item[
                "top_predictions"
            ] = json.loads(
                item["top_predictions"]
            )

        except Exception:

            item[
                "top_predictions"
            ] = []

        results.append(item)

    return jsonify(results)


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.delete("/api/history")
def clear_history():

    with get_db() as connection:

        connection.execute(
            "DELETE FROM predictions"
        )

        connection.commit()

    return jsonify(
        {
            "ok": True
        }
    )


# ============================================================
# STATISTICS
# ============================================================

@app.get("/api/stats")
def statistics():

    with get_db() as connection:

        total = connection.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            """
        ).fetchone()[0]

        average = connection.execute(
            """
            SELECT AVG(confidence)
            FROM predictions
            """
        ).fetchone()[0]

        bands = connection.execute(
            """
            SELECT
                screening_band,
                COUNT(*) AS count
            FROM predictions
            GROUP BY screening_band
            """
        ).fetchall()

    return jsonify(
        {
            "total": total,
            "average_confidence":
                round(
                    (average or 0) * 100,
                    1,
                ),
            "bands": {
                row["screening_band"]:
                    row["count"]
                for row in bands
            },
        }
    )


# ============================================================
# PDF REPORT
# ============================================================

@app.post("/api/report")
def create_report():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    try:

        label = str(
            data["label"]
        )

        confidence = float(
            data["confidence"]
        )

        top_predictions = (
            data.get(
                "top_predictions",
                [],
            )[:3]
        )

        filename = str(
            data.get(
                "filename",
                "uploaded-image",
            )
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        return (
            jsonify(
                {
                    "error":
                    "Invalid report data."
                }
            ),
            400,
        )

    generated = datetime.now().strftime(
        "%d %b %Y, %I:%M %p"
    )

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()

    story = [

        Paragraph(
            "Healthy AI — Screening Report",
            styles["Title"],
        ),

        Spacer(1, 12),

        Paragraph(
            "Educational skin-lesion "
            "image classification report",
            styles["Normal"],
        ),

        Spacer(1, 18),
    ]

    rows = [
        ["Field", "Value"],
        ["Generated", generated],
        ["Input image", filename],
        ["Predicted class", label],
        [
            "Model confidence",
            f"{confidence * 100:.1f}%",
        ],
        [
            "Screening band",
            screening_band(
                label,
                confidence,
            ),
        ],
        [
            "Model",
            "MobileNetV2 → ONNX INT8",
        ],
    ]

    table = Table(
        rows,
        colWidths=[
            150,
            330,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#123b35"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(1, 18)
    )

    story.append(
        Paragraph(
            "Top model outputs",
            styles["Heading2"],
        )
    )

    top_rows = [
        [
            "Class",
            "Probability",
        ]
    ]

    for prediction in top_predictions:

        try:

            probability = float(
                prediction.get(
                    "probability",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            probability = 0

        top_rows.append(
            [
                str(
                    prediction.get(
                        "label",
                        "",
                    )
                ),
                f"{probability * 100:.1f}%",
            ]
        )

    top_table = Table(
        top_rows,
        colWidths=[
            150,
            150,
        ],
    )

    top_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#edf5f2"
                    ),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(top_table)

    story.append(
        Spacer(1, 18)
    )

    story.append(
        Paragraph(
            "Important: Healthy AI is an "
            "educational research prototype. "
            "It is not a medical diagnosis and "
            "does not replace evaluation by a "
            "qualified dermatologist. Model "
            "probabilities are not clinical "
            "certainty.",
            styles["BodyText"],
        )
    )

    document.build(story)

    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="healthy_ai_report.pdf",
    )


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def request_too_large(_):

    return (
        jsonify(
            {
                "error":
                "Image is too large. "
                "Maximum upload size is 10 MB."
            }
        ),
        413,
    )


# ============================================================
# STARTUP
# ============================================================

init_database()


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("Healthy AI")
    print("=" * 60)
    print(
        "Open: http://127.0.0.1:5000"
    )
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )