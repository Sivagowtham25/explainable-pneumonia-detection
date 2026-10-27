import json
import logging
import os
import tempfile
from io import BytesIO

import azure.functions as func
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient


# ==========================================================
# Azure Functions application
# ==========================================================

app = func.FunctionApp(
    http_auth_level=func.AuthLevel.ANONYMOUS
)


# ==========================================================
# Configuration
# ==========================================================

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = (224, 224)
PREDICTION_THRESHOLD = 0.5

MODEL_BLOBS = {
    "custom_cnn": "CUSTOM_CNN_BLOB",
    "mobilenetv2": "MOBILENET_BLOB",
}

# Cache models and Blob Storage client between invocations.
_loaded_models = {}
_blob_service = None


def get_required_setting(setting_name: str) -> str:
    """Read a required environment variable safely."""

    value = os.environ.get(setting_name, "").strip()

    if not value:
        raise RuntimeError(
            f"Required application setting '{setting_name}' is missing."
        )

    return value


def get_model_blob_name(model_key: str) -> str:
    """Return the configured Blob Storage filename for a model."""

    if model_key not in MODEL_BLOBS:
        raise ValueError("Unsupported model selection.")

    setting_name = MODEL_BLOBS[model_key]

    return get_required_setting(setting_name)


# ==========================================================
# Azure Blob Storage
# ==========================================================

def get_blob_service():
    """Create a Blob Storage client using managed identity."""

    global _blob_service

    if _blob_service is None:
        storage_account = get_required_setting(
            "AZURE_STORAGE_ACCOUNT"
        )

        # Accept either an account name or a complete account URL.
        if storage_account.startswith("https://"):
            account_url = storage_account.rstrip("/")
        else:
            account_url = (
                f"https://{storage_account}.blob.core.windows.net"
            )

        # Uses the Function App's system-assigned managed identity.
        credential = ManagedIdentityCredential()

        _blob_service = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )

    return _blob_service


# ==========================================================
# Model loading
# ==========================================================

def load_model(model_key: str):
    """Download a Keras model from Blob Storage and cache it."""

    if model_key not in MODEL_BLOBS:
        raise ValueError("Unsupported model selection.")

    if model_key not in _loaded_models:
        model_container = get_required_setting("MODEL_CONTAINER")
        blob_name = get_model_blob_name(model_key)

        logging.info(
            "Loading model '%s' from Blob Storage.",
            model_key,
        )

        blob_client = get_blob_service().get_blob_client(
            container=model_container,
            blob=blob_name,
        )

        model_bytes = blob_client.download_blob().readall()

        # Keras 3 expects a filesystem path for .keras archives; it no
        # longer accepts a BytesIO object in load_model().
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(
                temp_dir,
                os.path.basename(blob_name),
            )

            with open(model_path, "wb") as model_file:
                model_file.write(model_bytes)

            model = tf.keras.models.load_model(
                model_path,
                compile=False,
            )

        _loaded_models[model_key] = model

        logging.info(
            "Model '%s' loaded successfully.",
            model_key,
        )

    return _loaded_models[model_key]


# ==========================================================
# JSON response helper
# ==========================================================

def json_response(data: dict, status_code: int = 200):
    """Return a JSON-formatted HTTP response."""

    return func.HttpResponse(
        body=json.dumps(data),
        status_code=status_code,
        mimetype="application/json",
    )


# ==========================================================
# Health endpoint
# GET /api/health
# ==========================================================

@app.route(
    route="health",
    methods=["GET"],
)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Check whether the API is running."""

    return json_response({
        "status": "success",
        "application": "PneumoScan AI",
        "message": "Azure Functions API is running.",
        "available_models": list(MODEL_BLOBS.keys()),
    })


# ==========================================================
# Prediction endpoint
# POST /api/predict?model=custom_cnn
# ==========================================================

@app.route(
    route="predict",
    methods=["POST"],
)
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """
    Accept an image through the HTTP request body.

    Example:
        POST /api/predict?model=custom_cnn

    Request body:
        Raw JPEG or PNG image bytes.

    Header:
        Content-Type: image/jpeg
        or
        Content-Type: image/png
    """

    try:
        # --------------------------------------------------
        # 1. Validate the selected model
        # --------------------------------------------------

        model_key = (
            req.params.get("model") or "custom_cnn"
        ).strip().lower()

        if model_key not in MODEL_BLOBS:
            return json_response({
                "error": "Invalid model selection.",
                "available_models": list(MODEL_BLOBS.keys()),
            }, status_code=400)

        # --------------------------------------------------
        # 2. Read the uploaded image
        # --------------------------------------------------

        image_bytes = req.get_body()

        if not image_bytes:
            return json_response({
                "error": "No image was provided.",
            }, status_code=400)

        # --------------------------------------------------
        # 3. Decode and preprocess the image
        # --------------------------------------------------

        try:
            with Image.open(BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
                image = image.resize(IMAGE_SIZE)

                image_array = np.asarray(
                    image,
                    dtype=np.float32,
                )

        except (UnidentifiedImageError, OSError):
            return json_response({
                "error": "The uploaded file is not a valid image.",
            }, status_code=400)

        # The model is assumed to contain its own preprocessing.
        # Do not normalize or apply preprocess_input here unless
        # required by the specific model's training pipeline.
        input_batch = np.expand_dims(
            image_array,
            axis=0,
        )

        # --------------------------------------------------
        # 4. Load the selected model
        # --------------------------------------------------

        model = load_model(model_key)

        # --------------------------------------------------
        # 5. Perform prediction
        # --------------------------------------------------

        prediction = model.predict(
            input_batch,
            verbose=0,
        )

        prediction_values = np.asarray(prediction).reshape(-1)

        if prediction_values.size != 1:
            raise ValueError(
                "Expected a single output value from the binary "
                "classification model."
            )

        score = float(prediction_values[0])

        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                "The model returned an invalid probability score."
            )

        # Assumes model output is the probability of PNEUMONIA,
        # with 0 = NORMAL and 1 = PNEUMONIA.
        predicted_index = (
            1 if score >= PREDICTION_THRESHOLD else 0
        )

        predicted_class = CLASS_NAMES[predicted_index]

        # --------------------------------------------------
        # 6. Return the prediction
        # --------------------------------------------------

        return json_response({
            "status": "success",
            "model": model_key,
            "prediction": predicted_class,
            "pneumonia_score": round(score, 6),
            "threshold": PREDICTION_THRESHOLD,
            "image_size": "224x224",
            "notice": (
                "Educational research output only. "
                "Not intended for clinical diagnosis."
            ),
        })

    except Exception:
        logging.exception(
            "Prediction request failed."
        )

        return json_response({
            "error": (
                "Prediction failed. Check the Function App logs."
            ),
        }, status_code=500)
