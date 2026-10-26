import json
import logging
import os
from io import BytesIO

import azure.functions as func
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient


app = func.FunctionApp(
    http_auth_level=func.AuthLevel.ANONYMOUS
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

STORAGE_ACCOUNT = os.environ["rgpneumoscan8401"]
MODEL_CONTAINER = os.environ["models"]

CUSTOM_CNN_BLOB = os.environ["baseline_cnn_best.keras"]
MOBILENET_BLOB = os.environ["mobilenetv2_transfer_best.keras"]

MODEL_BLOBS = {
    "custom_cnn": CUSTOM_CNN_BLOB,
    "mobilenetv2": MOBILENET_BLOB,
}

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = (224, 224)

# Cached in memory for subsequent requests.
_loaded_models = {}
_blob_service = None


def get_blob_service():
    """Create a Blob Storage client using managed identity."""
    global _blob_service

    if _blob_service is None:
        account = STORAGE_ACCOUNT.strip().rstrip("/")

        if account.startswith("https://"):
            account_url = account
        else:
            account_url = (
                f"https://{account}.blob.core.windows.net"
            )

        credential = ManagedIdentityCredential()

        _blob_service = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )

    return _blob_service


def load_model(model_key):
    """Download a model from Blob Storage and cache it."""
    if model_key not in MODEL_BLOBS:
        raise ValueError("Unsupported model selection.")

    if model_key not in _loaded_models:
        blob_name = MODEL_BLOBS[model_key]

        logging.info(
            "Loading model '%s' from blob '%s'.",
            model_key,
            blob_name,
        )

        blob_client = get_blob_service().get_blob_client(
            container=MODEL_CONTAINER,
            blob=blob_name,
        )

        model_bytes = blob_client.download_blob().readall()

        model = tf.keras.models.load_model(
            BytesIO(model_bytes),
            compile=False,
        )

        _loaded_models[model_key] = model

        logging.info("Model '%s' loaded successfully.", model_key)

    return _loaded_models[model_key]


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({
            "status": "success",
            "application": "PneumoScan AI",
            "message": "Azure Functions API is running.",
            "available_models": list(MODEL_BLOBS.keys()),
        }),
        status_code=200,
        mimetype="application/json",
    )


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """
    Accept raw image bytes.

    Example:
      POST /api/predict?model=custom_cnn

    The request body must contain the image bytes.
    Set Content-Type to image/jpeg or image/png.
    """

    try:
        model_key = (
            req.params.get("model") or "custom_cnn"
        ).lower()

        if model_key not in MODEL_BLOBS:
            return func.HttpResponse(
                json.dumps({
                    "error": "Invalid model selection.",
                    "available_models": list(MODEL_BLOBS.keys()),
                }),
                status_code=400,
                mimetype="application/json",
            )

        image_bytes = req.get_body()

        if not image_bytes:
            return func.HttpResponse(
                json.dumps({
                    "error": "No image was provided."
                }),
                status_code=400,
                mimetype="application/json",
            )

        # Open image and convert it to RGB.
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
                image = image.resize(IMAGE_SIZE)

                image_array = np.asarray(
                    image,
                    dtype=np.float32,
                )

        except (UnidentifiedImageError, OSError):
            return func.HttpResponse(
                json.dumps({
                    "error": "The uploaded file is not a valid image."
                }),
                status_code=400,
                mimetype="application/json",
            )

        # Models contain their own preprocessing layers.
        # Do not divide by 255 or apply preprocess_input here.
        input_batch = np.expand_dims(image_array, axis=0)

        model = load_model(model_key)

        prediction = model.predict(
            input_batch,
            verbose=0,
        )

        score = float(np.asarray(prediction).reshape(-1)[0])

        # The model was trained with binary labels:
        # 0 = NORMAL, 1 = PNEUMONIA.
        predicted_index = 1 if score >= 0.5 else 0
        predicted_class = CLASS_NAMES[predicted_index]

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "model": model_key,
                "prediction": predicted_class,
                "pneumonia_score": round(score, 6),
                "threshold": 0.5,
                "image_size": "224x224",
                "notice": (
                    "Educational research output only. "
                    "Not intended for clinical diagnosis."
                ),
            }),
            status_code=200,
            mimetype="application/json",
        )

    except Exception:
        logging.exception("Prediction request failed.")

        return func.HttpResponse(
            json.dumps({
                "error": "Prediction failed. Check the Function App logs."
            }),
            status_code=500,
            mimetype="application/json",
        )
