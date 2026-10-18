"""
Application configuration module for the Pneumonia Classification & Explainability API.
"""

from pathlib import Path
from typing import List, Set, Tuple
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """
    Central API configuration settings.
    """
    PROJECT_NAME: str = "Explainable Pneumonia Detection REST API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Medical AI diagnostic service providing real-time pneumonia classification "
        "and anatomical Grad-CAM visual heatmaps from chest radiographs."
    )
    API_V1_STR: str = "/api/v1"

    # Base workspace path resolution
    # config.py is at: e:/Pnuemonia Detection/backend/app/core/config.py
    # parents[3] is: e:/Pnuemonia Detection
    BASE_DIR: Path = Path(__file__).resolve().parents[3]
    MODELS_DIR: Path = BASE_DIR / "models"

    # Model Checkpoints & Target Conv Layers for Grad-CAM
    BASELINE_MODEL_ID: str = "baseline"
    BASELINE_MODEL_NAME: str = "Custom Baseline 4-Block CNN"
    BASELINE_MODEL_PATH: Path = MODELS_DIR / "baseline_cnn_best.keras"
    BASELINE_CONV_LAYER: str = "conv4_features"

    MOBILENET_MODEL_ID: str = "mobilenetv2"
    MOBILENET_MODEL_NAME: str = "Fine-Tuned MobileNetV2 Transfer Model"
    MOBILENET_MODEL_PATH: Path = MODELS_DIR / "mobilenetv2_transfer_best.keras"
    MOBILENET_CONV_LAYER: str = "Conv_1"

    DEFAULT_MODEL_ID: str = "mobilenetv2"
    IMAGE_SIZE: Tuple[int, int] = (224, 224)

    # Security & CORS settings (allows modern frontend dev servers: Vite 5173, CRA 3000)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "*"
    ]

    # File upload validation constraints
    ALLOWED_EXTENSIONS: Set[str] = {"jpg", "jpeg", "png", "bmp", "webp"}
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB max image upload size

    # Medical diagnosis class labels
    CLASS_LABELS: dict = {
        0: "NORMAL",
        1: "PNEUMONIA"
    }


settings = Settings()
