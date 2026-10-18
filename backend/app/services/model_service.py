"""
Model management and inference execution service.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import tensorflow as tf

from backend.app.core.config import settings
from backend.app.schemas.prediction import (
    ModelEvaluationMetrics,
    ModelMetadata,
    PredictionResponse,
)
from backend.app.services.cam_service import bytes_to_pil, image_to_base64
from src.explainability.gradcam import explain_chest_xray


class ModelService:
    """
    Singleton service managing deep learning model lifecycles, inference,
    and Grad-CAM visual explainability.
    """

    MODEL_REGISTRY = {
        "mobilenetv2": {
            "name": settings.MOBILENET_MODEL_NAME,
            "architecture": "MobileNetV2 (ImageNet Pre-trained) + Fine-Tuned Custom Classification Head",
            "path": settings.MOBILENET_MODEL_PATH,
            "target_conv_layer": settings.MOBILENET_CONV_LAYER,
            "total_parameters": 2427201,
            "metrics": ModelEvaluationMetrics(
                accuracy=78.53,
                roc_auc=0.9566,
                pneumonia_recall=98.72,
                pneumonia_precision=74.90,
                f1_score=85.18,
                normal_specificity=44.87,
                test_support=624,
            ),
        },
        "baseline": {
            "name": settings.BASELINE_MODEL_NAME,
            "architecture": "Custom 4-Block Convolutional Neural Network with BatchNorm & Dropout",
            "path": settings.BASELINE_MODEL_PATH,
            "target_conv_layer": settings.BASELINE_CONV_LAYER,
            "total_parameters": 258049,
            "metrics": ModelEvaluationMetrics(
                accuracy=83.17,
                roc_auc=0.9252,
                pneumonia_recall=98.21,
                pneumonia_precision=79.63,
                f1_score=87.94,
                normal_specificity=58.12,
                test_support=624,
            ),
        },
    }

    def __init__(self):
        self._loaded_models: Dict[str, tf.keras.Model] = {}

    def get_registered_ids(self) -> List[str]:
        """Returns all registered model IDs."""
        return list(self.MODEL_REGISTRY.keys())

    def get_loaded_ids(self) -> List[str]:
        """Returns all currently loaded model IDs."""
        return list(self._loaded_models.keys())

    def is_model_available(self, model_id: str) -> bool:
        """Checks if a model checkpoint file exists on disk."""
        if model_id not in self.MODEL_REGISTRY:
            return False
        return self.MODEL_REGISTRY[model_id]["path"].is_file()

    def get_model(self, model_id: str) -> tf.keras.Model:
        """
        Loads and caches the requested model in memory.
        """
        model_id = model_id.lower().strip()
        if model_id not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_id: '{model_id}'. Available models: {list(self.MODEL_REGISTRY.keys())}"
            )

        if model_id not in self._loaded_models:
            model_info = self.MODEL_REGISTRY[model_id]
            model_path: Path = model_info["path"]
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at: {model_path}. "
                    "Ensure Phase 3 or Phase 4 training checkpoints have been preserved."
                )
            print(f"[ModelService] Loading model '{model_id}' from: {model_path} ...")
            model = tf.keras.models.load_model(str(model_path))
            self._loaded_models[model_id] = model
            print(f"[ModelService] Model '{model_id}' successfully loaded and cached in memory.")

        return self._loaded_models[model_id]

    def preload_defaults(self) -> None:
        """
        Preloads the default model into memory during server startup.
        """
        default_id = settings.DEFAULT_MODEL_ID
        if self.is_model_available(default_id):
            self.get_model(default_id)
        elif self.is_model_available("baseline"):
            self.get_model("baseline")

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """
        Constructs metadata object for a specific model.
        """
        model_id = model_id.lower().strip()
        if model_id not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model_id: '{model_id}'")

        info = self.MODEL_REGISTRY[model_id]
        path: Path = info["path"]
        size_mb = round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0.0

        return ModelMetadata(
            model_id=model_id,
            name=info["name"],
            architecture=info["architecture"],
            total_parameters=info["total_parameters"],
            target_conv_layer=info["target_conv_layer"],
            is_loaded=(model_id in self._loaded_models),
            checkpoint_size_mb=size_mb,
            benchmark_metrics=info["metrics"],
        )

    def list_models_metadata(self) -> List[ModelMetadata]:
        """
        Lists metadata for all registered models.
        """
        return [self.get_model_metadata(m_id) for m_id in self.MODEL_REGISTRY]

    def predict_and_explain(
        self,
        image_bytes: bytes,
        filename: str = "chest_xray.png",
        model_id: str = "mobilenetv2",
        alpha: float = 0.4,
        generate_cam: bool = True,
    ) -> PredictionResponse:
        """
        Executes prediction and Grad-CAM saliency extraction for an input image.

        Parameters:
        -----------
        image_bytes : bytes
            Binary image content from upload.
        filename : str
            Source image filename.
        model_id : str
            Identifier of model to use ('mobilenetv2' or 'baseline').
        alpha : float
            Grad-CAM overlay blending opacity (0.0 to 1.0).
        generate_cam : bool
            Whether to compute Grad-CAM heatmap visualization.

        Returns:
        --------
        PredictionResponse
            Validated Pydantic response containing diagnostic prediction, probabilities,
            and base64-encoded visual heatmaps.
        """
        start_time = time.perf_counter()

        model_id = model_id.lower().strip()
        if model_id not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_id}' is not supported. Choose from: {list(self.MODEL_REGISTRY.keys())}"
            )

        # Validate and convert image bytes
        pil_image = bytes_to_pil(image_bytes)

        # Retrieve model and configuration
        model = self.get_model(model_id)
        model_info = self.MODEL_REGISTRY[model_id]
        target_layer = model_info["target_conv_layer"]

        # Run inference and Grad-CAM
        explanation = explain_chest_xray(
            img_input=pil_image,
            model=model,
            last_conv_layer_name=target_layer,
            target_size=settings.IMAGE_SIZE,
            alpha=alpha,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        predicted_label = explanation["predicted_label"]
        confidence = explanation["confidence"]
        prob_pneumonia = explanation["prob_pneumonia"]
        prob_normal = explanation["prob_normal"]
        is_pneumonia = (predicted_label == "PNEUMONIA")

        # Encode visualizations to base64 Data URIs
        gradcam_overlay_b64 = None
        colored_heatmap_b64 = None
        original_image_b64 = None

        if generate_cam:
            gradcam_overlay_b64 = image_to_base64(explanation["blended_image"], format="PNG")
            colored_heatmap_b64 = image_to_base64(explanation["colored_heatmap"], format="PNG")
            original_image_b64 = image_to_base64(explanation["original_image"], format="PNG")

        # Compose automated clinical explainability summary
        if is_pneumonia:
            notes = (
                f"Pathological opacification or pulmonary consolidation identified with {confidence:.1f}% confidence. "
                f"Grad-CAM visual attention highlights active infection loci in the lung fields."
            )
        else:
            notes = (
                f"Clear lung fields with no focal alveolar consolidation detected ({confidence:.1f}% confidence). "
                f"Grad-CAM reflects diffuse baseline activations without pathological focal clustering."
            )

        return PredictionResponse(
            filename=filename,
            model_id=model_id,
            model_name=model_info["name"],
            predicted_label=predicted_label,
            is_pneumonia=is_pneumonia,
            confidence=confidence,
            prob_pneumonia=prob_pneumonia,
            prob_normal=prob_normal,
            inference_time_ms=elapsed_ms,
            gradcam_generated=generate_cam,
            gradcam_overlay_base64=gradcam_overlay_b64,
            colored_heatmap_base64=colored_heatmap_b64,
            original_image_base64=original_image_b64,
            target_conv_layer=target_layer,
            alpha=alpha,
            clinical_notes=notes,
        )


# Global singleton instance
_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """
    Returns or creates the shared ModelService singleton instance.
    """
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance
