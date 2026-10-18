"""
Pydantic data schemas for API requests, model outputs, and health status responses.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ModelEvaluationMetrics(BaseModel):
    """
    Standardized benchmark metrics on the official 624-image test set.
    """
    accuracy: float = Field(..., description="Overall test accuracy percentage")
    roc_auc: float = Field(..., description="Area Under the ROC Curve")
    pneumonia_recall: float = Field(..., description="Pneumonia sensitivity / recall percentage")
    pneumonia_precision: float = Field(..., description="Pneumonia precision percentage")
    f1_score: float = Field(..., description="Harmonic mean of precision and recall")
    normal_specificity: float = Field(..., description="Normal specificity percentage")
    test_support: int = Field(624, description="Total number of evaluated test set radiographs")


class ModelMetadata(BaseModel):
    """
    Descriptive metadata, architecture details, and status for a registered model.
    """
    model_id: str = Field(..., description="Unique model identifier (e.g. 'baseline', 'mobilenetv2')")
    name: str = Field(..., description="Human-readable model name")
    architecture: str = Field(..., description="Underlying neural network architecture")
    total_parameters: int = Field(..., description="Total trainable and non-trainable parameter count")
    target_conv_layer: str = Field(..., description="Target convolutional layer used for Grad-CAM activations")
    is_loaded: bool = Field(..., description="Whether the model weights are currently resident in GPU/CPU memory")
    checkpoint_size_mb: float = Field(..., description="Size of model file on disk in megabytes")
    benchmark_metrics: ModelEvaluationMetrics = Field(..., description="Official test set evaluation results")


class ModelsListResponse(BaseModel):
    """
    Response schema returning all registered models and active default.
    """
    models: List[ModelMetadata]
    default_model_id: str


class PredictionResponse(BaseModel):
    """
    Diagnostic inference result along with optional Grad-CAM visual explainability.
    """
    filename: str = Field(..., description="Name of the evaluated image file")
    model_id: str = Field(..., description="Identifier of the model used for inference")
    model_name: str = Field(..., description="Full descriptive name of the model")
    predicted_label: str = Field(..., description="Predicted diagnostic class: 'NORMAL' or 'PNEUMONIA'")
    is_pneumonia: bool = Field(..., description="Boolean flag: True if pneumonia is detected")
    confidence: float = Field(..., description="Confidence score percentage for the predicted class")
    prob_pneumonia: float = Field(..., description="Estimated probability of pneumonia (0.0 to 100.0%)")
    prob_normal: float = Field(..., description="Estimated probability of normal radiograph (0.0 to 100.0%)")
    inference_time_ms: float = Field(..., description="Inference and Grad-CAM generation latency in milliseconds")
    gradcam_generated: bool = Field(..., description="True if Grad-CAM heatmaps were successfully generated")
    gradcam_overlay_base64: Optional[str] = Field(None, description="Base64 PNG data URI of Grad-CAM overlay on X-ray")
    colored_heatmap_base64: Optional[str] = Field(None, description="Base64 PNG data URI of raw JET colormap heatmap")
    original_image_base64: Optional[str] = Field(None, description="Base64 PNG data URI of preprocessed input radiograph")
    target_conv_layer: str = Field(..., description="Convolutional layer extracted for visual attention")
    alpha: float = Field(..., description="Heatmap overlay opacity factor utilized")
    clinical_notes: str = Field(..., description="Automated interpretability summary of visual findings")


class HealthResponse(BaseModel):
    """
    System health, readiness, and runtime environment status.
    """
    status: str = Field("healthy", description="System health status string")
    version: str = Field(..., description="API version")
    tensorflow_version: str = Field(..., description="Underlying TensorFlow engine version")
    default_model: str = Field(..., description="Default model ID selected for inference")
    available_models: List[str] = Field(..., description="List of registered model IDs available on disk")
    loaded_models: List[str] = Field(..., description="List of models currently loaded into memory")
    system_ready: bool = Field(..., description="True if at least one model is loaded and ready for queries")
