"""
Health check and system status endpoints.
"""

from fastapi import APIRouter, Depends
import tensorflow as tf

from backend.app.core.config import settings
from backend.app.schemas.prediction import HealthResponse
from backend.app.services.model_service import ModelService
from backend.app.api.deps import get_service

router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness",
    description="Returns API service health, loaded models, and underlying TensorFlow engine status.",
)
def get_health(service: ModelService = Depends(get_service)) -> HealthResponse:
    """
    Evaluates backend readiness and loaded model states.
    """
    available = [m for m in service.get_registered_ids() if service.is_model_available(m)]
    loaded = service.get_loaded_ids()
    ready = len(available) > 0

    return HealthResponse(
        status="healthy" if ready else "degraded",
        version=settings.VERSION,
        tensorflow_version=tf.__version__,
        default_model=settings.DEFAULT_MODEL_ID,
        available_models=available,
        loaded_models=loaded,
        system_ready=ready,
    )
