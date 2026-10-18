"""
Model registry and benchmark performance endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import settings
from backend.app.schemas.prediction import ModelMetadata, ModelsListResponse
from backend.app.services.model_service import ModelService
from backend.app.api.deps import get_service

router = APIRouter(prefix="/models", tags=["Models & Benchmarks"])


@router.get(
    "",
    response_model=ModelsListResponse,
    summary="List All Available Deep Learning Models",
    description="Returns metadata, architecture details, and official test benchmark metrics for all models.",
)
def list_models(service: ModelService = Depends(get_service)) -> ModelsListResponse:
    """
    Lists registered models, architectures, parameter counts, and test set metrics.
    """
    metadata_list = service.list_models_metadata()
    return ModelsListResponse(
        models=metadata_list,
        default_model_id=settings.DEFAULT_MODEL_ID,
    )


@router.get(
    "/{model_id}",
    response_model=ModelMetadata,
    summary="Get Specific Model Details",
    description="Returns comprehensive architecture and benchmark data for a single model ID.",
)
def get_model_details(model_id: str, service: ModelService = Depends(get_service)) -> ModelMetadata:
    """
    Retrieves metadata for a specific model ('mobilenetv2' or 'baseline').
    """
    try:
        return service.get_model_metadata(model_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
