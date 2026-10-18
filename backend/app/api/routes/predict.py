"""
Image prediction and Grad-CAM explainability endpoint.
"""

from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.core.config import settings
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.model_service import ModelService
from backend.app.api.deps import get_service

router = APIRouter(tags=["Inference & Explainability"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Pneumonia & Generate Grad-CAM Heatmap",
    description=(
        "Upload a chest radiograph image (JPEG, PNG). The endpoint runs deep learning inference, "
        "computes class probabilities, and generates an anatomical Grad-CAM visual heatmap overlay."
    ),
)
async def predict_xray(
    file: UploadFile = File(..., description="Chest X-ray image file (JPEG, PNG)"),
    model_id: str = Form(
        default=settings.DEFAULT_MODEL_ID,
        description="Target model identifier ('mobilenetv2' or 'baseline')",
    ),
    alpha: float = Form(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Grad-CAM heatmap overlay opacity blending factor (0.0 to 1.0)",
    ),
    generate_cam: bool = Form(
        default=True,
        description="Whether to generate and encode Grad-CAM saliency heatmaps",
    ),
    service: ModelService = Depends(get_service),
) -> PredictionResponse:
    """
    Handles chest radiograph upload, validation, model inference, and Grad-CAM generation.
    """
    # 1. Validate file extension
    filename = file.filename or "uploaded_xray.png"
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image format '.{ext}'. Supported extensions: "
                f"{sorted(list(settings.ALLOWED_EXTENSIONS))}"
            ),
        )

    # 2. Read and validate binary content
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid chest radiograph.",
        )

    if len(contents) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_BYTES / (1024*1024):.1f} MB.",
        )

    # 3. Model ID validation
    if model_id.lower().strip() not in service.get_registered_ids():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model_id '{model_id}'. Allowed choices: {service.get_registered_ids()}",
        )

    # 4. Execute prediction and Grad-CAM
    try:
        response = service.predict_and_explain(
            image_bytes=contents,
            filename=filename,
            model_id=model_id,
            alpha=alpha,
            generate_cam=generate_cam,
        )
        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(exc)}",
        ) from exc
