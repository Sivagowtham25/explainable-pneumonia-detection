"""
FastAPI dependency injection utilities.
"""

from backend.app.services.model_service import ModelService, get_model_service


def get_service() -> ModelService:
    """
    FastAPI dependency yielding the ModelService singleton.
    """
    return get_model_service()
