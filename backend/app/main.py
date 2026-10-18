"""
FastAPI application entrypoint for Explainable Pneumonia Detection Service.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app.core.config import settings
from backend.app.api.routes import health_router, models_router, predict_router
from backend.app.services.model_service import get_model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle management.
    Pre-warms deep learning models so first-request latency is minimized.
    """
    print(f"=== Starting {settings.PROJECT_NAME} (v{settings.VERSION}) ===")
    service = get_model_service()
    try:
        service.preload_defaults()
        print(f"Default model ready. Loaded models: {service.get_loaded_ids()}")
    except Exception as exc:
        print(f"[Warning] Model preloading encountered an issue: {exc}")

    yield

    print("=== Shutting down Pneumonia Detection API service ===")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS) for React / frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 sub-routers
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(models_router, prefix=settings.API_V1_STR)
app.include_router(predict_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
def root_redirect():
    """
    Redirects root index directly to the interactive Swagger OpenAPI documentation.
    """
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
