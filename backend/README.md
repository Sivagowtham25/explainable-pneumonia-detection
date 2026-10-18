# Backend REST API Service (`backend/`)

This directory houses the production-ready **FastAPI** web service for medical diagnostic inference, dual-model switching, and real-time **Grad-CAM visual explainability** from chest radiographs.

---

## 1. Directory Architecture

```text
backend/
├── app/
│   ├── __init__.py               # App package initialization
│   ├── main.py                   # FastAPI setup, lifespan preheating, CORS, and routing
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py             # Global settings (paths, model metadata, CORS, constraints)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py               # Dependency injection providers (ModelService singleton)
│   │   └── routes/
│   │       ├── __init__.py       # Router aggregators
│   │       ├── health.py         # GET /api/v1/health (system health & engine status)
│   │       ├── models.py         # GET /api/v1/models (catalog & benchmark metrics)
│   │       └── predict.py        # POST /api/v1/predict (multipart image upload & Grad-CAM)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py         # Pydantic schemas (requests, responses, model metrics)
│   └── services/
│       ├── __init__.py
│       ├── cam_service.py        # Image validation, serialization & base64 Data URI encoding
│       └── model_service.py      # Dual-model manager, caching, inference & Grad-CAM orchestration
├── test_backend.py               # Automated end-to-end integration test suite
└── README.md                     # Backend documentation
```

---

## 2. API Endpoints

### 1. Health & Engine Status
- **Method**: `GET`
- **Route**: `/api/v1/health`
- **Description**: Verifies service status, TensorFlow version, available models on disk, and models preloaded in memory.
- **Sample Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "tensorflow_version": "2.21.0",
  "default_model": "mobilenetv2",
  "available_models": ["mobilenetv2", "baseline"],
  "loaded_models": ["mobilenetv2"],
  "system_ready": true
}
```

### 2. Models Catalog & Benchmarks
- **Method**: `GET`
- **Route**: `/api/v1/models`
- **Description**: Returns architectural specifications, parameter counts, and official 624-image test set metrics for both models.
- **Route Detail**: `/api/v1/models/{model_id}` (e.g. `mobilenetv2` or `baseline`)

### 3. Chest Radiograph Diagnosis & Grad-CAM
- **Method**: `POST`
- **Route**: `/api/v1/predict`
- **Payload (`multipart/form-data`)**:
  - `file`: Chest X-ray image (JPEG, PNG, etc.)
  - `model_id` *(optional, default: "mobilenetv2")*: `"mobilenetv2"` or `"baseline"`
  - `alpha` *(optional, default: 0.4)*: Heatmap overlay blending factor ($0.0 \le \alpha \le 1.0$)
  - `generate_cam` *(optional, default: true)*: Boolean flag to generate visual heatmaps
- **Sample Response**:
```json
{
  "filename": "IM-0007-0001.jpeg",
  "model_id": "baseline",
  "model_name": "Custom Baseline 4-Block CNN",
  "predicted_label": "NORMAL",
  "is_pneumonia": false,
  "confidence": 53.59,
  "prob_pneumonia": 46.41,
  "prob_normal": 53.59,
  "inference_time_ms": 306.45,
  "gradcam_generated": true,
  "gradcam_overlay_base64": "data:image/png;base64,iVBORw0KGgo...",
  "colored_heatmap_base64": "data:image/png;base64,iVBORw0KGgo...",
  "original_image_base64": "data:image/png;base64,iVBORw0KGgo...",
  "target_conv_layer": "conv4_features",
  "alpha": 0.4,
  "clinical_notes": "Clear lung fields with no focal alveolar consolidation detected (53.6% confidence)..."
}
```

---

## 3. Running the Server

To launch the FastAPI development server with automatic reloading:

```powershell
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once started:
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 4. Running the Test Suite

Execute the automated test suite directly:

```powershell
python backend/test_backend.py
```
