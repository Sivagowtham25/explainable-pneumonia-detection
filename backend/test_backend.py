"""
Automated backend verification test suite for FastAPI REST API endpoints.
Tests health checks, model catalog, and image inference with Grad-CAM generation.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings

client = TestClient(app)

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT_DIR / "Dataset" / "chest_xray" / "test"
NORMAL_SAMPLE = next((TEST_DIR / "NORMAL").glob("*.jpeg"))
PNEUMONIA_SAMPLE = next((TEST_DIR / "PNEUMONIA").glob("*.jpeg"))


def test_health_endpoint():
    """Verify system health, engine version, and available models."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["version"] == settings.VERSION
    assert "mobilenetv2" in data["available_models"]
    assert "baseline" in data["available_models"]
    assert data["system_ready"] is True
    print("[PASS] test_health_endpoint passed successfully.")


def test_models_list_endpoint():
    """Verify registered models catalog and benchmark metrics."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 2
    model_ids = [m["model_id"] for m in data["models"]]
    assert "mobilenetv2" in model_ids
    assert "baseline" in model_ids

    # Check metrics existence
    for m in data["models"]:
        metrics = m["benchmark_metrics"]
        assert metrics["accuracy"] > 70.0
        assert metrics["roc_auc"] > 0.90
        assert metrics["pneumonia_recall"] > 95.0
    print("[PASS] test_models_list_endpoint passed successfully.")


def test_specific_model_endpoint():
    """Verify individual model detail endpoint."""
    response = client.get("/api/v1/models/mobilenetv2")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "mobilenetv2"
    assert data["target_conv_layer"] == "Conv_1"

    response_404 = client.get("/api/v1/models/nonexistent_model")
    assert response_404.status_code == 404
    print("[PASS] test_specific_model_endpoint passed successfully.")


def test_predict_mobilenetv2_normal():
    """Verify prediction and Grad-CAM overlay on a normal chest radiograph."""
    with open(NORMAL_SAMPLE, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"file": (NORMAL_SAMPLE.name, file_bytes, "image/jpeg")},
        data={"model_id": "mobilenetv2", "alpha": 0.4, "generate_cam": "true"},
    )
    assert response.status_code == 200, f"Predict failed: {response.text}"
    data = response.json()

    assert data["filename"] == NORMAL_SAMPLE.name
    assert data["model_id"] == "mobilenetv2"
    assert data["predicted_label"] in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= data["confidence"] <= 100.0
    assert 0.0 <= data["prob_pneumonia"] <= 100.0
    assert 0.0 <= data["prob_normal"] <= 100.0
    assert data["inference_time_ms"] > 0
    assert data["gradcam_generated"] is True
    assert data["gradcam_overlay_base64"].startswith("data:image/png;base64,")
    assert data["colored_heatmap_base64"].startswith("data:image/png;base64,")
    assert data["original_image_base64"].startswith("data:image/png;base64,")
    assert len(data["clinical_notes"]) > 10
    print(f"[PASS] test_predict_mobilenetv2_normal passed. Result: {data['predicted_label']} ({data['confidence']}%) in {data['inference_time_ms']}ms")


def test_predict_baseline_pneumonia():
    """Verify prediction and Grad-CAM overlay on a pneumonia chest radiograph with Baseline CNN."""
    with open(PNEUMONIA_SAMPLE, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"file": (PNEUMONIA_SAMPLE.name, file_bytes, "image/jpeg")},
        data={"model_id": "baseline", "alpha": 0.5, "generate_cam": "true"},
    )
    assert response.status_code == 200, f"Predict failed: {response.text}"
    data = response.json()

    assert data["filename"] == PNEUMONIA_SAMPLE.name
    assert data["model_id"] == "baseline"
    assert data["predicted_label"] == "PNEUMONIA"
    assert data["is_pneumonia"] is True
    assert data["confidence"] > 80.0
    assert data["gradcam_generated"] is True
    assert data["gradcam_overlay_base64"].startswith("data:image/png;base64,")
    print(f"[PASS] test_predict_baseline_pneumonia passed. Result: {data['predicted_label']} ({data['confidence']}%) in {data['inference_time_ms']}ms")


def test_predict_baseline_normal_case():
    """Verify prediction on a true normal radiograph confirming NORMAL label and confidence."""
    normal_case_path = TEST_DIR / "NORMAL" / "IM-0007-0001.jpeg"
    with open(normal_case_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"file": (normal_case_path.name, file_bytes, "image/jpeg")},
        data={"model_id": "baseline", "alpha": 0.4, "generate_cam": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == "NORMAL"
    assert data["is_pneumonia"] is False
    assert data["prob_normal"] > 50.0
    assert data["gradcam_overlay_base64"].startswith("data:image/png;base64,")
    print(f"[PASS] test_predict_baseline_normal_case passed. Result: NORMAL ({data['confidence']}%)")


def test_predict_validation_errors():
    """Verify bad inputs trigger appropriate HTTP error responses."""
    # 1. Invalid file extension (.txt)
    response_txt = client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"Invalid text data", "text/plain")},
        data={"model_id": "mobilenetv2"},
    )
    assert response_txt.status_code == 400
    assert "Unsupported image format" in response_txt.json()["detail"]

    # 2. Unknown model_id
    with open(NORMAL_SAMPLE, "rb") as f:
        file_bytes = f.read()
    response_bad_model = client.post(
        "/api/v1/predict",
        files={"file": ("test.png", file_bytes, "image/png")},
        data={"model_id": "invalid_model_name"},
    )
    assert response_bad_model.status_code == 400
    assert "Unknown model_id" in response_bad_model.json()["detail"]
    print("[PASS] test_predict_validation_errors passed successfully.")


if __name__ == "__main__":
    print("=== Running Backend API Verification Suite ===")
    test_health_endpoint()
    test_models_list_endpoint()
    test_specific_model_endpoint()
    test_predict_mobilenetv2_normal()
    test_predict_baseline_pneumonia()
    test_predict_baseline_normal_case()
    test_predict_validation_errors()
    print("\nALL BACKEND API TESTS COMPLETED SUCCESSFULLY!")
