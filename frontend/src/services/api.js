/**
 * API Service for communicating with the FastAPI Pneumonia Detection Backend.
 */

const API_BASE = '/api/v1';

/**
 * Checks backend health, loaded models, and readiness.
 */
export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return await response.json();
}

/**
 * Retrieves the catalog of models and their test set benchmark metrics.
 */
export async function fetchModels() {
  const response = await fetch(`${API_BASE}/models`);
  if (!response.ok) {
    throw new Error(`Failed to load models list: ${response.status}`);
  }
  return await response.json();
}

/**
 * Submits a chest X-ray file for classification and Grad-CAM generation.
 *
 * @param {File|Blob} file - Chest radiograph file.
 * @param {string} filename - Original or assigned filename.
 * @param {string} modelId - 'mobilenetv2' or 'baseline'.
 * @param {number} alpha - Grad-CAM overlay opacity (0.0 to 1.0).
 * @param {boolean} generateCam - Whether to compute Grad-CAM heatmaps.
 */
export async function predictImage(file, filename = 'radiograph.jpeg', modelId = 'mobilenetv2', alpha = 0.4, generateCam = true) {
  const formData = new FormData();
  formData.append('file', file, filename);
  formData.append('model_id', modelId);
  formData.append('alpha', alpha.toString());
  formData.append('generate_cam', generateCam ? 'true' : 'false');

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction request failed with status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Fetches a static sample image from public/samples and returns a File object.
 */
export async function loadSampleFile(sampleUrl, filename) {
  const response = await fetch(sampleUrl);
  if (!response.ok) {
    throw new Error(`Failed to load sample image: ${sampleUrl}`);
  }
  const blob = await response.blob();
  return new File([blob], filename, { type: 'image/jpeg' });
}
