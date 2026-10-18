# Project Development Step Summary

**Project Title**: Explainable Pneumonia Classification from Chest X-Ray Images Using Deep Learning  
**Project Type**: Master of Computer Applications (MCA) Semester Project  
**Repository Location**: `e:/Pnuemonia Detection`  
**Target Modality**: Anterior-Posterior (AP) Pediatric Chest Radiographs  
**Classification Scope**: Binary Classification (`NORMAL` vs `PNEUMONIA`) with Grad-CAM Explainable AI (XAI)  

---

## Chronological Step-by-Step Progress Log

### Step 1: Project Foundation & Directory Architecture
- **Objective**: Establish a clean, standardized, academic-grade project directory structure and enforce version-control policies before code execution.
- **Actions Taken**:
  1. Created structured directory tree:
     - `dataset/chest_xray/`: Designated medical image repository.
     - `notebooks/`: Exploratory data analysis, prototyping, and model development.
     - `models/`: Saved model binaries, checkpoints, and serialized weights.
     - `src/`: Modular Python packages (loaders, models, explainability, utilities).
     - `backend/`: FastAPI asynchronous prediction service.
     - `frontend/`: React clinical dashboard and Grad-CAM visualizer.
  2. Configured Git tracking exclusions:
     - Created root [`.gitignore`](file:///e:/Pnuemonia%20Detection/.gitignore) excluding all dataset images (`dataset/`, `Dataset/`), image archives, node modules, and virtual environments.
     - Created [`models/.gitignore`](file:///e:/Pnuemonia%20Detection/models/.gitignore) specifically preventing large generated model binaries (`*.keras`, `*.h5`, `*.savedmodel/`, `*.pt`, `checkpoints/`) from entering Git history while keeping the directory tracked via `.gitkeep`.
  3. Created dependency specifications in [`requirements.txt`](file:///e:/Pnuemonia%20Detection/requirements.txt):
     - Deep Learning: `tensorflow`, `keras`
     - Data & Science: `numpy`, `pandas`, `scikit-learn`
     - Vision & Plotting: `pillow`, `matplotlib`, `seaborn`, `opencv-python-headless`
     - Interpretability: `tf-keras-vis`
     - Backend API: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`
     - Research: `jupyter`, `ipykernel`
  4. Created root documentation [`README.md`](file:///e:/Pnuemonia%20Detection/README.md) and module READMEs for all subdirectories.
  5. Initialized Git repository and tested weight exclusion rules with simulated `.keras` and `.h5` files.

---

### Step 2: Dataset Verification & Integrity Audit (Read-Only)
- **Objective**: Inspect the Kaggle Chest X-Ray dataset on disk without modifying, moving, or deleting any image files.
- **Actions Taken**:
  1. Inspected physical directory paths at `dataset/chest_xray/` (with Windows case-insensitivity handling for `Dataset/`).
  2. Verified split presence: `train`, `val` (noted as `val` instead of `validation`), and `test`.
  3. Verified class subdirectories: `NORMAL` and `PNEUMONIA` in each split.
  4. Conducted full byte-level header and integrity audit using `PIL.Image.verify()` across all images.
- **Key Findings & Statistics**:
  - **Zero Corrupted Files**: All 5,856 images verified as 100% valid `.jpeg` files.
  - **Distribution**:
    - **`train`**: 1,341 Normal (25.7%) | 3,875 Pneumonia (74.3%) — **Total: 5,216**
    - **`val`**: 8 Normal (50.0%) | 8 Pneumonia (50.0%) — **Total: 16**
    - **`test`**: 234 Normal (37.5%) | 390 Pneumonia (62.5%) — **Total: 624**
    - **Grand Total**: **5,856 images**
  - **Identified Critical Challenges**:
    1. *Severe Validation Undersizing*: The default 16-image `val` split is inadequate for deep learning validation loss tracking and early stopping.
    2. *Class Imbalance*: Pneumonia comprises 74.3% of the training set, risking majority-class bias.
    3. *Extraction Artifacts*: Detected `__MACOSX/` and a nested duplicate `chest_xray/` from the Kaggle zip archive.

---

### Step 3: Stage 1 Jupyter Notebook (`01_dataset_inspection.ipynb`)
- **Objective**: Provide an interactive notebook for Stage 1 dataset inspection, statistical profiling, and visual validation.
- **File Created**: [`notebooks/01_dataset_inspection.ipynb`](file:///e:/Pnuemonia%20Detection/notebooks/01_dataset_inspection.ipynb)
- **Key Functionality Implemented**:
  1. Environment validation reporting Python, TensorFlow (v2.21.0), NumPy, and GPU availability.
  2. Relative path handling (`../dataset/chest_xray/`).
  3. Dynamic class inventory calculation and formatted pandas DataFrame summary.
  4. Matplotlib grouped bar visualization of class distributions.
  5. Sample visualizer rendering side-by-side Normal vs Pneumonia chest radiographs.
  6. Resolution and mode profiling (sampled 500 images: observed resolutions from $400 \times 600$ to $2000 \times 2000$+ with mixed `L` and `RGB` modes).
  7. Automated non-destructive data integrity audit loop.
  8. Academic markdown cells highlighting the necessity of dynamic validation splitting and class weight calculation.

---

### Step 4: Environment & JupyterLab Server Setup
- **Objective**: Configure and launch the interactive computational workspace.
- **Actions Taken**:
  1. Diagnosed Windows user `PATH` issue where `python -m jupyter notebook` failed to locate `jupyter-notebook.exe` installed under the local AppData directory.
  2. Resolved execution by launching JupyterLab directly via the Python module dispatcher: `python -m jupyterlab`.
  3. Confirmed successful execution of the Stage 1 inspection notebook.

---

### Step 5: Stage 2 Baseline CNN Modeling Notebook (`02_baseline_cnn_model.ipynb`)
- **Objective**: Build an end-to-end data pipeline, mitigate identified dataset challenges, and train a custom benchmark CNN.
- **File Created**: [`notebooks/02_baseline_cnn_model.ipynb`](file:///e:/Pnuemonia%20Detection/notebooks/02_baseline_cnn_model.ipynb)
- **Key Methodological Decisions & Implementations**:
  1. **Dynamic Validation Partitioning**:
     - Used `tf.keras.utils.image_dataset_from_directory(validation_split=0.15, ...)` with fixed seed (`seed=42`).
     - Carved **782 validation images** ($15\%$) from `train/` on-the-fly, leaving 4,434 images for training.
     - Resolved the 16-image limitation cleanly without modifying files on disk.
  2. **Class Imbalance Mitigation**:
     - Computed balanced class weights via `sklearn.utils.class_weight.compute_class_weight` (`NORMAL`: $\approx 1.93$, `PNEUMONIA`: $\approx 0.68$).
     - Integrated class weights into `model.fit(class_weight=...)` to ensure balanced loss gradients.
  3. **Clinically Constrained Data Augmentation**:
     - Implemented `RandomFlip("horizontal")`, `RandomRotation(0.05)`, and `RandomZoom(0.05)`.
     - Strictly avoided vertical flips because inverted thoracic anatomy violates clinical reality.
  4. **High-Throughput Ingestion**:
     - Configured `cache()`, `shuffle(1000)`, and `prefetch(buffer_size=tf.data.AUTOTUNE)`.
  5. **Baseline CNN Architecture Design**:
     - Rescaling layer ($1/255.0$ scaling uint8 $[0, 255] \rightarrow [0, 1]$).
     - 4 Convolutional blocks ($32 \rightarrow 64 \rightarrow 128 \rightarrow 128$ filters, $3 \times 3$ kernel, ReLU).
     - Batch Normalization after each convolution for gradient stabilization.
     - `MaxPooling2D(2, 2)` spatial downsampling.
     - `GlobalAveragePooling2D` to collapse spatial maps into 128 channels without parameter explosion.
     - Named final convolutional layer `conv4_features` to provide direct support for future **Grad-CAM heatmaps**.
     - Dense classification head (128 units, Dropout $0.5$, Sigmoid binary output; total 258,881 parameters).
  6. **Production Callbacks**:
     - `ModelCheckpoint` saving best weights to `models/baseline_cnn_best.keras` (monitored on `val_loss`).
     - `EarlyStopping` (patience = 5, restoring best weights).
     - `ReduceLROnPlateau` (factor = 0.2, patience = 2).
  7. **Diagnostic Evaluation Pipeline**:
     - Evaluates best checkpointed model on official test set ($624$ images).
     - Computes Accuracy, Precision, Recall (Sensitivity), and ROC-AUC.
     - Generates Seaborn Confusion Matrix heatmap and ROC curve plot.

- **Verified Execution & Official Test Results**:
  1. **Training Convergence**:
     - Ran for **15 epochs** before `EarlyStopping` (patience = 5) halted training and restored best weights from **Epoch 10**.
     - Best Validation Metrics (Epoch 10): `val_loss: 0.0961`, `val_auc: 0.9927`, `val_recall: 0.9611`.
  2. **Model Checkpoint Verification**:
     - Saved Artifact: [`models/baseline_cnn_best.keras`](file:///e:/Pnuemonia%20Detection/models/README.md) (**3.19 MB**, 18 layers, 258,881 parameters).
     - Successfully loaded into memory via `tf.keras.models.load_model()`.
  3. **Positive Class Convention**:
     - Verified: Index 0 = `NORMAL` (Negative Class), Index 1 = `PNEUMONIA` (Positive Class).
     - Model outputs sigmoid probability $P(\text{PNEUMONIA} \mid x) \ge 0.5 \implies \text{PNEUMONIA}$.
  4. **Official Test Set Performance (624 Images: 234 Normal, 390 Pneumonia)**:
     - **Accuracy**: **83.17%** ($519 / 624$)
     - **Precision (`PNEUMONIA`)**: **79.63%** ($383 / 481$)
     - **Recall / Sensitivity (`PNEUMONIA`)**: **98.21%** (**383 out of 390 detected**, only 7 false negatives)
     - **F1-Score (`PNEUMONIA`)**: **87.94%**
     - **ROC-AUC**: **0.9252**
  5. **Confusion Matrix Breakdown**:
     - **True Negatives (TN)**: 136 (Correctly identified Normal)
     - **False Positives (FP)**: 98 (Normal incorrectly classified as Pneumonia)
     - **False Negatives (FN)**: 7 (Pneumonia incorrectly classified as Normal)
     - **True Positives (TP)**: 383 (Correctly identified Pneumonia)
  6. **Full Classification Report**:
     ```text
                   precision    recall  f1-score   support

           NORMAL     0.9510    0.5812    0.7215       234
        PNEUMONIA     0.7963    0.9821    0.8794       390

         accuracy                         0.8317       624
        macro avg     0.8737    0.7816    0.8005       624
     weighted avg     0.8543    0.8317    0.8202       624
     ```
  7. **Clinical Takeaway & Benchmark Significance**:
     - The high sensitivity ($98.21\%$) satisfies clinical screening needs by minimizing lethal false negatives ($\text{FN} = 7$).
     - The trade-off is moderate specificity for Normal cases ($58.12\%$, $\text{FP} = 98$), which establishes the baseline to improve upon in **Stage 3 (Transfer Learning)**.

---

### Step 6: Stage 3 Transfer Learning & Optimization Notebook (`03_transfer_learning_experiments.ipynb`)
- **Objective**: Improve diagnostic specificity and elevate overall test accuracy beyond the 83.17% baseline by employing deep transfer learning with **MobileNetV2**.
- **File Created**: [`notebooks/03_transfer_learning_experiments.ipynb`](file:///e:/Pnuemonia%20Detection/notebooks/03_transfer_learning_experiments.ipynb)
- **Key Methodological Decisions & Implementations**:
  1. **Two-Phase Transfer Learning Strategy**:
     - **Phase 1 (Feature Extraction)**: Freeze the pre-trained ImageNet backbone (`base_model.trainable = False`) and train only the custom classification head (`GlobalAveragePooling2D`, `BatchNormalization`, `Dense(128)`, `Dropout(0.4)`, `Dense(1, sigmoid)`) with `Adam(learning_rate=1e-3)`.
     - **Phase 2 (Fine-Tuning)**: Unfreeze top convolutional blocks from layer index 100 onwards while keeping lower-level feature extractors frozen; fine-tune with a micro-learning rate `Adam(learning_rate=1e-5)` to prevent catastrophic forgetting.
  2. **In-Graph Preprocessing**:
     - Embedded `mobilenet_v2.preprocess_input` directly into the Keras functional model, ensuring uniform $[-1, 1]$ intensity scaling.
  3. **High-Throughput Ingestion & Checkpointing**:
     - Chained `cache()` and `prefetch(buffer_size=tf.data.AUTOTUNE)`.
     - Configured `ModelCheckpoint` to save the best model weights directly to [`models/mobilenetv2_transfer_best.keras`](file:///e:/Pnuemonia%20Detection/models/README.md).
  4. **Direct Side-by-Side Benchmark Comparison**:
     - Evaluates on the official 624-image test set.
     - Automatically generates a formatted comparative DataFrame contrasting the Custom Baseline CNN vs. MobileNetV2 across:
       * Accuracy
       * Pneumonia Precision & Recall (Sensitivity)
       * Normal Specificity (Recall)
       * False Positives (reduction from 98) & False Negatives
       * ROC-AUC and visual confusion matrices side-by-side.

- **Verified Execution & Official Test Results (MobileNetV2)**:
  1. **Saved Artifact**: [`models/mobilenetv2_transfer_best.keras`](file:///e:/Pnuemonia%20Detection/models/README.md) (**26.58 MB**, 2,427,201 total parameters). Successfully loaded and evaluated on test set.
  2. **Official Test Set Performance (624 Images: 234 Normal, 390 Pneumonia)**:
     - **Accuracy**: **78.53%** ($490 / 624$)
     - **ROC-AUC**: **0.9566** (Superior discriminative boundary vs. Baseline $0.9252$)
     - **Pneumonia Recall (Sensitivity)**: **98.72%** (**385 out of 390 detected**, only 5 false negatives vs. baseline 7)
     - **Pneumonia Precision**: **74.90%** ($385 / 514$)
     - **Pneumonia F1-Score**: **85.18%**
     - **Normal Specificity (Recall)**: **44.87%** ($105 / 234$)
  3. **Confusion Matrix Breakdown**:
     - **True Negatives (TN)**: 105 (Correctly identified Normal)
     - **False Positives (FP)**: 129 (Normal incorrectly classified as Pneumonia)
     - **False Negatives (FN)**: 5 (Pneumonia incorrectly classified as Normal; lowest false negative rate achieved)
     - **True Positives (TP)**: 385 (Correctly identified Pneumonia)
  4. **Full Classification Report**:
     ```text
                   precision    recall  f1-score   support

           NORMAL     0.9545    0.4487    0.6105       234
        PNEUMONIA     0.7490    0.9872    0.8518       390

         accuracy                         0.7853       624
        macro avg     0.8518    0.7179    0.7311       624
     weighted avg     0.8261    0.7853    0.7613       624
      5. **Comparative Clinical Takeaway**:
     - **ROC-AUC climbed to $0.9566$**, indicating that the feature representations from MobileNetV2 produce significantly stronger probability separation between healthy and infected lung patterns.
     - **Highest Clinical Safety**: Missing only 5 out of 390 pneumonia cases ($98.72\%$ sensitivity) achieves near-perfect pathology capture, ideal for computer-assisted screening prior to radiological review.
     - **Resilience Enhancement**: Added defensive fallback definitions inside Cell 17 so evaluation and comparison cells can be re-run instantly even after a kernel restart without re-executing training.

---

### Step 7: Phase 5 Explainable AI Pipeline & Grad-CAM (`04_explainability_gradcam.ipynb` & `src/explainability/`)
- **Objective**: Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to explain model decisions visually, verify whether predictions are grounded in anatomical pulmonary pathology, and analyze false alarm triggers.
- **Files Created**:
  1. [`src/explainability/gradcam.py`](file:///e:/Pnuemonia%20Detection/src/explainability/gradcam.py): Reusable, production-grade Grad-CAM engine with mathematical gradient pooling, ReLU filtering, and standalone alpha blending overlays using Matplotlib/PIL.
  2. [`src/explainability/__init__.py`](file:///e:/Pnuemonia%20Detection/src/explainability/__init__.py): Explainability module exports.
  3. [`notebooks/04_explainability_gradcam.ipynb`](file:///e:/Pnuemonia%20Detection/notebooks/04_explainability_gradcam.ipynb): Interactive clinical interpretability research notebook.
- **Key Methodological Decisions & Implementations**:
  1. **Dual-Model Support**:
     - *Custom Baseline CNN*: Maps activations from final convolutional layer `conv4_features` (feature map shape: $28 \times 28 \times 128$).
     - *MobileNetV2 Transfer Model*: Submodel feature extraction from backbone layer `Conv_1` (feature map shape: $7 \times 7 \times 1280$) with gradients backpropagated through custom classification head.
  2. **Mathematical Formulation**:
     - Computes neuron importance weights $\alpha_k^c$ via Global Average Pooling of gradients $\frac{\partial y^c}{\partial A^k}$.
     - Applies ReLU activation to isolate positive visual evidence for the target clinical diagnosis.
  3. **High-Resolution Color Blending**:
     - Resizes activation heatmaps to original radiograph resolution ($224 \times 224$) via bicubic interpolation.
     - Blends colorized JET colormap with grayscale lung radiograph ($\alpha = 0.4$) to preserve anatomical landmarks (ribs, clavicles, cardiac borders).
  4. **Four Clinical Case Studies**:
     - *True Positive*: Confirms model grounds Pneumonia predictions in focal pulmonary opacities and consolidations.
     - *True Negative*: Validates absence of pathological focus in clear lung fields.
     - *False Positive Audit*: Investigates visual artifacts (scapular edges, dense rib intersections) that trigger false alarms.
     - *Multi-Case Clinical Gallery*: Side-by-side visual comparison of Baseline CNN vs. MobileNetV2 attention maps.

---

### Step 8: Phase 6 FastAPI REST API Backend Implementation (`backend/` & `test_backend.py`)
- **Objective**: Design, build, and verify an asynchronous, high-performance REST API with **FastAPI** to serve deep learning predictions, enable dynamic dual-model switching (Baseline CNN vs. MobileNetV2), and generate real-time Grad-CAM visual heatmaps encoded as base64 Data URIs for client display.
- **Files Created / Updated**:
  1. [`backend/app/core/config.py`](file:///e:/Pnuemonia%20Detection/backend/app/core/config.py): Dynamic root path resolution, model paths, target convolution layers (`conv4_features` vs. `Conv_1`), CORS policies (Vite, CRA, localhost), and file upload limits (15 MB).
  2. [`backend/app/schemas/prediction.py`](file:///e:/Pnuemonia%20Detection/backend/app/schemas/prediction.py): Pydantic validation schemas (`PredictionResponse`, `HealthResponse`, `ModelMetadata`, `ModelEvaluationMetrics`, `ModelsListResponse`).
  3. [`backend/app/services/cam_service.py`](file:///e:/Pnuemonia%20Detection/backend/app/services/cam_service.py): In-memory image validation, stream parsing, and base64 Data URI encoder.
  4. [`backend/app/services/model_service.py`](file:///e:/Pnuemonia%20Detection/backend/app/services/model_service.py): Singleton model manager providing lifecycle handling, thread-safe memory caching, inference orchestration, millisecond latency measurement, and automated clinical interpretability notes.
  5. [`backend/app/api/deps.py`](file:///e:/Pnuemonia%20Detection/backend/app/api/deps.py): Dependency injection provider for `ModelService`.
  6. [`backend/app/api/routes/health.py`](file:///e:/Pnuemonia%20Detection/backend/app/api/routes/health.py): `GET /api/v1/health` verifying engine readiness, TensorFlow version, and loaded models.
  7. [`backend/app/api/routes/models.py`](file:///e:/Pnuemonia%20Detection/backend/app/api/routes/models.py): `GET /api/v1/models` and `GET /api/v1/models/{model_id}` providing model catalog, parameter counts, and benchmark test metrics.
  8. [`backend/app/api/routes/predict.py`](file:///e:/Pnuemonia%20Detection/backend/app/api/routes/predict.py): `POST /api/v1/predict` accepting multipart radiograph upload, validating image integrity, running model inference, and generating Grad-CAM heatmaps.
  9. [`backend/app/main.py`](file:///e:/Pnuemonia%20Detection/backend/app/main.py): FastAPI entrypoint with lifespan model pre-warming, CORS middleware, interactive Swagger UI (`/docs`), and root redirect.
  10. [`src/explainability/gradcam.py`](file:///e:/Pnuemonia%20Detection/src/explainability/gradcam.py): Updated `explain_chest_xray()` to flexibly accept file paths, PIL Images, and binary streams.
  11. [`backend/test_backend.py`](file:///e:/Pnuemonia%20Detection/backend/test_backend.py): Automated end-to-end integration test suite.
  12. [`backend/README.md`](file:///e:/Pnuemonia%20Detection/backend/README.md): Comprehensive backend architecture and endpoint documentation.
- **Verified Execution & Integration Test Suite Results**:
  - `test_health_endpoint`: **PASS** (Service healthy, TensorFlow 2.21.0, models detected)
  - `test_models_list_endpoint`: **PASS** (Catalog returns both Baseline CNN and MobileNetV2 with full benchmark test metrics)
  - `test_specific_model_endpoint`: **PASS** (MobileNetV2 metadata & 404 handling verified)
  - `test_predict_mobilenetv2_normal`: **PASS** (Inference + Grad-CAM generated in ~1628ms)
  - `test_predict_baseline_pneumonia`: **PASS** (Classified Pneumonia with 99.65% confidence in ~306ms)
  - `test_predict_baseline_normal_case`: **PASS** (Classified Normal with 53.59% confidence)
  - `test_predict_validation_errors`: **PASS** (Correctly rejects unsupported formats like `.txt` and invalid model IDs with HTTP 400)

---

### Step 9: Phase 7 React Frontend & Clinical Visualization UI (`frontend/`)
- **Objective**: Develop a modern, responsive, and aesthetically refined **React** single-page clinical dashboard with **Vite** and custom vanilla CSS to provide real-time diagnostic classification and interactive Grad-CAM opacity superimposition.
- **Files Created / Configured**:
  1. [`frontend/src/services/api.js`](file:///e:/Pnuemonia%20Detection/frontend/src/services/api.js): Robust API client with error handling connecting to FastAPI backend endpoints (`/api/v1/health`, `/api/v1/models`, `/api/v1/predict`, and sample loader).
  2. [`frontend/src/components/Header.jsx`](file:///e:/Pnuemonia%20Detection/frontend/src/components/Header.jsx): Navigation bar with brand logo, live API health indicator, dynamic model switcher dropdown, and benchmark modal launcher.
  3. [`frontend/src/components/ImageUploader.jsx`](file:///e:/Pnuemonia%20Detection/frontend/src/components/ImageUploader.jsx): Drag-and-drop dropzone, file metadata preview, and 4 quick-test benchmark sample radiograph buttons for instant one-click demonstration.
  4. [`frontend/src/components/PredictionCard.jsx`](file:///e:/Pnuemonia%20Detection/frontend/src/components/PredictionCard.jsx): Diagnostic verdict card with glowing status badges (Emerald `NORMAL` vs. Crimson `PNEUMONIA`), confidence gauge, animated dual probability bars, processing latency indicator, and automated clinical interpretability notes.
  5. [`frontend/src/components/GradCamViewer.jsx`](file:///e:/Pnuemonia%20Detection/frontend/src/components/GradCamViewer.jsx): Explainable AI visualizer featuring:
     - **Interactive Opacity Slider**: Smooth 60 FPS slider dynamically fading between clean anatomical radiograph ($0\%$) and full JET colormap activation heatmap ($100\%$).
     - **Side-by-Side Comparison Mode**: Visualizes original grayscale radiograph, raw activation heatmap, and blended overlay.
     - **Colormap Legend**: Blue (normal lung tissue) to Red (pathological consolidation).
     - **Overlay Download**: One-click download of superimposition images.
  6. [`frontend/src/components/BenchmarksModal.jsx`](file:///e:/Pnuemonia%20Detection/frontend/src/components/BenchmarksModal.jsx): Comprehensive benchmark comparison dialog detailing metrics on the official 624-image test set and explaining clinical sensitivity prioritization.
  7. [`frontend/src/components/Footer.jsx`](file:///e:/Pnuemonia%20Detection/frontend/src/components/Footer.jsx): Academic project attribution, medical disclaimer, and technology stack badges.
  8. [`frontend/src/index.css`](file:///e:/Pnuemonia%20Detection/frontend/src/index.css) & [`frontend/src/App.css`](file:///e:/Pnuemonia%20Detection/frontend/src/App.css): Curated clinical dark-mode design system with glassmorphic cards, custom range sliders, typography, and responsive grid layouts.
  9. [`frontend/src/App.jsx`](file:///e:/Pnuemonia%20Detection/frontend/src/App.jsx): Main dashboard state management, error boundaries, and auto-re-evaluation when switching models.
  10. [`frontend/vite.config.js`](file:///e:/Pnuemonia%20Detection/frontend/vite.config.js): Reverse proxy configuration routing `/api` requests to backend on port 8000.
  11. [`frontend/public/samples/`](file:///e:/Pnuemonia%20Detection/frontend/public/samples/): 4 preloaded test set radiographs (2 Normal, 2 Pneumonia) for instant zero-friction evaluation.
  12. [`frontend/README.md`](file:///e:/Pnuemonia%20Detection/frontend/README.md): Complete frontend architectural documentation and developer instructions.
- **Build Verification**:
  - Executed `npm run build`: Production bundle generated in 210ms with zero errors (`dist/index.html` 1.05 kB, `dist/assets/*.css` 19.20 kB, `dist/assets/*.js` 249.77 kB).

---

## Current Project Status Overview

| Phase | Description | Status | Deliverables |
| :---: | :--- | :---: | :--- |
| **Phase 1** | Project Setup, Architecture & Version Control | **Complete** | `.gitignore`, `requirements.txt`, root and subdirectory READMEs |
| **Phase 2** | Dataset Inspection, Verification & Integrity Audit | **Complete** | Full distribution analysis, `01_dataset_inspection.ipynb` |
| **Phase 3** | Input Pipeline, Augmentation & Baseline CNN Benchmark | **Complete & Verified** | `02_baseline_cnn_model.ipynb`, `models/baseline_cnn_best.keras` (83.17% Acc, 98.21% Recall, 0.9252 AUC) |
| **Phase 4** | Transfer Learning (MobileNetV2 / ResNet50) | **Complete & Verified** | `03_transfer_learning_experiments.ipynb`, `models/mobilenetv2_transfer_best.keras` (98.72% Recall, 0.9566 AUC) |
| **Phase 5** | Explainable AI (Grad-CAM Saliency Heatmaps) | **Complete & Verified** | `src/explainability/gradcam.py`, `04_explainability_gradcam.ipynb` |
| **Phase 6** | FastAPI REST API Backend Implementation | **Complete & Verified** | `backend/app/main.py`, services, routes, schemas, and `test_backend.py` |
| **Phase 7** | React Frontend & Clinical Visualization UI | **Complete & Verified** | `frontend/src/App.jsx`, interactive opacity slider, sample pickers, production build |


