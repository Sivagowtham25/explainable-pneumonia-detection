# Explainable Pneumonia Classification from Chest X-Ray Images Using Deep Learning

An MCA Semester Project focusing on computer-assisted medical diagnosis through deep learning and Explainable Artificial Intelligence (XAI).

---

## 1. Project Overview

Pneumonia is an acute respiratory infection affecting the lungs, primarily diagnosed through chest radiography (X-rays). While chest X-rays are widely available, radiological interpretation requires specialized expertise, can be subjective, and faces significant bottlenecks in resource-constrained clinical settings.

Recent advances in deep learning and Convolutional Neural Networks (CNNs) have enabled high-accuracy automated image classification. However, standard deep learning models operate as "black boxes," providing predictions without transparent clinical rationale. In medical diagnostics, unexplained predictions hinder clinician trust and present safety concerns.

This project addresses these challenges by developing an **end-to-end Explainable AI (XAI) system**:
1. Classifying chest X-ray images into **NORMAL** and **PNEUMONIA** categories using deep neural networks.
2. Generating visual explanation heatmaps via **Grad-CAM (Gradient-weighted Class Activation Mapping)** to pinpoint lung regions, opacities, and consolidations influencing each diagnosis.
3. Providing an accessible web-based interface (FastAPI + React) for healthcare practitioners and researchers to upload scans, review model confidence, and inspect saliency overlays interactively.

---

## 2. Project Objectives

- **Automated Binary Classification**: Develop and validate a deep learning model to accurately distinguish between normal and pneumonia-affected pediatric chest radiographs.
- **Model Interpretability (Explainable AI)**: Integrate Grad-CAM to visualize and validate whether the neural network grounds its predictions in medically relevant thoracic features (e.g., infiltrates and consolidations) rather than spurious background artifacts.
- **Transfer Learning & Performance Benchmarking**: Compare a baseline custom CNN against pre-trained architectures (e.g., MobileNetV2, ResNet50, EfficientNet) to achieve optimal sensitivity (recall), specificity, and ROC-AUC scores.
- **Decoupled REST API Service**: Implement a high-performance backend with **FastAPI** to handle image validation, batching, inference, and dynamic heatmap generation.
- **Interactive Web Interface**: Build a responsive **React** frontend supporting drag-and-drop X-ray uploads, prediction displays with probability meters, and side-by-side or blended heatmap opacity inspection.

---

## 3. Dataset Information

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al.).
- **Modality**: Anterior-Posterior (AP) chest radiographs from pediatric patients (1–5 years of age).
- **Target Classes**:
  - `NORMAL`: Healthy lungs with clear lung fields and visible cardiac silhouette.
  - `PNEUMONIA`: Lungs presenting focal or diffuse consolidation, interstitial opacities, or infiltrates (both bacterial and viral etiologies).
- **Directory Organization**:
  ```text
  dataset/chest_xray/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
  ```
- **Data Exclusion Policy**: Raw and processed medical image files are strictly excluded from Git tracking via `.gitignore`.

---

## 4. Planned Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming language for modeling, processing, and backend |
| **Deep Learning Framework** | TensorFlow & Keras | Model architecture development, transfer learning, and training |
| **Data Processing & ML** | NumPy, pandas, scikit-learn | Numerical operations, tabular analysis, and evaluation metrics |
| **Image Processing** | Pillow (PIL), OpenCV | Image resizing, color space conversions, and heatmap blending |
| **Plotting & Visualization** | Matplotlib, Seaborn | Training curve plots, confusion matrices, and distribution graphs |
| **Explainable AI (XAI)** | Grad-CAM (`tf-keras-vis` / custom) | Visual explanation via gradient-weighted feature map localization |
| **Backend API** | FastAPI, Uvicorn, Pydantic | Asynchronous REST API, request validation, and prediction endpoints |
| **Frontend UI** | React, JavaScript, CSS3 | Interactive web dashboard, image uploader, and heatmap viewer |
| **Experimentation** | Jupyter Notebooks | Iterative exploration, hyperparameter tuning, and prototyping |

---

## 5. Project Directory Structure

```text
PneumoniaDetection/
├── dataset/                    # Dataset directory (excluded from git tracking)
│   ├── chest_xray/             # Kaggle chest X-ray images (train/val/test)
│   └── README.md               # Dataset specifications and guidelines
├── notebooks/                  # Jupyter notebooks for iterative experimentation
│   ├── .gitkeep
│   └── README.md
├── models/                     # Saved model artifacts (*.keras, *.h5 - git ignored)
│   ├── .gitignore              # Local rule ignoring large weights
│   ├── .gitkeep
│   └── README.md
├── src/                        # Modular Python source packages
│   ├── .gitkeep
│   └── README.md               # Architecture specification for future modules
├── backend/                    # FastAPI web application service
│   ├── .gitkeep
│   └── README.md               # API architecture and endpoint blueprint
├── frontend/                   # React web application user interface
│   ├── .gitkeep
│   └── README.md               # Frontend component blueprint and specifications
├── PROJECT_STEPS_SUMMARY.md   # Chronological development steps log (Markdown)
├── PROJECT_STEPS_SUMMARY.txt  # Plain-text step-by-step progress summary
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git exclusions for datasets, weights, and caches
└── README.md                   # Project documentation and roadmap
```

---

## 6. Project Execution Log & Current Progress

For detailed step-by-step logs, data distribution tables, and architectural rationale, refer to:
- **[PROJECT_STEPS_SUMMARY.md](file:///e:/Pnuemonia%20Detection/PROJECT_STEPS_SUMMARY.md)** (Full Markdown log)
- **[PROJECT_STEPS_SUMMARY.txt](file:///e:/Pnuemonia%20Detection/PROJECT_STEPS_SUMMARY.txt)** (Plain-text summary)

### Summary of Completed Milestones:
- **Milestone 1 (Project Architecture)**: Repository directory tree established; version control rules configured in root and `models/` `.gitignore` to exclude datasets and large binaries; dependencies defined.
- **Milestone 2 (Dataset Inspection & Audit)**: Validated all 5,856 images with zero corruptions (train: 5,216, val: 16, test: 624). Identified critical class imbalance (~74% Pneumonia) and severe undersizing of the default 16-image validation partition.
- **Milestone 3 (Exploratory Data Analysis)**: Created [`notebooks/01_dataset_inspection.ipynb`](file:///e:/Pnuemonia%20Detection/notebooks/01_dataset_inspection.ipynb) for relative path resolution, inventory statistics, class balance visualization, and resolution profiling.
- **Milestone 5 (Transfer Learning Framework & Evaluation)**: Created [`notebooks/03_transfer_learning_experiments.ipynb`](file:///e:/Pnuemonia%20Detection/notebooks/03_transfer_learning_experiments.ipynb) implementing a two-phase transfer learning pipeline with pre-trained MobileNetV2 (Phase 1 Feature Extraction with frozen backbone; Phase 2 Fine-Tuning unfreezing top layers with Adam $10^{-5}$), in-graph preprocessing, and checkpointing to `models/mobilenetv2_transfer_best.keras`. Achieved **0.9566 ROC-AUC** and **98.72% Pneumonia Recall** (only 5 false negatives across 390 cases).
- **Milestone 6 (Explainable AI & Grad-CAM Pipeline)**: Created reusable explainability engine [`src/explainability/gradcam.py`](file:///e:/Pnuemonia%20Detection/src/explainability/gradcam.py) and research notebook [`notebooks/04_explainability_gradcam.ipynb`](file:///e:/Pnuemonia%20Detection/notebooks/04_explainability_gradcam.ipynb). Implemented mathematical Grad-CAM with global average pooled gradients, ReLU feature filtering, and high-resolution JET colormap overlay blending ($\alpha = 0.4$) supporting both Custom Baseline CNN (`conv4_features`) and MobileNetV2 (`Conv_1`), including 4 detailed clinical case studies.
- **Milestone 7 (FastAPI REST API Backend Implementation)**: Developed asynchronous, modular web service in [`backend/`](file:///e:/Pnuemonia%20Detection/backend/README.md) featuring dual-model serving (`baseline` and `mobilenetv2`), health and engine monitoring (`/api/v1/health`), model catalog with benchmark metrics (`/api/v1/models`), image upload prediction with real-time Grad-CAM heatmap overlays encoded in base64 (`/api/v1/predict`), CORS middleware, Swagger UI at `/docs`, and automated integration tests ([`backend/test_backend.py`](file:///e:/Pnuemonia%20Detection/backend/test_backend.py)) passing 100%.
- **Milestone 8 (React Frontend & Clinical Visualization UI)**: Built state-of-the-art interactive clinical dashboard in [`frontend/`](file:///e:/Pnuemonia%20Detection/frontend/README.md) with React 19, Vite, and custom CSS design system. Features drag-and-drop radiograph upload, 4 bundled benchmark quick-test samples, dynamic dual-model switching, diagnostic verdicts with confidence meters, and an interactive 60 FPS Grad-CAM opacity superimposition slider allowing clinicians to dynamically fade between raw anatomical lung tissue and pathological activation heatmaps. Build verified with zero errors.

---

## 7. Planned Development Stages

```mermaid
flowchart LR
    A[Phase 1: Setup & Data Prep] --> B[Phase 2: Baseline Modeling]
    B --> C[Phase 3: Transfer Learning]
    C --> D[Phase 4: Explainability Grad-CAM]
    D --> E[Phase 5: FastAPI Backend]
    E --> F[Phase 6: React Frontend]
    F --> G[Phase 7: Testing & Documentation]
```

### Stage 1: Setup & Data Preprocessing
- Verify dataset integrity, directory structure, and balance across training, validation, and testing splits.
- Build image preprocessing pipeline: resizing to fixed dimensions (e.g., $224 \times 224$), channel normalization ($[0, 1]$ or $[-1, 1]$), and robust data augmentation (zoom, slight rotation, shear) to prevent overfitting.

### Stage 2: Exploratory Data Analysis & Baseline CNN
- Conduct exploratory data analysis (EDA) in Jupyter notebooks.
- Design, compile, and train a baseline Convolutional Neural Network (Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense).
- Establish baseline benchmark metrics (Accuracy, Loss, Precision, Recall, F1-Score).

### Stage 3: Transfer Learning & Hyperparameter Optimization
- Implement transfer learning using established deep architectures (e.g., MobileNetV2, ResNet50, EfficientNetB0).
- Compare feature extraction (frozen backbones) vs. fine-tuning top convolutional layers.
- Evaluate model performance with emphasis on **Recall/Sensitivity** (minimizing false negatives in clinical diagnosis) and **ROC-AUC**.

### Stage 4: Explainability Pipeline (Grad-CAM)
- Implement Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional layer.
- Generate coarse 2D heatmaps highlighting discriminative visual features.
- Blend heatmaps with original grayscale X-rays using OpenCV colormaps (e.g., `COLORMAP_JET`).
- Validate model visual focus against anatomical lung field locations.

### Stage 5: Backend API Development (FastAPI)
- Build modular REST API with FastAPI.
- Implement endpoints for health checks, single-image inference, and Grad-CAM generation.
- Set up input sanitization, error handling, and CORS middleware for frontend communication.

### Stage 6: Frontend Development (React)
- Create a modern, responsive React interface.
- Implement intuitive file drag-and-drop, image preview, prediction indicators, and confidence meters.
- Build an interactive heatmap inspection component with adjustable transparency sliders to allow clinicians to inspect the underlying lung structure.

### Stage 7: Integration, Testing & Final Documentation
- Perform end-to-end integration testing between React frontend and FastAPI backend.
- Measure inference latency and optimize model serving.
- Prepare comprehensive academic project report, thesis documentation, and demonstration presentation.

---

## 7. Medical & Academic Disclaimer

> **Important Notice**: This project is developed exclusively for educational, academic, and research purposes as part of an MCA curriculum. The deep learning models and predictions generated by this software are **not** certified medical devices and must **not** be used as a primary diagnostic tool for patient management or clinical decision-making. All clinical decisions must be made by qualified healthcare professionals.
