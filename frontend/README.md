# Frontend Application (`frontend/`)

This directory contains the **PneumoScan AI** interactive clinical dashboard built with **React 19** and **Vite**, featuring custom vanilla CSS design system and real-time **Grad-CAM visual explainability**.

---

## 1. Key Features

- **Clinical Image Ingestion**:
  - Drag-and-drop or file browser supporting chest radiographs (`JPEG`, `PNG`, `WebP`).
  - Image preview with file metadata (filename, size).
  - Quick-test with benchmark test radiographs (2 Normal, 2 Pneumonia) for instant demonstration.
- **Model Switching**:
  - Toggle between **MobileNetV2 Fine-Tuned Transfer Model** (0.9566 ROC-AUC, 98.72% Sensitivity) and **Custom Baseline CNN** (83.17% Accuracy, 98.21% Sensitivity).
  - Automatically re-evaluates the active radiograph when the model selection changes.
- **Diagnostic Verdict Card**:
  - Semantic glowing banners: Emerald for `NORMAL (CLEAR LUNG FIELDS)` and Crimson for `PNEUMONIA DETECTED`.
  - Confidence percentage gauge and animated dual probability bars.
  - End-to-end inference and Grad-CAM generation latency indicator (in milliseconds).
  - Automated diagnostic notes summarizing focal pulmonary findings.
- **Explainable AI (Grad-CAM) Visualizer**:
  - **Interactive Opacity Slider**: Smooth 60 FPS slider allowing clinicians to dynamically fade between the raw grayscale anatomical radiograph (0%) and the full JET colormap activation heatmap (100%).
  - **Side-by-Side Comparison Mode**: Visualizes Original Radiograph, Raw Grad-CAM Activation Heatmap, and Superimposed Blended Overlay.
  - Colormap activation intensity legend (Blue = normal lung tissue $\rightarrow$ Red = critical consolidation focus).
  - Image download feature to save blended Grad-CAM overlays.
- **Model Benchmark Modal**:
  - Detailed side-by-side performance comparison on the official 624-image test set.
  - Clinical explanation regarding why prioritizing Recall/Sensitivity (>98%) is paramount to minimize fatal False Negatives.

---

## 2. Component Hierarchy

```text
frontend/src/
├── services/
│   └── api.js                   # Client requests to /api/v1 (health, models, predict)
├── components/
│   ├── Header.jsx               # Brand, health status badge, model selector, benchmark trigger
│   ├── ImageUploader.jsx        # Drag-and-drop zone, file preview, benchmark sample quick-pickers
│   ├── PredictionCard.jsx       # Diagnostic verdict, confidence gauge, probability bars, clinical notes
│   ├── GradCamViewer.jsx        # Dual-mode Grad-CAM visualizer with interactive opacity slider
│   ├── BenchmarksModal.jsx      # Benchmark metrics modal (Accuracy, ROC-AUC, Sensitivity, Specificity)
│   └── Footer.jsx               # Academic research disclaimer and tech stack attribution
├── App.css                      # Component styling and responsive layout
├── App.jsx                      # Main dashboard layout and reactive state management
├── index.css                    # Design system, CSS tokens, glassmorphism, animations
└── main.jsx                     # Vite React entrypoint
```

---

## 3. Running the Development Server

1. Ensure the FastAPI backend is running:
   ```powershell
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

2. Start the Vite React development server:
   ```powershell
   cd frontend
   npm run dev
   ```

3. Open [http://localhost:5173](http://localhost:5173) in your browser. All `/api` requests are automatically proxied to `http://127.0.0.1:8000`.

---

## 4. Production Build

To build the optimized production bundle:

```powershell
npm run build
```
Output bundle is generated into `frontend/dist/`.
