# Source Code Directory (`src/`)

This directory will house reusable, modular Python source modules for the core machine learning and explainability pipeline.

## Planned Modular Structure
```text
src/
├── __init__.py
├── config.py                 # Hyperparameters, image sizes, paths, and training configs
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py     # Image loading, validation, and batch generators
│   └── augmentation.py       # Data augmentation and preprocessing transforms
├── models/
│   ├── __init__.py
│   ├── cnn_baseline.py       # Custom CNN architecture definition
│   └── transfer_models.py    # Pretrained backbones (e.g., MobileNetV2, ResNet50)
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Training and validation routines
│   └── callbacks.py          # Early stopping, model checkpoints, LR schedulers
├── explainability/
│   ├── __init__.py
│   └── gradcam.py            # Gradient-weighted Class Activation Mapping & heatmap overlays
└── utils/
    ├── __init__.py
    ├── metrics.py            # Precision, Recall, F1, Confusion Matrix, ROC-AUC calculations
    └── visualization.py      # Matplotlib plotting helpers for training curves and sample plots
```

*Note: Code implementation will take place in subsequent project stages.*
