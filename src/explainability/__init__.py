"""
Explainability package providing Gradient-weighted Class Activation Mapping (Grad-CAM)
utilities for chest radiograph interpretability.
"""

from .gradcam import make_gradcam_heatmap, generate_gradcam_overlay, explain_chest_xray

__all__ = ["make_gradcam_heatmap", "generate_gradcam_overlay", "explain_chest_xray"]
