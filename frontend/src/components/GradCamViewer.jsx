import React, { useState } from 'react';
import { Layers, Sliders, Columns, Download, Info } from 'lucide-react';

export default function GradCamViewer({ result }) {
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' or 'side-by-side'
  const [interactiveAlpha, setInteractiveAlpha] = useState(0.4);

  if (!result || !result.gradcam_generated) return null;

  const originalImg = result.original_image_base64;
  const coloredHeatmap = result.colored_heatmap_base64;
  const blendedOverlay = result.gradcam_overlay_base64;

  const downloadImage = (dataUrl, filename) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="gradcam-card glass-panel">
      <div className="card-header">
        <div className="title-group">
          <h2 className="card-title">
            <Layers size={20} className="card-icon" />
            <span>Grad-CAM Visual Saliency Heatmap</span>
          </h2>
          <span className="subtitle-tag">
            Anatomical Feature Localization ({result.target_conv_layer})
          </span>
        </div>

        {/* View Mode Toggle */}
        <div className="view-mode-toggle">
          <button
            type="button"
            className={`toggle-btn ${viewMode === 'overlay' ? 'active' : ''}`}
            onClick={() => setViewMode('overlay')}
          >
            <Sliders size={14} />
            <span>Opacity Slider</span>
          </button>
          <button
            type="button"
            className={`toggle-btn ${viewMode === 'side-by-side' ? 'active' : ''}`}
            onClick={() => setViewMode('side-by-side')}
          >
            <Columns size={14} />
            <span>Side-by-Side</span>
          </button>
        </div>
      </div>

      {/* Mode 1: Interactive Opacity Slider */}
      {viewMode === 'overlay' && (
        <div className="overlay-view-wrapper">
          <div className="interactive-stage">
            <div className="image-stack-container">
              {/* Base Layer: Original Grayscale Radiograph */}
              <img
                src={originalImg}
                alt="Original Radiograph"
                className="stack-base-img"
              />
              {/* Heatmap Layer: Adjusted via interactive slider opacity */}
              <img
                src={coloredHeatmap}
                alt="Colorized Grad-CAM Heatmap"
                className="stack-overlay-img"
                style={{
                  opacity: interactiveAlpha,
                  mixBlendMode: 'normal',
                }}
              />
            </div>

            <div className="stage-controls">
              <div className="slider-container">
                <div className="slider-header">
                  <label htmlFor="alpha-slider" className="slider-label">
                    <Sliders size={14} /> Heatmap Blend Opacity:
                  </label>
                  <span className="slider-value font-mono">
                    {Math.round(interactiveAlpha * 100)}%
                  </span>
                </div>
                <input
                  id="alpha-slider"
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.02"
                  value={interactiveAlpha}
                  onChange={(e) => setInteractiveAlpha(parseFloat(e.target.value))}
                />
                <div className="slider-scale-labels">
                  <span>0% (Clean Radiograph)</span>
                  <span>50% (Balanced)</span>
                  <span>100% (Pure Activation Map)</span>
                </div>
              </div>

              <button
                type="button"
                className="download-overlay-btn"
                onClick={() => downloadImage(blendedOverlay, `gradcam_${result.filename}`)}
                title="Download Blended Grad-CAM image"
              >
                <Download size={15} />
                <span>Save Grad-CAM Overlay</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Mode 2: Side-by-Side Comparison */}
      {viewMode === 'side-by-side' && (
        <div className="side-by-side-grid">
          {/* Card 1: Original */}
          <div className="side-card">
            <div className="side-card-header">
              <span>1. Original AP Radiograph</span>
              <span className="badge-small">Grayscale</span>
            </div>
            <div className="side-img-wrapper">
              <img src={originalImg} alt="Original Radiograph" className="side-img" />
            </div>
          </div>

          {/* Card 2: Raw Heatmap */}
          <div className="side-card">
            <div className="side-card-header">
              <span>2. Grad-CAM Activation</span>
              <span className="badge-small">JET Colormap</span>
            </div>
            <div className="side-img-wrapper">
              <img src={coloredHeatmap} alt="Grad-CAM Heatmap" className="side-img" />
            </div>
          </div>

          {/* Card 3: Blended Overlay */}
          <div className="side-card active-card">
            <div className="side-card-header">
              <span>3. Clinical Superimposition</span>
              <span className="badge-small highlighted">&alpha; = 0.40</span>
            </div>
            <div className="side-img-wrapper">
              <img src={blendedOverlay} alt="Blended Overlay" className="side-img" />
            </div>
          </div>
        </div>
      )}

      {/* Diagnostic Heatmap Color Legend */}
      <div className="legend-container">
        <div className="legend-title">
          <Info size={14} />
          <span>Grad-CAM Activation Intensity Legend:</span>
        </div>
        <div className="legend-gradient-wrapper">
          <div className="legend-bar"></div>
          <div className="legend-labels">
            <span>Low Activation (Normal tissue)</span>
            <span>Moderate Attention</span>
            <span>High Activation (Pathological Focus)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
