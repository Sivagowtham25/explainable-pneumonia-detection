import React from 'react';
import { Activity, Cpu, BarChart2 } from 'lucide-react';

export default function Header({
  health,
  selectedModel,
  onSelectModel,
  onOpenBenchmarks,
}) {
  return (
    <header className="header-container glass-panel">
      <div className="header-brand">
        <div className="logo-badge">
          <Activity className="brand-icon" size={26} />
        </div>
        <div>
          <h1 className="brand-title">
            Pneumo<span>Scan</span> AI
          </h1>
          <p className="brand-subtitle">
            Explainable Deep Learning Diagnosis & Real-time Grad-CAM
          </p>
        </div>
      </div>

      <div className="header-actions">
        {/* Model Selector */}
        <div className="model-selector-wrapper">
          <label htmlFor="model-select" className="selector-label">
            <Cpu size={14} /> Active Model:
          </label>
          <select
            id="model-select"
            className="model-select-dropdown"
            value={selectedModel}
            onChange={(e) => onSelectModel(e.target.value)}
          >
            <option value="mobilenetv2">
              MobileNetV2 Transfer (AUC: 0.9566 | Rec: 98.7%)
            </option>
            <option value="baseline">
              Custom Baseline CNN (Acc: 83.17% | Rec: 98.2%)
            </option>
          </select>
        </div>

        {/* Benchmark Comparison Button */}
        <button
          type="button"
          className="benchmark-btn"
          onClick={onOpenBenchmarks}
          title="Compare Baseline CNN vs MobileNetV2 Benchmark Metrics"
        >
          <BarChart2 size={16} />
          <span>Benchmarks</span>
        </button>

        {/* Health Status Indicator */}
        <div className="health-badge-container">
          {health?.system_ready ? (
            <div className="health-badge ready" title="FastAPI and TensorFlow ready">
              <span className="status-dot ready pulse-animation"></span>
              <span>API Ready</span>
            </div>
          ) : (
            <div className="health-badge degraded" title="Connecting to backend API...">
              <span className="status-dot degraded"></span>
              <span>Connecting</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
