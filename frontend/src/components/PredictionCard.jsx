import React from 'react';
import { AlertTriangle, CheckCircle2, Clock, HelpCircle, Activity } from 'lucide-react';

export default function PredictionCard({ result }) {
  if (!result) return null;

  const isPneumonia = result.is_pneumonia;
  const pneuProb = result.prob_pneumonia;
  const normProb = result.prob_normal;

  return (
    <div className={`prediction-card glass-panel ${isPneumonia ? 'border-pneumonia' : 'border-normal'}`}>
      <div className="card-header">
        <h2 className="card-title">
          <Activity size={20} className="card-icon" />
          <span>Diagnostic Classification</span>
        </h2>
        <div className="latency-badge" title="Total processing time including Grad-CAM generation">
          <Clock size={13} />
          <span>{result.inference_time_ms} ms</span>
        </div>
      </div>

      {/* Main Diagnostic Badge */}
      <div className={`verdict-banner ${isPneumonia ? 'verdict-pneumonia' : 'verdict-normal'}`}>
        <div className="verdict-icon-wrapper">
          {isPneumonia ? <AlertTriangle size={32} /> : <CheckCircle2 size={32} />}
        </div>
        <div className="verdict-text-group">
          <span className="verdict-subtitle">Model Diagnostic Verdict</span>
          <h3 className="verdict-title">
            {isPneumonia ? 'PNEUMONIA DETECTED' : 'NORMAL (CLEAR LUNG FIELDS)'}
          </h3>
        </div>
        <div className="verdict-confidence-badge">
          <span className="confidence-label">Confidence</span>
          <span className="confidence-value">{result.confidence}%</span>
        </div>
      </div>

      {/* Probability Distribution Meters */}
      <div className="probabilities-container">
        {/* Pneumonia Meter */}
        <div className="meter-group">
          <div className="meter-labels">
            <span className="meter-name pneumonia-text">
              <span className="status-dot dot-red"></span> Pneumonia Likelihood
            </span>
            <span className="meter-val font-mono">{pneuProb}%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill progress-pneumonia"
              style={{ width: `${pneuProb}%` }}
            ></div>
          </div>
        </div>

        {/* Normal Meter */}
        <div className="meter-group">
          <div className="meter-labels">
            <span className="meter-name normal-text">
              <span className="status-dot dot-green"></span> Normal Likelihood
            </span>
            <span className="meter-val font-mono">{normProb}%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill progress-normal"
              style={{ width: `${normProb}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Clinical Notes & Model Parameters */}
      <div className="notes-box">
        <div className="notes-header">
          <HelpCircle size={15} className="notes-icon" />
          <span>Automated Interpretability Findings</span>
        </div>
        <p className="notes-text">{result.clinical_notes}</p>
      </div>

      {/* Diagnostic Specs Footer */}
      <div className="specs-footer">
        <div className="spec-item">
          <span className="spec-label">Model:</span>
          <span className="spec-value">{result.model_name}</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">Grad-CAM Layer:</span>
          <span className="spec-value font-mono">{result.target_conv_layer}</span>
        </div>
      </div>
    </div>
  );
}
