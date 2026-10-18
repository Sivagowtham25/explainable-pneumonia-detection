import React from 'react';
import { X, Award, ShieldAlert, Cpu, Check, Layers, BarChart } from 'lucide-react';

export default function BenchmarksModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const benchmarkData = [
    {
      metric: 'Test Accuracy',
      baseline: '83.17%',
      mobilenet: '78.53%',
      note: 'Baseline achieved higher overall threshold accuracy at default 0.5 boundary.',
    },
    {
      metric: 'ROC-AUC',
      baseline: '0.9252',
      mobilenet: '0.9566',
      note: 'MobileNetV2 exhibits superior probability separation between classes.',
      highlight: true,
    },
    {
      metric: 'Pneumonia Recall (Sensitivity)',
      baseline: '98.21%',
      mobilenet: '98.72%',
      note: 'MobileNetV2 missed only 5 out of 390 cases (lowest clinical risk).',
      highlight: true,
    },
    {
      metric: 'Pneumonia Precision',
      baseline: '79.63%',
      mobilenet: '74.90%',
      note: 'Baseline maintained slightly lower false positive rate on normal images.',
    },
    {
      metric: 'Pneumonia F1-Score',
      baseline: '87.94%',
      mobilenet: '85.18%',
      note: 'Harmonic balance between sensitivity and precision.',
    },
    {
      metric: 'Normal Specificity',
      baseline: '58.12%',
      mobilenet: '44.87%',
      note: 'Reflects trade-off chosen to prevent fatal false-negative pneumonia misses.',
    },
    {
      metric: 'Total Parameters',
      baseline: '258,049 (~1 MB)',
      mobilenet: '2,427,201 (~26 MB)',
      note: 'Baseline is ultra-lightweight; MobileNetV2 features pre-trained ImageNet depth.',
    },
    {
      metric: 'Target Grad-CAM Layer',
      baseline: 'conv4_features (28x28x128)',
      mobilenet: 'Conv_1 (7x7x1280)',
      note: 'Layer used for extracting class-discriminative spatial activation maps.',
    },
  ];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog glass-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <BarChart size={22} className="modal-icon" />
            <h2 className="modal-title">Model Benchmark &amp; Clinical Evaluation</h2>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <p className="modal-subtitle">
          Evaluated on the official un-augmented 624-image test set (234 Normal, 390 Pneumonia).
        </p>

        {/* Comparison Table */}
        <div className="modal-table-container">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Diagnostic Metric</th>
                <th>Custom Baseline CNN</th>
                <th>MobileNetV2 (Fine-Tuned)</th>
                <th>Clinical Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkData.map((row, idx) => (
                <tr key={idx} className={row.highlight ? 'row-highlight' : ''}>
                  <td className="metric-cell font-semibold">{row.metric}</td>
                  <td className="baseline-cell font-mono">{row.baseline}</td>
                  <td className="mobilenet-cell font-mono">{row.mobilenet}</td>
                  <td className="note-cell">{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Clinical Prioritization Callout */}
        <div className="clinical-callout">
          <ShieldAlert size={20} className="callout-icon" />
          <div className="callout-body">
            <h4>Clinical Triage Prioritization (Minimizing False Negatives)</h4>
            <p>
              In medical radiograph diagnostics, a <strong>False Negative</strong> (classifying an active
              pneumonia infection as normal) can be fatal to the patient, whereas a False Positive merely
              prompts secondary radiological review. Both models achieve over <strong>98% Sensitivity</strong>,
              with MobileNetV2 capturing <strong>385 out of 390 pneumonia cases</strong>.
            </p>
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="close-action-btn" onClick={onClose}>
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
