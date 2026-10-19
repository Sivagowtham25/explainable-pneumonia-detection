import React, { useState, useRef } from 'react';
import { UploadCloud, Sparkles, X, FileText } from 'lucide-react';

export default function ImageUploader({
  selectedFile,
  previewUrl,
  onFileSelected,
  onClearFile,
  onAnalyze,
  isLoading,
  onLoadSample,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelected(file);
      }
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelected(e.target.files[0]);
    }
  };

  const sampleImages = [
    { label: 'Normal Scan #1', path: '/samples/sample_normal_1.jpeg', filename: 'sample_normal_1.jpeg', type: 'NORMAL' },
    { label: 'Normal Scan #2', path: '/samples/sample_normal_2.jpeg', filename: 'sample_normal_2.jpeg', type: 'NORMAL' },
    { label: 'Pneumonia #1', path: '/samples/sample_pneumonia_1.jpeg', filename: 'sample_pneumonia_1.jpeg', type: 'PNEUMONIA' },
    { label: 'Pneumonia #2', path: '/samples/sample_pneumonia_2.jpeg', filename: 'sample_pneumonia_2.jpeg', type: 'PNEUMONIA' },
  ];

  return (
    <div className="uploader-card glass-panel">
      <div className="card-header">
        <h2 className="card-title">
          <UploadCloud size={20} className="card-icon" />
          <span>Chest Radiograph Ingestion</span>
        </h2>
        <span className="format-hint">JPEG / PNG &bull; Max 15 MB</span>
      </div>

      {/* Dropzone Area */}
      {!previewUrl ? (
        <div
          className={`dropzone ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInputChange}
            accept="image/jpeg,image/png,image/webp"
            style={{ display: 'none' }}
          />
          <div className="dropzone-content">
            <div className="upload-icon-circle">
              <UploadCloud size={32} />
            </div>
            <p className="dropzone-primary-text">
              Drag &amp; drop anterior-posterior (AP) chest X-ray
            </p>
            <p className="dropzone-secondary-text">or click to browse files from your computer</p>
          </div>
        </div>
      ) : (
        <div className="preview-container">
          <div className="preview-image-wrapper">
            <img src={previewUrl} alt="Selected radiograph preview" className="preview-img" />
            <button
              type="button"
              className="clear-preview-btn"
              onClick={onClearFile}
              title="Remove image"
            >
              <X size={16} />
            </button>
          </div>
          <div className="preview-metadata">
            <div className="file-info">
              <FileText size={16} className="file-icon" />
              <div>
                <p className="filename-text">{selectedFile?.name || 'radiograph.jpeg'}</p>
                <p className="filesize-text">
                  {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : 'Preloaded Image'}
                </p>
              </div>
            </div>
            <button
              type="button"
              className="change-file-btn"
              onClick={() => fileInputRef.current?.click()}
            >
              Change
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept="image/jpeg,image/png,image/webp"
              style={{ display: 'none' }}
            />
          </div>
        </div>
      )}

      {/* Quick Test Samples */}
      <div className="samples-section">
        <p className="samples-title">Quick Test with Benchmark Radiographs:</p>
        <div className="samples-grid">
          {sampleImages.map((s, idx) => (
            <button
              key={idx}
              type="button"
              className={`sample-pill ${s.type === 'PNEUMONIA' ? 'pneu-sample' : 'norm-sample'}`}
              onClick={() => onLoadSample(s.path, s.filename)}
            >
              <span className={`sample-dot ${s.type === 'PNEUMONIA' ? 'dot-red' : 'dot-green'}`}></span>
              <span>{s.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Analysis Call to Action */}
      <button
        type="button"
        className={`analyze-submit-btn ${isLoading ? 'loading' : ''}`}
        disabled={!selectedFile || isLoading}
        onClick={onAnalyze}
      >
        {isLoading ? (
          <>
            <span className="spinner"></span>
            <span>Running Inference &amp; Grad-CAM...</span>
          </>
        ) : (
          <>
            <Sparkles size={18} />
            <span>Run Diagnostic Inference</span>
          </>
        )}
      </button>
    </div>
  );
}
