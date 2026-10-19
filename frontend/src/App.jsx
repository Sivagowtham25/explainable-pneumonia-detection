import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import PredictionCard from './components/PredictionCard';
import GradCamViewer from './components/GradCamViewer';
import BenchmarksModal from './components/BenchmarksModal';
import Footer from './components/Footer';
import { fetchHealth, fetchModels, predictImage, loadSampleFile } from './services/api';
import { Layers, AlertCircle, X } from 'lucide-react';
import './App.css';

export default function App() {
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('mobilenetv2');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isBenchmarksOpen, setIsBenchmarksOpen] = useState(false);

  // Check health and models on initial load
  useEffect(() => {
    let mounted = true;

    async function initBackend() {
      try {
        const healthData = await fetchHealth();
        if (mounted) setHealth(healthData);
      } catch (err) {
        console.warn('Initial health check failed:', err);
      }

      try {
        const modelsData = await fetchModels();
        if (mounted && modelsData?.models) {
          setModels(modelsData.models);
        }
      } catch (err) {
        console.warn('Models catalog fetch failed:', err);
      }
    }

    initBackend();

    // Periodic health poll every 10 seconds
    const interval = setInterval(async () => {
      try {
        const h = await fetchHealth();
        if (mounted) setHealth(h);
      } catch {
        if (mounted) setHealth({ system_ready: false });
      }
    }, 10000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleFileSelected = (file) => {
    setError(null);
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleClearFile = () => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const executeInference = async (file, modelId) => {
    setIsLoading(true);
    setError(null);
    try {
      const prediction = await predictImage(
        file,
        file.name || 'radiograph.jpeg',
        modelId,
        0.4,
        true
      );
      setResult(prediction);
    } catch (err) {
      console.error('Inference error:', err);
      setError(err.message || 'An unexpected diagnostic inference error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;
    executeInference(selectedFile, selectedModel);
  };

  const handleSelectModel = (newModelId) => {
    setSelectedModel(newModelId);
    // If an image is already loaded and was analyzed, re-run inference with the new model
    if (selectedFile && result) {
      executeInference(selectedFile, newModelId);
    }
  };

  const handleLoadSample = async (samplePath, filename) => {
    setError(null);
    setIsLoading(true);
    try {
      const file = await loadSampleFile(samplePath, filename);
      setSelectedFile(file);
      setPreviewUrl(samplePath);
      // Automatically trigger inference for seamless demonstration
      await executeInference(file, selectedModel);
    } catch (err) {
      console.error('Failed to load sample image:', err);
      setError(`Failed to load sample radiograph: ${err.message}`);
      setIsLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <Header
        health={health}
        models={models}
        selectedModel={selectedModel}
        onSelectModel={handleSelectModel}
        onOpenBenchmarks={() => setIsBenchmarksOpen(true)}
      />

      {/* Global Error Banner */}
      {error && (
        <div className="error-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
          <button
            type="button"
            className="error-dismiss-btn"
            onClick={() => setError(null)}
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* Main Clinical Dashboard */}
      <main className="dashboard-grid">
        <section className="left-panel">
          <ImageUploader
            selectedFile={selectedFile}
            previewUrl={previewUrl}
            onFileSelected={handleFileSelected}
            onClearFile={handleClearFile}
            onAnalyze={handleAnalyze}
            isLoading={isLoading}
            onLoadSample={handleLoadSample}
          />
        </section>

        <section className="right-panel">
          {result ? (
            <>
              <PredictionCard result={result} />
              <GradCamViewer result={result} />
            </>
          ) : (
            <div className="empty-state-card glass-panel">
              <div className="upload-icon-circle">
                <Layers size={36} className="empty-icon" />
              </div>
              <h3 className="empty-title">Awaiting Radiograph Ingestion</h3>
              <p className="empty-desc">
                Upload an anterior-posterior (AP) chest radiograph or choose one of the benchmark sample scans
                on the left to generate real-time AI pneumonia classification and Grad-CAM anatomical saliency maps.
              </p>
            </div>
          )}
        </section>
      </main>

      <Footer />

      <BenchmarksModal
        isOpen={isBenchmarksOpen}
        onClose={() => setIsBenchmarksOpen(false)}
      />
    </div>
  );
}
