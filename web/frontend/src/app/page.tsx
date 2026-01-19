"use client";

import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CATEGORIES = [
  "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
  "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
  "transistor", "wood", "zipper"
];

const MODEL_TYPES = ["CAE", "VAE", "DAE"];

interface PredictionResult {
  success: boolean;
  model: string;
  category: string;
  anomaly_score: number;
  original_image: string;
  reconstruction: string;
  heatmap: string;
  processing_time: number;
  timestamp?: string;
  filename?: string;
}

interface HistoryItem extends PredictionResult {
  id: string;
  timestamp: string;
  filename: string;
}

export default function Home() {
  const [selectedCategory, setSelectedCategory] = useState("bottle");
  const [uploadedImages, setUploadedImages] = useState<{file: File, preview: string}[]>([]);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  const [compareMode, setCompareMode] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeTab, setActiveTab] = useState<"detect" | "history">("detect");
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });

  // Load history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("defect_history");
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load history", e);
      }
    }
  }, []);

  // Save history to localStorage
  const saveToHistory = (result: PredictionResult, filename: string) => {
    const item: HistoryItem = {
      ...result,
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      filename,
    };
    const newHistory = [item, ...history].slice(0, 50); // Keep last 50
    setHistory(newHistory);
    localStorage.setItem("defect_history", JSON.stringify(newHistory));
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newImages = acceptedFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file)
    }));
    setUploadedImages(prev => [...prev, ...newImages]);
    setResults([]);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".bmp"] },
    multiple: true,
  });

  const removeImage = (index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setUploadedImages([]);
    setResults([]);
    setError(null);
  };

  const analyzeImage = async (file: File, modelType: string): Promise<PredictionResult | null> => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_URL}/predict?model_type=${modelType}&category=${selectedCategory}`,
        { method: "POST", body: formData }
      );

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Prediction failed");
      }

      return await response.json();
    } catch (err) {
      console.error(`Error with ${modelType}:`, err);
      return null;
    }
  };

  const handleAnalyze = async () => {
    if (uploadedImages.length === 0) return;

    setLoading(true);
    setError(null);
    setResults([]);

    const modelsToRun = compareMode ? MODEL_TYPES : ["CAE"];
    const totalOperations = uploadedImages.length * modelsToRun.length;
    setBatchProgress({ current: 0, total: totalOperations });

    const allResults: PredictionResult[] = [];

    try {
      for (let i = 0; i < uploadedImages.length; i++) {
        const { file } = uploadedImages[i];
        
        for (const modelType of modelsToRun) {
          const result = await analyzeImage(file, modelType);
          if (result) {
            allResults.push(result);
            saveToHistory(result, file.name);
          }
          setBatchProgress({ current: allResults.length, total: totalOperations });
        }
      }

      setResults(allResults);
      
      if (allResults.length === 0) {
        setError("No models could process the images");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
      setBatchProgress({ current: 0, total: 0 });
    }
  };

  const downloadImage = (base64: string, filename: string) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${base64}`;
    link.download = filename;
    link.click();
  };

  const downloadAllResults = () => {
    results.forEach((result, index) => {
      setTimeout(() => {
        downloadImage(result.heatmap, `heatmap_${result.model}_${index + 1}.png`);
        downloadImage(result.reconstruction, `reconstruction_${result.model}_${index + 1}.png`);
      }, index * 200);
    });
  };

  const getScoreColor = (score: number) => {
    if (score < 0.3) return "text-green-500";
    if (score < 0.6) return "text-yellow-500";
    return "text-red-500";
  };

  const getScoreLabel = (score: number) => {
    if (score < 0.3) return "Normal";
    if (score < 0.6) return "Suspicious";
    return "Anomaly Detected";
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("defect_history");
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-gray-900"}`}>
      {/* Header */}
      <header className={`border-b ${darkMode ? "border-gray-800 bg-gray-900/80" : "border-gray-200 bg-white/80"} backdrop-blur-sm sticky top-0 z-50`}>
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">🔍</span>
            </div>
            <div>
              <h1 className="text-xl font-bold">Industrial Defect Detection</h1>
              <p className={`text-sm ${darkMode ? "text-gray-400" : "text-gray-600"}`}>Powered by Deep Learning</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* Tabs */}
            <div className={`flex rounded-lg ${darkMode ? "bg-gray-800" : "bg-gray-100"}`}>
              <button
                onClick={() => setActiveTab("detect")}
                className={`px-4 py-2 rounded-lg transition-colors ${activeTab === "detect" ? "bg-blue-500 text-white" : ""}`}
              >
                🔍 Detect
              </button>
              <button
                onClick={() => setActiveTab("history")}
                className={`px-4 py-2 rounded-lg transition-colors ${activeTab === "history" ? "bg-blue-500 text-white" : ""}`}
              >
                📜 History ({history.length})
              </button>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg ${darkMode ? "bg-gray-800 hover:bg-gray-700" : "bg-gray-100 hover:bg-gray-200"}`}
            >
              {darkMode ? "☀️" : "🌙"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === "detect" ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Panel - Upload & Settings */}
            <div className="space-y-6">
              {/* Settings */}
              <div className={`p-6 rounded-xl ${darkMode ? "bg-gray-800" : "bg-white shadow-lg"}`}>
                <h2 className="text-lg font-semibold mb-4">Settings</h2>
                <div className="space-y-4">
                  <div>
                    <label className={`block text-sm font-medium mb-2 ${darkMode ? "text-gray-300" : "text-gray-700"}`}>
                      Category
                    </label>
                    <select
                      value={selectedCategory}
                      onChange={(e) => setSelectedCategory(e.target.value)}
                      className={`w-full p-3 rounded-lg border ${darkMode ? "bg-gray-700 border-gray-600" : "bg-white border-gray-300"} focus:ring-2 focus:ring-blue-500`}
                    >
                      {CATEGORIES.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat.charAt(0).toUpperCase() + cat.slice(1).replace("_", " ")}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Compare Mode Toggle */}
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={compareMode}
                      onChange={(e) => setCompareMode(e.target.checked)}
                      className="w-5 h-5 rounded text-blue-500"
                    />
                    <span>
                      <strong>Compare All Models</strong>
                      <span className={`block text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
                        Run CAE, VAE, and DAE on each image
                      </span>
                    </span>
                  </label>
                </div>
              </div>

              {/* Image Upload */}
              <div className={`p-6 rounded-xl ${darkMode ? "bg-gray-800" : "bg-white shadow-lg"}`}>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Upload Images</h2>
                  {uploadedImages.length > 0 && (
                    <button onClick={clearAll} className="text-sm text-red-400 hover:text-red-300">
                      Clear All
                    </button>
                  )}
                </div>
                
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors
                    ${isDragActive ? "border-blue-500 bg-blue-500/10" : darkMode ? "border-gray-600 hover:border-gray-500" : "border-gray-300 hover:border-gray-400"}
                  `}
                >
                  <input {...getInputProps()} />
                  <div className="text-4xl mb-2">📁</div>
                  <p className={`${darkMode ? "text-gray-300" : "text-gray-600"}`}>
                    {isDragActive ? "Drop images here..." : "Drag & drop images, or click to select"}
                  </p>
                  <p className={`text-sm mt-1 ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
                    Supports multiple files (PNG, JPG, JPEG, BMP)
                  </p>
                </div>

                {/* Preview Grid */}
                {uploadedImages.length > 0 && (
                  <div className="mt-4 grid grid-cols-4 gap-2">
                    {uploadedImages.map((img, index) => (
                      <div key={index} className="relative group">
                        <img src={img.preview} alt={`Preview ${index}`} className="w-full h-20 object-cover rounded-lg" />
                        <button
                          onClick={() => removeImage(index)}
                          className="absolute top-1 right-1 w-5 h-5 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity text-xs"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Analyze Button */}
                <button
                  onClick={handleAnalyze}
                  disabled={uploadedImages.length === 0 || loading}
                  className={`w-full mt-4 py-3 px-6 rounded-lg font-semibold transition-all
                    ${uploadedImages.length > 0 && !loading
                      ? "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white"
                      : "bg-gray-600 text-gray-400 cursor-not-allowed"}
                  `}
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Processing {batchProgress.current}/{batchProgress.total}...
                    </span>
                  ) : (
                    `🔍 Analyze ${uploadedImages.length} Image${uploadedImages.length > 1 ? "s" : ""}${compareMode ? " (×3 Models)" : ""}`
                  )}
                </button>

                {error && (
                  <div className="mt-4 p-3 bg-red-500/20 border border-red-500 rounded-lg text-red-400">
                    {error}
                  </div>
                )}
              </div>
            </div>

            {/* Right Panel - Results */}
            <div className={`p-6 rounded-xl ${darkMode ? "bg-gray-800" : "bg-white shadow-lg"}`}>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold">Results</h2>
                {results.length > 0 && (
                  <button
                    onClick={downloadAllResults}
                    className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg text-sm flex items-center gap-2"
                  >
                    💾 Download All
                  </button>
                )}
              </div>

              {results.length > 0 ? (
                <div className="space-y-6 max-h-[600px] overflow-y-auto">
                  {compareMode && uploadedImages.length === 1 ? (
                    // Model Comparison View
                    <div className="space-y-4">
                      <h3 className={`font-medium ${darkMode ? "text-gray-300" : "text-gray-600"}`}>Model Comparison</h3>
                      <div className="grid grid-cols-3 gap-4">
                        {results.map((result, index) => (
                          <div key={index} className={`p-3 rounded-lg ${darkMode ? "bg-gray-700" : "bg-gray-50"}`}>
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-semibold">{result.model}</span>
                              <span className={`font-bold ${getScoreColor(result.anomaly_score)}`}>
                                {result.anomaly_score.toFixed(3)}
                              </span>
                            </div>
                            <img
                              src={`data:image/png;base64,${result.heatmap}`}
                              alt={`${result.model} Heatmap`}
                              className="w-full rounded-lg cursor-pointer hover:scale-105 transition-transform"
                              onClick={() => downloadImage(result.heatmap, `${result.model}_heatmap.png`)}
                            />
                            <p className={`text-xs mt-2 text-center ${getScoreColor(result.anomaly_score)}`}>
                              {getScoreLabel(result.anomaly_score)}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    // Standard Results View
                    results.map((result, index) => (
                      <div key={index} className={`p-4 rounded-lg ${darkMode ? "bg-gray-700" : "bg-gray-50"}`}>
                        <div className="flex items-center justify-between mb-3">
                          <span className="font-medium">{result.model} - {result.category}</span>
                          <span className={`text-lg font-bold ${getScoreColor(result.anomaly_score)}`}>
                            {result.anomaly_score.toFixed(4)} ({getScoreLabel(result.anomaly_score)})
                          </span>
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                          <div>
                            <p className="text-xs text-gray-400 mb-1">Original</p>
                            <img src={`data:image/png;base64,${result.original_image}`} alt="Original" className="w-full rounded" />
                          </div>
                          <div>
                            <p className="text-xs text-gray-400 mb-1">Reconstruction</p>
                            <img
                              src={`data:image/png;base64,${result.reconstruction}`}
                              alt="Reconstruction"
                              className="w-full rounded cursor-pointer hover:ring-2 ring-blue-500"
                              onClick={() => downloadImage(result.reconstruction, `recon_${result.model}_${index}.png`)}
                            />
                          </div>
                          <div>
                            <p className="text-xs text-gray-400 mb-1">Heatmap</p>
                            <img
                              src={`data:image/png;base64,${result.heatmap}`}
                              alt="Heatmap"
                              className="w-full rounded cursor-pointer hover:ring-2 ring-blue-500"
                              onClick={() => downloadImage(result.heatmap, `heatmap_${result.model}_${index}.png`)}
                            />
                          </div>
                        </div>
                        <p className="text-xs text-gray-500 mt-2">Time: {result.processing_time.toFixed(2)}s | Click image to download</p>
                      </div>
                    ))
                  )}
                </div>
              ) : (
                <div className={`text-center py-16 ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
                  <div className="text-6xl mb-4">📊</div>
                  <p>Upload images and click Analyze to see results</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          // History Tab
          <div className={`p-6 rounded-xl ${darkMode ? "bg-gray-800" : "bg-white shadow-lg"}`}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold">Prediction History</h2>
              {history.length > 0 && (
                <button onClick={clearHistory} className="text-sm text-red-400 hover:text-red-300">
                  Clear History
                </button>
              )}
            </div>

            {history.length > 0 ? (
              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {history.map((item) => (
                  <div key={item.id} className={`p-4 rounded-lg ${darkMode ? "bg-gray-700" : "bg-gray-50"} flex items-center gap-4`}>
                    <img
                      src={`data:image/png;base64,${item.heatmap}`}
                      alt="Heatmap"
                      className="w-20 h-20 rounded-lg object-cover"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{item.model}</span>
                        <span className={`px-2 py-0.5 rounded text-xs ${getScoreColor(item.anomaly_score)} bg-opacity-20`}>
                          {getScoreLabel(item.anomaly_score)}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400">{item.category} | Score: {item.anomaly_score.toFixed(4)}</p>
                      <p className="text-xs text-gray-500">{new Date(item.timestamp).toLocaleString()}</p>
                    </div>
                    <button
                      onClick={() => downloadImage(item.heatmap, `heatmap_${item.id}.png`)}
                      className="px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm"
                    >
                      💾
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className={`text-center py-16 ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
                <div className="text-6xl mb-4">�</div>
                <p>No prediction history yet</p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className={`border-t ${darkMode ? "border-gray-800" : "border-gray-200"} mt-8`}>
        <div className="max-w-7xl mx-auto px-4 py-6 text-center">
          <p className={`${darkMode ? "text-gray-500" : "text-gray-400"}`}>
            Industrial Defect Detection | Bachelor&apos;s Thesis Project | 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
