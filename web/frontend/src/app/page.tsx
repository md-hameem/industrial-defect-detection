"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CATEGORIES = [
  "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
  "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
  "transistor", "wood", "zipper"
];

interface PredictionResult {
  success: boolean;
  model: string;
  category: string;
  anomaly_score: number;
  original_image: string;
  reconstruction: string;
  heatmap: string;
  processing_time: number;
}

export default function Home() {
  const [selectedModel, setSelectedModel] = useState("CAE");
  const [selectedCategory, setSelectedCategory] = useState("bottle");
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setUploadedImage(reader.result as string);
        setResult(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".bmp"] },
    multiple: false,
  });

  const handleAnalyze = async () => {
    if (!uploadedImage) return;

    setLoading(true);
    setError(null);

    try {
      // Convert base64 to blob
      const response = await fetch(uploadedImage);
      const blob = await response.blob();
      
      const formData = new FormData();
      formData.append("file", blob, "image.png");

      const apiResponse = await fetch(
        `${API_URL}/predict?model_type=${selectedModel}&category=${selectedCategory}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!apiResponse.ok) {
        const errData = await apiResponse.json();
        throw new Error(errData.detail || "Prediction failed");
      }

      const data: PredictionResult = await apiResponse.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
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
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-lg ${darkMode ? "bg-gray-800 hover:bg-gray-700" : "bg-gray-100 hover:bg-gray-200"}`}
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Upload & Settings */}
          <div className="space-y-6">
            {/* Model Selection */}
            <div className={`p-6 rounded-xl ${darkMode ? "bg-gray-800" : "bg-white shadow-lg"}`}>
              <h2 className="text-lg font-semibold mb-4">Model Settings</h2>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className={`block text-sm font-medium mb-2 ${darkMode ? "text-gray-300" : "text-gray-700"}`}>
                    Model Type
                  </label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className={`w-full p-3 rounded-lg border ${darkMode ? "bg-gray-700 border-gray-600" : "bg-white border-gray-300"} focus:ring-2 focus:ring-blue-500`}
                  >
                    <option value="CAE">CAE (Convolutional AE)</option>
                    <option value="VAE">VAE (Variational AE)</option>
                    <option value="DAE">DAE (Denoising AE)</option>
                  </select>
                </div>
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
              </div>
            </div>

            {/* Image Upload */}
            <div className={`p-6 rounded-xl ${darkMode ? "bg-gray-800" : "bg-white shadow-lg"}`}>
              <h2 className="text-lg font-semibold mb-4">Upload Image</h2>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors
                  ${isDragActive ? "border-blue-500 bg-blue-500/10" : darkMode ? "border-gray-600 hover:border-gray-500" : "border-gray-300 hover:border-gray-400"}
                `}
              >
                <input {...getInputProps()} />
                {uploadedImage ? (
                  <img
                    src={uploadedImage}
                    alt="Uploaded"
                    className="max-h-64 mx-auto rounded-lg"
                  />
                ) : (
                  <div className="py-8">
                    <div className="text-4xl mb-4">📁</div>
                    <p className={`${darkMode ? "text-gray-300" : "text-gray-600"}`}>
                      {isDragActive ? "Drop the image here..." : "Drag & drop an image, or click to select"}
                    </p>
                    <p className={`text-sm mt-2 ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
                      Supports PNG, JPG, JPEG, BMP
                    </p>
                  </div>
                )}
              </div>

              {/* Analyze Button */}
              <button
                onClick={handleAnalyze}
                disabled={!uploadedImage || loading}
                className={`w-full mt-4 py-3 px-6 rounded-lg font-semibold transition-all
                  ${uploadedImage && !loading
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
                    Analyzing...
                  </span>
                ) : (
                  "🔍 Analyze Image"
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
            <h2 className="text-lg font-semibold mb-4">Detection Results</h2>

            {result ? (
              <div className="space-y-6">
                {/* Score */}
                <div className={`p-4 rounded-lg ${darkMode ? "bg-gray-700" : "bg-gray-50"}`}>
                  <div className="flex items-center justify-between">
                    <span className={`${darkMode ? "text-gray-300" : "text-gray-600"}`}>Anomaly Score</span>
                    <span className={`text-2xl font-bold ${getScoreColor(result.anomaly_score)}`}>
                      {result.anomaly_score.toFixed(4)}
                    </span>
                  </div>
                  <div className={`mt-2 text-lg font-semibold ${getScoreColor(result.anomaly_score)}`}>
                    {getScoreLabel(result.anomaly_score)}
                  </div>
                  <div className="mt-2 text-sm text-gray-500">
                    Model: {result.model} | Category: {result.category} | Time: {result.processing_time.toFixed(2)}s
                  </div>
                </div>

                {/* Images Grid */}
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p className={`text-sm font-medium mb-2 ${darkMode ? "text-gray-300" : "text-gray-600"}`}>Original</p>
                    <img
                      src={`data:image/png;base64,${result.original_image}`}
                      alt="Original"
                      className="w-full rounded-lg"
                    />
                  </div>
                  <div>
                    <p className={`text-sm font-medium mb-2 ${darkMode ? "text-gray-300" : "text-gray-600"}`}>Reconstruction</p>
                    <img
                      src={`data:image/png;base64,${result.reconstruction}`}
                      alt="Reconstruction"
                      className="w-full rounded-lg"
                    />
                  </div>
                  <div>
                    <p className={`text-sm font-medium mb-2 ${darkMode ? "text-gray-300" : "text-gray-600"}`}>Anomaly Heatmap</p>
                    <img
                      src={`data:image/png;base64,${result.heatmap}`}
                      alt="Heatmap"
                      className="w-full rounded-lg"
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className={`text-center py-16 ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
                <div className="text-6xl mb-4">📊</div>
                <p>Upload an image and click &quot;Analyze&quot; to see results</p>
              </div>
            )}
          </div>
        </div>
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
