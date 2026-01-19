"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Settings, Upload, X, Zap, BarChart3, Download, Clock, RefreshCw } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"];
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
}

export default function DetectPage() {
  const [selectedCategory, setSelectedCategory] = useState("bottle");
  const [uploadedImages, setUploadedImages] = useState<{file: File, preview: string}[]>([]);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });

  const saveToHistory = (result: PredictionResult, filename: string) => {
    const saved = localStorage.getItem("defect_history");
    const history = saved ? JSON.parse(saved) : [];
    const item = { ...result, id: Date.now().toString(), timestamp: new Date().toISOString(), filename };
    localStorage.setItem("defect_history", JSON.stringify([item, ...history].slice(0, 50)));
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newImages = acceptedFiles.map(file => ({ file, preview: URL.createObjectURL(file) }));
    setUploadedImages(prev => [...prev, ...newImages]);
    setResults([]);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { "image/*": [".png", ".jpg", ".jpeg", ".bmp"] }, multiple: true });

  const analyzeImage = async (file: File, modelType: string): Promise<PredictionResult | null> => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_URL}/predict?model_type=${modelType}&category=${selectedCategory}`, { method: "POST", body: formData });
      if (!response.ok) throw new Error((await response.json()).detail || "Failed");
      return await response.json();
    } catch { return null; }
  };

  const handleAnalyze = async () => {
    if (uploadedImages.length === 0) return;
    setLoading(true); setError(null); setResults([]);
    const modelsToRun = compareMode ? MODEL_TYPES : ["CAE"];
    const total = uploadedImages.length * modelsToRun.length;
    setBatchProgress({ current: 0, total });
    const allResults: PredictionResult[] = [];
    for (const { file } of uploadedImages) {
      for (const modelType of modelsToRun) {
        const result = await analyzeImage(file, modelType);
        if (result) { allResults.push(result); saveToHistory(result, file.name); }
        setBatchProgress({ current: allResults.length, total });
      }
    }
    setResults(allResults);
    if (allResults.length === 0) setError("No models could process the images");
    setLoading(false);
  };

  const downloadImage = (base64: string, filename: string) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${base64}`;
    link.download = filename;
    link.click();
  };

  const getScoreColor = (s: number) => s < 0.3 ? "text-emerald-400" : s < 0.6 ? "text-amber-400" : "text-rose-400";
  const getScoreLabel = (s: number) => s < 0.3 ? "Normal" : s < 0.6 ? "Suspicious" : "Anomaly";
  const getScoreBg = (s: number) => s < 0.3 ? "from-emerald-500/20" : s < 0.6 ? "from-amber-500/20" : "from-rose-500/20";

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <motion.h1 initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-3xl font-bold mb-8 flex items-center gap-3">
        <Search className="w-8 h-8 text-blue-400" />
        <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Defect Detection</span>
      </motion.h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Panel */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="space-y-6">
          {/* Settings */}
          <div className="p-6 rounded-2xl bg-slate-800/50 border border-white/10 backdrop-blur">
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-blue-400" /> Settings
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-slate-300 mb-2">Category</label>
                <select value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)} className="w-full p-3 rounded-xl bg-slate-700/50 border border-slate-600 text-white">
                  {CATEGORIES.map((cat) => <option key={cat} value={cat}>{cat.charAt(0).toUpperCase() + cat.slice(1).replace("_", " ")}</option>)}
                </select>
              </div>
              <label className={`flex items-center gap-3 p-3 rounded-xl cursor-pointer ${compareMode ? "bg-blue-500/20 border border-blue-500/30" : "bg-slate-700/30"}`}>
                <input type="checkbox" checked={compareMode} onChange={(e) => setCompareMode(e.target.checked)} className="w-5 h-5 accent-blue-500" />
                <div>
                  <span className="font-semibold flex items-center gap-2"><RefreshCw className="w-4 h-4" /> Compare All Models</span>
                  <span className="block text-sm text-slate-400">Run CAE, VAE, DAE side-by-side</span>
                </div>
              </label>
            </div>
          </div>

          {/* Upload */}
          <div className="p-6 rounded-2xl bg-slate-800/50 border border-white/10 backdrop-blur">
            <div className="flex justify-between mb-4">
              <h2 className="text-lg font-bold flex items-center gap-2">
                <Upload className="w-5 h-5 text-blue-400" /> Upload
              </h2>
              {uploadedImages.length > 0 && <button onClick={() => { setUploadedImages([]); setResults([]); }} className="text-sm text-rose-400 flex items-center gap-1"><X className="w-4 h-4" /> Clear</button>}
            </div>
            <div {...getRootProps()} className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer ${isDragActive ? "border-blue-500 bg-blue-500/10" : "border-slate-600"}`}>
              <input {...getInputProps()} />
              <Upload className={`w-12 h-12 mx-auto mb-3 ${isDragActive ? "text-blue-400" : "text-slate-500"}`} />
              <p className="text-slate-300">{isDragActive ? "Drop it!" : "Drag & drop or click"}</p>
            </div>
            {uploadedImages.length > 0 && (
              <div className="mt-4 grid grid-cols-4 gap-3">
                {uploadedImages.map((img, i) => (
                  <div key={i} className="relative group rounded-xl overflow-hidden">
                    <img src={img.preview} alt="" className="w-full h-20 object-cover" />
                    <button onClick={() => setUploadedImages(prev => prev.filter((_, idx) => idx !== i))} className="absolute top-1 right-1 w-6 h-6 bg-rose-500 text-white rounded-full opacity-0 group-hover:opacity-100 flex items-center justify-center"><X className="w-4 h-4" /></button>
                  </div>
                ))}
              </div>
            )}
            <motion.button onClick={handleAnalyze} disabled={uploadedImages.length === 0 || loading} whileHover={uploadedImages.length > 0 && !loading ? { scale: 1.02 } : {}} className={`w-full mt-5 py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 ${uploadedImages.length > 0 && !loading ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white" : "bg-slate-700 text-slate-400 cursor-not-allowed"}`}>
              {loading ? <><Zap className="w-5 h-5 animate-pulse" /> Processing {batchProgress.current}/{batchProgress.total}...</> : <><Search className="w-5 h-5" /> Analyze {uploadedImages.length} Image{uploadedImages.length > 1 ? "s" : ""}{compareMode ? " x 3" : ""}</>}
            </motion.button>
            {error && <div className="mt-4 p-4 bg-rose-500/20 border border-rose-500/30 rounded-xl text-rose-400">{error}</div>}
          </div>
        </motion.div>

        {/* Results */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="p-6 rounded-2xl bg-slate-800/50 border border-white/10 backdrop-blur">
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-400" /> Results
          </h2>
          <AnimatePresence mode="wait">
            {results.length > 0 ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4 max-h-[600px] overflow-y-auto">
                {results.map((result, i) => (
                  <motion.div key={i} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`p-4 rounded-xl bg-gradient-to-br ${getScoreBg(result.anomaly_score)} to-transparent border border-white/5`}>
                    <div className="flex justify-between mb-3">
                      <span className="px-3 py-1 rounded-lg bg-slate-700/50 font-bold">{result.model}</span>
                      <div className="text-right">
                        <span className={`text-2xl font-black ${getScoreColor(result.anomaly_score)}`}>{result.anomaly_score.toFixed(3)}</span>
                        <span className={`block text-sm ${getScoreColor(result.anomaly_score)}`}>{getScoreLabel(result.anomaly_score)}</span>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      {[{ label: "Original", img: result.original_image }, { label: "Reconstruction", img: result.reconstruction }, { label: "Heatmap", img: result.heatmap }].map((item, j) => (
                        <div key={j} className="cursor-pointer hover:scale-105 transition-transform" onClick={() => downloadImage(item.img, `${item.label}_${result.model}.png`)}>
                          <p className="text-xs text-slate-400 mb-1">{item.label}</p>
                          <img src={`data:image/png;base64,${item.img}`} alt={item.label} className="w-full rounded-lg" />
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-slate-500 mt-2 flex items-center gap-1"><Clock className="w-3 h-3" /> {result.processing_time.toFixed(2)}s | Click to download</p>
                  </motion.div>
                ))}
              </motion.div>
            ) : (
              <div className="text-center py-20 text-slate-500">
                <Search className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p>Upload an image to start</p>
              </div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}
