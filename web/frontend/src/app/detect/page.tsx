"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { useTheme } from "@/context/ThemeContext";
import { ErrorAlert, ProgressBar, ResultSkeleton, Spinner } from "@/components/LoadingUI";
import { 
  Search, Settings, Upload, X, Zap, BarChart3, Download, Clock, RefreshCw, 
  Image as ImageIcon, Layers, CheckCircle, AlertTriangle, AlertCircle,
  ChevronDown, Info, Tag, Cpu, WifiOff
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"];
const AUTOENCODER_TYPES = ["CAE", "VAE", "DAE", "SKIP_CAE", "PATCHCORE"];
const NEU_CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"];

interface AutoencoderResult {
  success: boolean;
  model: string;
  model_type: "autoencoder" | "anomaly_detector";
  category: string;
  anomaly_score: number;
  original_image: string;
  reconstruction: string;
  heatmap: string;
  processing_time: number;
}

interface ClassifierResult {
  success: boolean;
  model: string;
  model_type: "classifier";
  category: string;
  predicted_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  original_image: string;
  chart_image: string;
  processing_time: number;
}

type PredictionResult = AutoencoderResult | ClassifierResult;

export default function DetectPage() {
  const { darkMode } = useTheme();
  const [selectedCategory, setSelectedCategory] = useState("bottle");
  const [selectedModel, setSelectedModel] = useState("CAE");
  const [useCNN, setUseCNN] = useState(false);
  const [uploadedImages, setUploadedImages] = useState<{file: File, preview: string}[]>([]);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [expandedResult, setExpandedResult] = useState<number | string | null>(null);

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
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);
      const response = await fetch(`${API_URL}/predict?model_type=${modelType}&category=${selectedCategory}`, { 
        method: "POST", 
        body: formData,
        signal: controller.signal 
      });
      clearTimeout(timeoutId);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('Request timed out. The server may be overloaded.');
        } else if (err.message.includes('Failed to fetch')) {
          setError('Cannot connect to server. Please ensure the backend is running on port 8000.');
        } else {
          setError(err.message);
        }
      }
      return null;
    }
  };

  const analyzeBatch = async (file: File, modelTypes: string[]): Promise<PredictionResult[]> => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);
      const response = await fetch(
        `${API_URL}/predict/batch?model_types=${modelTypes.join(",")}&category=${selectedCategory}`,
        { method: "POST", body: formData, signal: controller.signal }
      );
      clearTimeout(timeoutId);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }
      const data = await response.json();
      return (data.results || []).filter((r: any) => r.success !== false);
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('Batch request timed out.');
        } else if (err.message.includes('Failed to fetch')) {
          setError('Cannot connect to server.');
        }
      }
      return [];
    }
  };

  const handleAnalyze = async () => {
    if (uploadedImages.length === 0) return;
    setLoading(true); setError(null); setResults([]);
    
    let modelsToRun: string[];
    if (useCNN) {
      modelsToRun = ["CNN"];
    } else if (compareMode) {
      modelsToRun = AUTOENCODER_TYPES;
    } else {
      modelsToRun = [selectedModel];
    }
    
    const total = uploadedImages.length * modelsToRun.length;
    setBatchProgress({ current: 0, total });
    const allResults: PredictionResult[] = [];
    
    if (compareMode && modelsToRun.length > 1) {
      for (const { file } of uploadedImages) {
        const batchResults = await analyzeBatch(file, modelsToRun);
        for (const result of batchResults) {
          allResults.push(result);
          saveToHistory(result, file.name);
        }
        setBatchProgress({ current: allResults.length, total });
      }
    } else {
      for (const { file } of uploadedImages) {
        for (const modelType of modelsToRun) {
          const result = await analyzeImage(file, modelType);
          if (result) { allResults.push(result); saveToHistory(result, file.name); }
          setBatchProgress({ current: allResults.length, total });
        }
      }
    }
    
    setResults(allResults);
    if (allResults.length === 0) setError("No models could process the images. Make sure the backend is running.");
    setLoading(false);
  };

  const downloadImage = (base64: string, filename: string) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${base64}`;
    link.download = filename;
    link.click();
  };

  const downloadAllResults = () => {
    results.forEach((result, i) => {
      setTimeout(() => {
        if (result.model_type === "classifier") {
          downloadImage(result.chart_image, `chart_CNN_${i+1}.png`);
        } else {
          downloadImage(result.heatmap, `heatmap_${result.model}_${result.category}_${i+1}.png`);
        }
      }, i * 100);
    });
  };

  const getScoreColor = (s: number) => s < 0.3 ? "text-emerald-400" : s < 0.6 ? "text-amber-400" : "text-rose-400";
  const getScoreLabel = (s: number) => s < 0.3 ? "Normal" : s < 0.6 ? "Suspicious" : "Anomaly";
  const getScoreBg = (s: number) => s < 0.3 ? "from-emerald-500/10 to-emerald-500/5" : s < 0.6 ? "from-amber-500/10 to-amber-500/5" : "from-rose-500/10 to-rose-500/5";
  const getScoreGlow = (s: number) => s < 0.3 ? "shadow-emerald-500/10" : s < 0.6 ? "shadow-amber-500/10" : "shadow-rose-500/10";
  const getScoreIcon = (s: number) => s < 0.3 ? CheckCircle : s < 0.6 ? AlertTriangle : AlertCircle;

  const autoencoderResults = results.filter((r): r is AutoencoderResult => r.model_type === "autoencoder" || r.model_type === "anomaly_detector");
  const classifierResults = results.filter((r): r is ClassifierResult => r.model_type === "classifier");

  const groupedResults = compareMode ? (() => {
    const groups: { imageIndex: number; original: string; models: { [key: string]: AutoencoderResult } }[] = [];
    const numImages = uploadedImages.length;
    for (let i = 0; i < numImages; i++) {
      const group: { imageIndex: number; original: string; models: { [key: string]: AutoencoderResult } } = {
        imageIndex: i,
        original: uploadedImages[i]?.preview || "",
        models: {}
      };
      AUTOENCODER_TYPES.forEach((modelType, modelIdx) => {
        const resultIdx = i * AUTOENCODER_TYPES.length + modelIdx;
        if (autoencoderResults[resultIdx]) {
          group.models[modelType] = autoencoderResults[resultIdx];
        }
      });
      if (Object.keys(group.models).length > 0) {
        groups.push(group);
      }
    }
    return groups;
  })() : [];

  const stats = results.length > 0 ? {
    total: results.length,
    normal: autoencoderResults.filter(r => r.anomaly_score < 0.3).length,
    suspicious: autoencoderResults.filter(r => r.anomaly_score >= 0.3 && r.anomaly_score < 0.6).length,
    anomaly: autoencoderResults.filter(r => r.anomaly_score >= 0.6).length,
    avgTime: results.reduce((sum, r) => sum + r.processing_time, 0) / results.length,
  } : null;

  // Theme-aware classes
  const cardClass = darkMode ? "glass inner-glow" : "bg-white/80 border border-slate-200 shadow-sm";
  const textPrimary = darkMode ? "text-white" : "text-slate-900";
  const textSecondary = darkMode ? "text-slate-400" : "text-slate-600";
  const textMuted = darkMode ? "text-slate-500" : "text-slate-400";
  const inputBg = darkMode ? "bg-white/[0.05] border-white/[0.08] text-white" : "bg-slate-100 border-slate-300 text-slate-900";
  const subtleBg = darkMode ? "bg-white/[0.03]" : "bg-slate-100";

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold mb-3 flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-blue-500/30 rounded-xl blur-xl glow-halo" />
            <div className="relative w-12 h-12 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
              <Search className="w-6 h-6 text-white" />
            </div>
          </div>
          <span className="text-gradient-animated">Defect Detection</span>
        </h1>
        <p className={textSecondary}>Upload industrial images for AI-powered anomaly analysis</p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-7 gap-6">
        {/* ===== LEFT PANEL ===== */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="lg:col-span-2 space-y-5">
          
          {/* Detection Mode */}
          <div className={`p-6 rounded-2xl ${cardClass}`}>
            <h2 className={`text-lg font-bold mb-4 flex items-center gap-2 ${textPrimary}`}>
              <Cpu className="w-5 h-5 text-blue-400" /> Detection Mode
            </h2>
            <div className="grid grid-cols-2 gap-3">
              <motion.button
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setUseCNN(false)}
                className={`p-4 rounded-xl text-left transition-all cursor-pointer ${
                  !useCNN
                    ? `bg-gradient-to-br from-blue-500/15 to-purple-500/10 border-2 ${darkMode ? "border-blue-500/40" : "border-blue-400"}`
                    : `${subtleBg} border-2 border-transparent hover:border-blue-500/20`
                }`}
              >
                <div className={`font-bold mb-1 ${textPrimary}`}>Anomaly Detection</div>
                <div className={`text-xs ${textSecondary}`}>CAE, VAE, DAE + 2 New</div>
                <div className={`text-xs ${textMuted} mt-2`}>Heatmaps + PatchCore SOTA</div>
              </motion.button>
              <motion.button
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => { setUseCNN(true); setCompareMode(false); }}
                className={`p-4 rounded-xl text-left transition-all cursor-pointer ${
                  useCNN
                    ? `bg-gradient-to-br from-emerald-500/15 to-teal-500/10 border-2 ${darkMode ? "border-emerald-500/40" : "border-emerald-400"}`
                    : `${subtleBg} border-2 border-transparent hover:border-emerald-500/20`
                }`}
              >
                <div className={`font-bold mb-1 flex items-center gap-2 ${textPrimary}`}>CNN <Tag className="w-3 h-3" /></div>
                <div className={`text-xs ${textSecondary}`}>Classifier</div>
                <div className={`text-xs ${textMuted} mt-2`}>Defect type classification</div>
              </motion.button>
            </div>
          </div>

          {/* Settings Card */}
          <div className={`p-6 rounded-2xl ${cardClass}`}>
            <h2 className={`text-lg font-bold mb-4 flex items-center gap-2 ${textPrimary}`}>
              <Settings className="w-5 h-5 text-blue-400" /> Settings
            </h2>
            
            {!useCNN ? (
              <>
                {/* Model Selection */}
                <div className="mb-4">
                  <label className={`block text-sm mb-2 ${textSecondary}`}>Model</label>
                  <div className="grid grid-cols-3 gap-2 mb-2">
                    {["CAE", "VAE", "DAE"].map((model) => (
                      <motion.button
                        key={model}
                        whileHover={{ y: -1 }}
                        whileTap={{ scale: 0.97 }}
                        onClick={() => { setSelectedModel(model); setCompareMode(false); }}
                        disabled={compareMode}
                        className={`p-3 rounded-xl text-center transition-all cursor-pointer ${
                          !compareMode && selectedModel === model
                            ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg shadow-blue-500/20"
                            : compareMode
                            ? `${subtleBg} text-slate-500 cursor-not-allowed`
                            : `${subtleBg} ${darkMode ? "text-slate-300 hover:bg-white/[0.06]" : "text-slate-700 hover:bg-slate-200"}`
                        }`}
                      >
                        <div className="font-bold">{model}</div>
                        <div className="text-xs opacity-70">{model === "CAE" ? "Standard" : model === "VAE" ? "Probabilistic" : "Denoising"}</div>
                      </motion.button>
                    ))}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <motion.button
                      whileHover={{ y: -1 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => { setSelectedModel("SKIP_CAE"); setCompareMode(false); }}
                      disabled={compareMode}
                      className={`p-3 rounded-xl text-center transition-all cursor-pointer ${
                        !compareMode && selectedModel === "SKIP_CAE"
                          ? "bg-gradient-to-r from-teal-500 to-cyan-600 text-white shadow-lg shadow-teal-500/20"
                          : compareMode
                          ? `${subtleBg} text-slate-500 cursor-not-allowed`
                          : `${subtleBg} ${darkMode ? "text-slate-300 hover:bg-white/[0.06]" : "text-slate-700 hover:bg-slate-200"}`
                      }`}
                    >
                      <div className="font-bold">Skip-CAE</div>
                      <div className="text-xs opacity-70">U-Net Style</div>
                    </motion.button>
                    <motion.button
                      whileHover={{ y: -1 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => { setSelectedModel("PATCHCORE"); setCompareMode(false); }}
                      disabled={compareMode}
                      className={`p-3 rounded-xl text-center transition-all cursor-pointer ${
                        !compareMode && selectedModel === "PATCHCORE"
                          ? "bg-gradient-to-r from-rose-500 to-pink-600 text-white shadow-lg shadow-rose-500/20"
                          : compareMode
                          ? `${subtleBg} text-slate-500 cursor-not-allowed`
                          : `${subtleBg} ${darkMode ? "text-slate-300 hover:bg-white/[0.06]" : "text-slate-700 hover:bg-slate-200"}`
                      }`}
                    >
                      <div className="font-bold flex items-center gap-1 justify-center">PatchCore <span className="text-[10px] px-1.5 py-0.5 bg-white/20 rounded-full">SOTA</span></div>
                      <div className="text-xs opacity-70">Feature-Based</div>
                    </motion.button>
                  </div>
                </div>

                {/* Category Selection */}
                <div className="mb-4">
                  <label className={`block text-sm mb-2 ${textSecondary}`}>Category</label>
                  <div className="relative">
                    <select value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)} className={`w-full p-3 rounded-xl border appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500/40 ${inputBg}`}>
                      {CATEGORIES.map((cat) => <option key={cat} value={cat}>{cat.charAt(0).toUpperCase() + cat.slice(1).replace("_", " ")}</option>)}
                    </select>
                    <ChevronDown className={`absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 ${textMuted} pointer-events-none`} />
                  </div>
                </div>

                {/* Compare Mode */}
                <label className={`flex items-center gap-3 p-4 rounded-xl cursor-pointer transition-all ${
                  compareMode
                    ? `bg-gradient-to-r from-blue-500/10 to-purple-500/10 border ${darkMode ? "border-blue-500/30" : "border-blue-300"}`
                    : subtleBg
                }`}>
                  <input type="checkbox" checked={compareMode} onChange={(e) => setCompareMode(e.target.checked)} className="w-5 h-5 accent-blue-500 cursor-pointer" />
                  <div className="flex-1">
                    <span className={`font-semibold flex items-center gap-2 ${textPrimary}`}><RefreshCw className={`w-4 h-4 ${compareMode ? "text-blue-400" : ""}`} /> Compare All</span>
                    <span className={`block text-sm ${textSecondary}`}>Run all 5 models together</span>
                  </div>
                  {compareMode && <span className="text-xs bg-blue-500/20 text-blue-300 px-2 py-1 rounded-full font-bold">5×</span>}
                </label>
              </>
            ) : (
              <div className={`p-4 rounded-xl ${darkMode ? "bg-emerald-500/10 border border-emerald-500/20" : "bg-emerald-50 border border-emerald-200"}`}>
                <div className="flex items-center gap-2 mb-2">
                  <Tag className="w-5 h-5 text-emerald-400" />
                  <span className="font-bold text-emerald-400">CNN Classifier</span>
                </div>
                <p className={`text-sm mb-3 ${textSecondary}`}>Classifies steel surface defects into 6 categories:</p>
                <div className="flex flex-wrap gap-2">
                  {NEU_CLASSES.map((cls) => (
                    <span key={cls} className={`text-xs px-2 py-1 rounded-lg ${subtleBg} ${darkMode ? "text-slate-300" : "text-slate-700"}`}>{cls}</span>
                  ))}
                </div>
                <div className={`mt-3 pt-3 border-t ${darkMode ? "border-emerald-500/20" : "border-emerald-200"} text-xs ${textMuted}`}>
                  <strong className={textSecondary}>Accuracy:</strong> 99% on NEU Surface Defect dataset
                </div>
              </div>
            )}
          </div>



          {/* Upload Card */}
          <div className={`p-6 rounded-2xl ${cardClass}`}>
            <div className="flex justify-between items-center mb-4">
              <h2 className={`text-lg font-bold flex items-center gap-2 ${textPrimary}`}>
                <Upload className="w-5 h-5 text-blue-400" /> Upload Images
              </h2>
              {uploadedImages.length > 0 && (
                <button onClick={() => { setUploadedImages([]); setResults([]); }} className="text-sm text-rose-400 flex items-center gap-1 hover:text-rose-300 cursor-pointer transition-colors">
                  <X className="w-4 h-4" /> Clear ({uploadedImages.length})
                </button>
              )}
            </div>
            
            <div {...getRootProps()} className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
              isDragActive 
                ? `${darkMode ? "border-blue-500 bg-blue-500/10" : "border-blue-400 bg-blue-50"} shadow-[inset_0_0_30px_rgba(59,130,246,0.1)]`
                : `${darkMode ? "border-white/[0.08] hover:border-white/[0.15]" : "border-slate-300 hover:border-slate-400"}`
            }`}>
              <input {...getInputProps()} />
              <div className={`w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center ${isDragActive ? "bg-blue-500/20" : subtleBg}`}>
                <Upload className={`w-8 h-8 ${isDragActive ? "text-blue-400" : textMuted}`} />
              </div>
              <p className={`font-medium ${textSecondary}`}>{isDragActive ? "Drop images here" : "Drag & drop images"}</p>
              <p className={`text-sm mt-1 ${textMuted}`}>or click to browse</p>
            </div>

            {/* Preview Grid */}
            <AnimatePresence>
              {uploadedImages.length > 0 && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="mt-4">
                  <div className={`flex items-center gap-2 mb-2 text-sm ${textSecondary}`}>
                    <ImageIcon className="w-4 h-4" />
                    {uploadedImages.length} image{uploadedImages.length > 1 ? "s" : ""} selected
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    {uploadedImages.map((img, i) => (
                      <motion.div key={i} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} className="relative group rounded-xl overflow-hidden aspect-square">
                        <img src={img.preview} alt="" className="w-full h-full object-cover" />
                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <button onClick={() => setUploadedImages(prev => prev.filter((_, idx) => idx !== i))} className="w-8 h-8 bg-rose-500 rounded-full flex items-center justify-center hover:bg-rose-600 cursor-pointer transition-colors">
                            <X className="w-4 h-4 text-white" />
                          </button>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Progress bar */}
            {loading && (
              <div className="my-4">
                <ProgressBar current={batchProgress.current} total={batchProgress.total} />
              </div>
            )}

            {/* Analyze Button */}
            <motion.button
              onClick={handleAnalyze}
              disabled={uploadedImages.length === 0 || loading}
              whileHover={uploadedImages.length > 0 && !loading ? { scale: 1.02, y: -2 } : {}}
              whileTap={uploadedImages.length > 0 && !loading ? { scale: 0.98 } : {}}
              className={`w-full mt-5 py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-3 transition-all cursor-pointer ${
                uploadedImages.length > 0 && !loading
                  ? useCNN
                    ? "bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-lg shadow-emerald-500/20 btn-shimmer"
                    : "bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 text-white shadow-lg shadow-blue-500/20 btn-shimmer"
                  : `${darkMode ? "bg-white/[0.04] text-slate-600" : "bg-slate-200 text-slate-400"} cursor-not-allowed`
              }`}
            >
              {loading ? (
                <><Spinner size="sm" /> <span className="ml-2">Processing...</span></>
              ) : (
                <><Search className="w-5 h-5" /> {useCNN ? "Classify" : "Analyze"} {uploadedImages.length} Image{uploadedImages.length > 1 ? "s" : ""} {!useCNN && compareMode && <span className="text-sm opacity-70">× 5</span>}</>
              )}
            </motion.button>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="mt-4">
                  <div className={`p-4 rounded-xl ${darkMode ? "bg-rose-500/10 border border-rose-500/20" : "bg-rose-50 border border-rose-200"}`}>
                    <div className="flex items-start gap-3">
                      <WifiOff className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="text-rose-400 font-medium">Connection Error</p>
                        <p className={`text-sm mt-1 ${darkMode ? "text-rose-300/70" : "text-rose-600"}`}>{error}</p>
                        <button onClick={handleAnalyze} className="mt-3 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 text-sm transition cursor-pointer">
                          <RefreshCw className="w-3 h-3" /> Retry
                        </button>
                      </div>
                      <button onClick={() => setError(null)} className="text-rose-400 hover:text-rose-300 p-1 cursor-pointer">
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* ===== RIGHT PANEL — RESULTS ===== */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="lg:col-span-3">
          <div className={`p-6 rounded-2xl min-h-[600px] ${cardClass}`}>
            <div className="flex items-center justify-between mb-6">
              <h2 className={`text-lg font-bold flex items-center gap-2 ${textPrimary}`}>
                <BarChart3 className="w-5 h-5 text-blue-400" /> Results
              </h2>
              {results.length > 0 && (
                <button onClick={downloadAllResults} className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 cursor-pointer transition-colors text-sm font-medium">
                  <Download className="w-4 h-4" /> Download All
                </button>
              )}
            </div>
            
            {/* Stats Summary */}
            <AnimatePresence>
              {stats && autoencoderResults.length > 0 && (
                <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-5 gap-3 mb-6">
                  <div className={`p-3 rounded-xl text-center ${subtleBg}`}><div className={`text-2xl font-bold ${textPrimary}`}>{stats.total}</div><div className={`text-xs ${textMuted}`}>Total</div></div>
                  <div className={`p-3 rounded-xl text-center ${darkMode ? "bg-emerald-500/10" : "bg-emerald-50"}`}><div className="text-2xl font-bold text-emerald-400">{stats.normal}</div><div className={`text-xs ${textMuted}`}>Normal</div></div>
                  <div className={`p-3 rounded-xl text-center ${darkMode ? "bg-amber-500/10" : "bg-amber-50"}`}><div className="text-2xl font-bold text-amber-400">{stats.suspicious}</div><div className={`text-xs ${textMuted}`}>Suspicious</div></div>
                  <div className={`p-3 rounded-xl text-center ${darkMode ? "bg-rose-500/10" : "bg-rose-50"}`}><div className="text-2xl font-bold text-rose-400">{stats.anomaly}</div><div className={`text-xs ${textMuted}`}>Anomaly</div></div>
                  <div className={`p-3 rounded-xl text-center ${subtleBg}`}><div className="text-2xl font-bold text-blue-400">{stats.avgTime.toFixed(1)}s</div><div className={`text-xs ${textMuted}`}>Avg Time</div></div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Results List */}
            <AnimatePresence mode="wait">
              {results.length > 0 ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                  {/* Classifier Results */}
                  {classifierResults.map((result, i) => (
                    <motion.div key={`cnn-${i}`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`p-4 rounded-xl card-3d-subtle ${darkMode ? "bg-emerald-500/10 border border-emerald-500/20" : "bg-emerald-50 border border-emerald-200"}`}>
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <span className="px-3 py-1 rounded-lg bg-emerald-500/20 font-bold text-sm text-emerald-400">CNN</span>
                          <span className={`text-sm ${textSecondary}`}>NEU Classifier</span>
                        </div>
                        <div className="text-right">
                          <div className="text-2xl font-black text-emerald-400 flex items-center gap-2"><Tag className="w-5 h-5" />{result.predicted_class}</div>
                          <div className="text-sm text-emerald-300">{(result.confidence * 100).toFixed(1)}% confidence</div>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <p className={`text-xs mb-1 ${textMuted}`}>Original</p>
                          <img src={`data:image/png;base64,${result.original_image}`} alt="Original" className="w-full rounded-lg" />
                        </div>
                        <div>
                          <p className={`text-xs mb-1 ${textMuted}`}>Class Probabilities</p>
                          <img src={`data:image/png;base64,${result.chart_image}`} alt="Chart" className="w-full rounded-lg" />
                        </div>
                      </div>
                      <p className={`text-xs mt-2 flex items-center gap-1 ${textMuted}`}><Clock className="w-3 h-3" /> {result.processing_time.toFixed(2)}s</p>
                    </motion.div>
                  ))}
                  
                  {/* Comparison View */}
                  {compareMode && groupedResults.length > 0 ? (
                    <div className="space-y-6">
                      {groupedResults.map((group, groupIdx) => (
                        <motion.div
                          key={`comparison-${groupIdx}`}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: groupIdx * 0.1 }}
                          className={`p-4 rounded-xl ${cardClass}`}
                        >
                          <div className={`flex items-center gap-3 mb-4 pb-3 border-b ${darkMode ? "border-white/[0.06]" : "border-slate-200"}`}>
                            <div className="w-16 h-16 rounded-lg overflow-hidden flex-shrink-0">
                              <img src={group.original} alt="Original" className="w-full h-full object-cover" />
                            </div>
                            <div>
                              <h3 className={`font-bold ${textPrimary}`}>Image {groupIdx + 1}</h3>
                              <p className={`text-sm ${textMuted}`}>{uploadedImages[groupIdx]?.file.name || "Uploaded image"}</p>
                            </div>
                          </div>
                          
                          <div className="grid grid-cols-3 gap-4 mb-4">
                            {AUTOENCODER_TYPES.map((modelType) => {
                              const result = group.models[modelType];
                              if (!result) return (
                                <div key={modelType} className={`p-3 rounded-lg ${subtleBg} text-center`}>
                                  <span className={`text-sm ${textMuted}`}>{modelType} - No result</span>
                                </div>
                              );
                              
                              const ScoreIcon = getScoreIcon(result.anomaly_score);
                              const modelColors: Record<string, string> = {
                                CAE: "from-blue-500/15 to-blue-500/5 border-blue-500/20",
                                VAE: "from-purple-500/15 to-purple-500/5 border-purple-500/20",
                                DAE: "from-orange-500/15 to-orange-500/5 border-orange-500/20",
                                SKIP_CAE: "from-teal-500/15 to-teal-500/5 border-teal-500/20",
                                PATCHCORE: "from-rose-500/15 to-rose-500/5 border-rose-500/20",
                              };
                              const labelColors: Record<string, string> = {
                                CAE: "bg-blue-500/20 text-blue-400",
                                VAE: "bg-purple-500/20 text-purple-400",
                                DAE: "bg-orange-500/20 text-orange-400",
                                SKIP_CAE: "bg-teal-500/20 text-teal-400",
                                PATCHCORE: "bg-rose-500/20 text-rose-400",
                              };
                              
                              return (
                                <div key={modelType} className={`p-3 rounded-xl bg-gradient-to-br ${modelColors[modelType]} border`}>
                                  <div className="flex items-center justify-between mb-2">
                                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${labelColors[modelType]}`}>{modelType}</span>
                                    <div className={`flex items-center gap-1 font-bold ${getScoreColor(result.anomaly_score)}`}>
                                      <ScoreIcon className="w-4 h-4" />
                                      <span>{result.anomaly_score.toFixed(3)}</span>
                                    </div>
                                  </div>
                                  
                                  <motion.div 
                                    whileHover={{ scale: 1.02 }} 
                                    className="cursor-pointer group relative rounded-lg overflow-hidden"
                                    onClick={() => downloadImage(result.heatmap, `heatmap_${modelType}_${groupIdx + 1}.png`)}
                                  >
                                    <img src={`data:image/png;base64,${result.heatmap}`} alt={`${modelType} Heatmap`} className="w-full rounded-lg" />
                                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                      <Download className="w-5 h-5 text-white" />
                                    </div>
                                  </motion.div>
                                  
                                  <div className={`mt-2 text-center text-xs font-medium ${getScoreColor(result.anomaly_score)}`}>
                                    {getScoreLabel(result.anomaly_score)}
                                  </div>
                                </div>
                              );
                            })}
                          </div>

                          {/* Detailed Cards */}
                          <div className={`pt-4 border-t ${darkMode ? "border-white/[0.06]" : "border-slate-200"}`}>
                            <p className={`text-xs font-medium mb-3 ${textMuted}`}>Detailed Results</p>
                            <div className="space-y-3">
                              {AUTOENCODER_TYPES.map((modelType) => {
                                const result = group.models[modelType];
                                if (!result) return null;
                                
                                const ScoreIcon = getScoreIcon(result.anomaly_score);
                                const resultKey = `${groupIdx}-${modelType}`;
                                const isExpanded = expandedResult === resultKey;
                                
                                return (
                                  <motion.div 
                                    key={resultKey} 
                                    className={`p-3 rounded-xl bg-gradient-to-br ${getScoreBg(result.anomaly_score)} border ${darkMode ? "border-white/[0.04]" : "border-slate-200"} cursor-pointer`}
                                    onClick={() => setExpandedResult(isExpanded ? null : resultKey)}
                                  >
                                    <div className="flex items-center justify-between mb-2">
                                      <div className="flex items-center gap-2">
                                        <span className={`px-2 py-0.5 rounded-lg font-bold text-xs ${subtleBg}`}>{result.model}</span>
                                        <span className={`text-xs ${textSecondary}`}>{result.category}</span>
                                      </div>
                                      <div className="flex items-center gap-2">
                                        <div className={`font-bold flex items-center gap-1 ${getScoreColor(result.anomaly_score)}`}>
                                          <ScoreIcon className="w-4 h-4" />
                                          <span className="text-sm">{result.anomaly_score.toFixed(3)}</span>
                                        </div>
                                        <ChevronDown className={`w-4 h-4 ${textMuted} transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                                      </div>
                                    </div>
                                    
                                    <AnimatePresence>
                                      {isExpanded && (
                                        <motion.div 
                                          initial={{ opacity: 0, height: 0 }} 
                                          animate={{ opacity: 1, height: "auto" }} 
                                          exit={{ opacity: 0, height: 0 }}
                                          className="pt-3"
                                        >
                                          <div className="grid grid-cols-3 gap-2">
                                            {[{ label: "Original", img: result.original_image }, { label: "Reconstruction", img: result.reconstruction }, { label: "Heatmap", img: result.heatmap }].map((item, j) => (
                                              <motion.div key={j} whileHover={{ scale: 1.03 }} className="cursor-pointer group relative" onClick={(e) => { e.stopPropagation(); downloadImage(item.img, `${item.label.toLowerCase()}_${result.model}.png`); }}>
                                                <p className={`text-xs mb-1 ${textMuted}`}>{item.label}</p>
                                                <div className="relative rounded-lg overflow-hidden">
                                                  <img src={`data:image/png;base64,${item.img}`} alt={item.label} className="w-full rounded-lg" />
                                                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"><Download className="w-4 h-4 text-white" /></div>
                                                </div>
                                              </motion.div>
                                            ))}
                                          </div>
                                          <div className={`mt-2 pt-2 border-t ${darkMode ? "border-white/[0.06]" : "border-slate-200"} flex justify-between text-xs ${textMuted}`}>
                                            <span><Clock className="w-3 h-3 inline mr-1" />{result.processing_time.toFixed(2)}s</span>
                                            <span><Layers className="w-3 h-3 inline mr-1" />{result.model}</span>
                                          </div>
                                        </motion.div>
                                      )}
                                    </AnimatePresence>
                                  </motion.div>
                                );
                              })}
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    /* Standard Results */
                    autoencoderResults.map((result, i) => {
                      const ScoreIcon = getScoreIcon(result.anomaly_score);
                      return (
                        <motion.div key={`ae-${i}`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }} className={`p-4 rounded-xl bg-gradient-to-br ${getScoreBg(result.anomaly_score)} border ${darkMode ? "border-white/[0.04]" : "border-slate-200"} cursor-pointer shadow-lg ${getScoreGlow(result.anomaly_score)} card-3d-subtle`} onClick={() => setExpandedResult(expandedResult === i ? null : i)}>
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-3">
                              <span className={`px-3 py-1 rounded-lg font-bold text-sm ${subtleBg}`}>{result.model}</span>
                              <span className={`text-sm ${textSecondary}`}>{result.category}</span>
                            </div>
                            <div className="flex items-center gap-3">
                              <div className="text-right">
                                <div className={`text-2xl font-black flex items-center gap-2 ${getScoreColor(result.anomaly_score)}`}><ScoreIcon className="w-5 h-5" />{result.anomaly_score.toFixed(3)}</div>
                                <div className={`text-sm ${getScoreColor(result.anomaly_score)}`}>{getScoreLabel(result.anomaly_score)}</div>
                              </div>
                              <ChevronDown className={`w-5 h-5 ${textMuted} transition-transform ${expandedResult === i ? "rotate-180" : ""}`} />
                            </div>
                          </div>
                          <div className="grid grid-cols-3 gap-3">
                            {[{ label: "Original", img: result.original_image }, { label: "Reconstruction", img: result.reconstruction }, { label: "Heatmap", img: result.heatmap }].map((item, j) => (
                              <motion.div key={j} whileHover={{ scale: 1.05 }} className="cursor-pointer group relative" onClick={(e) => { e.stopPropagation(); downloadImage(item.img, `${item.label.toLowerCase()}_${result.model}.png`); }}>
                                <p className={`text-xs mb-1 ${textMuted}`}>{item.label}</p>
                                <div className="relative rounded-lg overflow-hidden">
                                  <img src={`data:image/png;base64,${item.img}`} alt={item.label} className="w-full rounded-lg" />
                                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"><Download className="w-6 h-6 text-white" /></div>
                                </div>
                              </motion.div>
                            ))}
                          </div>
                          <AnimatePresence>
                            {expandedResult === i && (
                              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className={`mt-4 pt-4 border-t ${darkMode ? "border-white/[0.06]" : "border-slate-200"}`}>
                                <div className="grid grid-cols-3 gap-4 text-sm">
                                  <div><span className={textMuted}>Processing Time</span><div className={`font-bold flex items-center gap-1 ${textPrimary}`}><Clock className="w-4 h-4 text-blue-400" /> {result.processing_time.toFixed(2)}s</div></div>
                                  <div><span className={textMuted}>Model Type</span><div className={`font-bold flex items-center gap-1 ${textPrimary}`}><Layers className="w-4 h-4 text-purple-400" /> {result.model}</div></div>
                                  <div><span className={textMuted}>Category</span><div className={`font-bold ${textPrimary}`}>{result.category}</div></div>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      );
                    })
                  )}
                </motion.div>
              ) : (
                <div className="text-center py-20">
                  <div className={`w-20 h-20 rounded-2xl mx-auto mb-4 flex items-center justify-center ${subtleBg}`}>
                    <Search className={`w-10 h-10 ${textMuted}`} />
                  </div>
                  <p className={`font-medium ${textMuted}`}>No results yet</p>
                  <p className={`text-sm mt-1 ${textMuted}`}>Upload images and click {useCNN ? "Classify" : "Analyze"} to start</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* ===== 3RD COLUMN — UNDERSTANDING SCORES (always visible) ===== */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2"
        >
          <div className={`p-5 rounded-2xl ${cardClass} sticky top-6`}>
            <h2 className={`text-base font-bold mb-4 flex items-center gap-2 ${textPrimary}`}>
              <Info className="w-4 h-4 text-blue-400" /> Understanding Scores
            </h2>

            <div className="space-y-4 text-sm">
              {/* Score explanation */}
              <div className={`p-3 rounded-xl ${darkMode ? "bg-white/[0.03]" : "bg-slate-50"}`}>
                <p className={`font-semibold mb-1 ${textSecondary}`}>How is it calculated?</p>
                <p className={`text-xs leading-relaxed ${textMuted}`}>
                  The <strong className={textPrimary}>anomaly score</strong> is the{" "}
                  <strong className="text-blue-400">mean reconstruction error</strong> — how different the AI&apos;s reconstruction is from your original image.
                </p>
              </div>

              {/* Thresholds */}
              <div>
                <p className={`font-semibold mb-2 ${textSecondary}`}>Score Thresholds</p>
                <div className="space-y-2">
                  {[
                    { range: "0.0 – 0.3", label: "Normal", sub: "No defect detected", color: "emerald", Icon: CheckCircle },
                    { range: "0.3 – 0.6", label: "Suspicious", sub: "Possible anomaly", color: "amber", Icon: AlertTriangle },
                    { range: "0.6+", label: "Anomaly", sub: "Likely defect", color: "rose", Icon: AlertCircle },
                  ].map((t) => (
                    <div
                      key={t.range}
                      className={`flex items-start gap-2.5 p-2.5 rounded-xl ${
                        darkMode ? `bg-${t.color}-500/10 border border-${t.color}-500/20` : `bg-${t.color}-50 border border-${t.color}-100`
                      }`}
                    >
                      <t.Icon className={`w-4 h-4 text-${t.color}-400 mt-0.5 shrink-0`} />
                      <div>
                        <span className={`font-bold text-${t.color}-400 text-xs`}>{t.range}</span>
                        <p className={`font-semibold text-xs ${textSecondary}`}>{t.label}</p>
                        <p className={`text-xs ${textMuted}`}>{t.sub}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Heatmap tip */}
              <div className={`p-3 rounded-xl ${darkMode ? "bg-blue-500/10 border border-blue-500/20" : "bg-blue-50 border border-blue-200"}`}>
                <p className={`text-xs leading-relaxed ${textMuted}`}>
                  <strong className="text-blue-400">Heatmap Tip:</strong> Red/yellow areas show where the model detected differences from normal.
                </p>
              </div>

              {/* CNN note — only when CNN mode active */}
              {useCNN && (
                <div className={`p-3 rounded-xl ${darkMode ? "bg-emerald-500/10 border border-emerald-500/20" : "bg-emerald-50 border border-emerald-200"}`}>
                  <p className={`text-xs leading-relaxed ${textMuted}`}>
                    <strong className="text-emerald-400">CNN Mode:</strong> Outputs a defect class + confidence score instead of a heatmap.
                  </p>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
