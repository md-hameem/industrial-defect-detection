"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTheme } from "@/context/ThemeContext";
import { Download, Trash2, History, Filter, Calendar } from "lucide-react";

interface HistoryItem {
  id: string;
  model: string;
  category: string;
  anomaly_score: number;
  heatmap: string;
  timestamp: string;
  filename: string;
}

export default function HistoryPage() {
  const { darkMode } = useTheme();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    const saved = localStorage.getItem("defect_history");
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("defect_history");
  };

  const downloadImage = (base64: string, filename: string) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${base64}`;
    link.download = filename;
    link.click();
  };

  const getScoreColor = (s: number) => s < 0.3 ? "text-emerald-400" : s < 0.6 ? "text-amber-400" : "text-rose-400";
  const getScoreLabel = (s: number) => s < 0.3 ? "Normal" : s < 0.6 ? "Suspicious" : "Anomaly";

  const filteredHistory = history.filter(item => {
    if (filter === "all") return true;
    if (filter === "normal") return item.anomaly_score < 0.3;
    if (filter === "suspicious") return item.anomaly_score >= 0.3 && item.anomaly_score < 0.6;
    if (filter === "anomaly") return item.anomaly_score >= 0.6;
    return item.model === filter;
  });

  // Theme-aware classes
  const cardBg = darkMode ? "bg-slate-800/50 border-white/5" : "bg-white/80 border-slate-200 shadow-sm";
  const cardHover = darkMode ? "hover:bg-slate-800/70" : "hover:bg-slate-50";
  const textPrimary = darkMode ? "text-white" : "text-slate-900";
  const textSecondary = darkMode ? "text-slate-400" : "text-slate-600";
  const textMuted = darkMode ? "text-slate-500" : "text-slate-400";
  const filterBg = darkMode ? "bg-slate-800/50 text-slate-400 hover:bg-slate-700" : "bg-slate-100 text-slate-600 hover:bg-slate-200";
  const filterActive = "bg-blue-500 text-white";

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <History className="w-8 h-8 text-blue-400" />
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Prediction History</span>
        </h1>
        {history.length > 0 && (
          <button onClick={clearHistory} className="flex items-center gap-2 px-4 py-2 bg-rose-500/20 text-rose-400 rounded-lg hover:bg-rose-500/30 transition">
            <Trash2 className="w-4 h-4" /> Clear All
          </button>
        )}
      </motion.div>

      {/* Filters */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 mb-6 flex-wrap">
        <Filter className={`w-4 h-4 ${textMuted}`} />
        {["all", "CAE", "VAE", "DAE", "CNN", "normal", "suspicious", "anomaly"].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-2 rounded-lg transition ${filter === f ? filterActive : filterBg}`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </motion.div>

      {/* Stats */}
      {history.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-4 gap-4 mb-8">
          {[
            { label: "Total", value: history.length, color: "blue" },
            { label: "Normal", value: history.filter(h => h.anomaly_score < 0.3).length, color: "emerald" },
            { label: "Suspicious", value: history.filter(h => h.anomaly_score >= 0.3 && h.anomaly_score < 0.6).length, color: "amber" },
            { label: "Anomalies", value: history.filter(h => h.anomaly_score >= 0.6).length, color: "rose" },
          ].map((stat, i) => (
            <div key={i} className={`p-4 rounded-xl border text-center ${cardBg}`}>
              <div className={`text-2xl font-bold text-${stat.color}-400`}>{stat.value}</div>
              <div className={`text-sm ${textSecondary}`}>{stat.label}</div>
            </div>
          ))}
        </motion.div>
      )}

      {/* History List */}
      <AnimatePresence>
        {filteredHistory.length > 0 ? (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
            {filteredHistory.map((item, i) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className={`p-4 rounded-xl border flex items-center gap-4 transition ${cardBg} ${cardHover}`}
              >
                <img src={`data:image/png;base64,${item.heatmap}`} alt="" className="w-20 h-20 rounded-lg object-cover" />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`font-bold ${textPrimary}`}>{item.model}</span>
                    <span className={textMuted}>|</span>
                    <span className={textSecondary}>{item.category}</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${getScoreColor(item.anomaly_score)}`}>
                      {getScoreLabel(item.anomaly_score)}
                    </span>
                  </div>
                  <div className={`text-lg font-bold ${getScoreColor(item.anomaly_score)}`}>
                    Score: {item.anomaly_score.toFixed(4)}
                  </div>
                  <div className={`text-xs flex items-center gap-1 ${textMuted}`}>
                    <Calendar className="w-3 h-3" /> {new Date(item.timestamp).toLocaleString()}
                  </div>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => downloadImage(item.heatmap, `heatmap_${item.id}.png`)}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg"
                >
                  <Download className="w-4 h-4" /> Download
                </motion.button>
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={`text-center py-20 ${textMuted}`}>
            <History className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p>{history.length === 0 ? "No history yet. Start detecting!" : "No results match the filter."}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
