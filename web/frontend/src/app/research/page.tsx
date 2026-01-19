"use client";

import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { useState } from "react";
import { BarChart3, TrendingUp, Shuffle, Brain, CheckCircle, XCircle, Award, Layers } from "lucide-react";

// Full CSV data embedded
const mvtecResults = {
  CAE: [
    { category: "bottle", auc: 0.55 },
    { category: "cable", auc: 0.458 },
    { category: "capsule", auc: 0.477 },
    { category: "carpet", auc: 0.330 },
    { category: "grid", auc: 0.779 },
    { category: "hazelnut", auc: 0.877 },
    { category: "leather", auc: 0.447 },
    { category: "metal_nut", auc: 0.268 },
    { category: "pill", auc: 0.751 },
    { category: "screw", auc: 0.979 },
    { category: "tile", auc: 0.822 },
    { category: "toothbrush", auc: 0.656 },
    { category: "transistor", auc: 0.403 },
    { category: "wood", auc: 0.948 },
    { category: "zipper", auc: 0.506 },
  ],
  VAE: [
    { category: "bottle", auc: 0.457 },
    { category: "cable", auc: 0.403 },
    { category: "capsule", auc: 0.590 },
    { category: "carpet", auc: 0.250 },
    { category: "grid", auc: 0.500 },
    { category: "hazelnut", auc: 0.326 },
    { category: "leather", auc: 0.238 },
    { category: "metal_nut", auc: 0.673 },
    { category: "pill", auc: 0.408 },
    { category: "screw", auc: 1.000 },
    { category: "tile", auc: 0.388 },
    { category: "toothbrush", auc: 0.711 },
    { category: "transistor", auc: 0.400 },
    { category: "wood", auc: 0.360 },
    { category: "zipper", auc: 0.441 },
  ],
  DAE: [
    { category: "bottle", auc: 0.537 },
    { category: "cable", auc: 0.464 },
    { category: "capsule", auc: 0.466 },
    { category: "carpet", auc: 0.332 },
    { category: "grid", auc: 0.870 },
    { category: "hazelnut", auc: 0.888 },
    { category: "leather", auc: 0.389 },
    { category: "metal_nut", auc: 0.268 },
    { category: "pill", auc: 0.762 },
    { category: "screw", auc: 0.986 },
    { category: "tile", auc: 0.808 },
    { category: "toothbrush", auc: 0.650 },
    { category: "transistor", auc: 0.445 },
    { category: "wood", auc: 0.962 },
    { category: "zipper", auc: 0.487 },
  ],
};

const crossDatasetResults = [
  { trained: "bottle", cae: 0.637, dae: 0.609, vae: null },
  { trained: "carpet", cae: 0.665, dae: 0.682, vae: 0.536 },
  { trained: "grid", cae: 0.690, dae: 0.688, vae: null },
  { trained: "leather", cae: 0.668, dae: 0.646, vae: 0.532 },
  { trained: "metal_nut", cae: 0.622, dae: 0.617, vae: null },
  { trained: "tile", cae: 0.649, dae: 0.575, vae: 0.532 },
  { trained: "wood", cae: 0.662, dae: 0.652, vae: null },
];

const figures = [
  { src: "/figures/thesis_fig1_datasets.png", title: "Datasets Overview", desc: "MVTec AD, KolektorSDD2, and NEU Surface Defect samples" },
  { src: "/figures/thesis_fig2_model_comparison.png", title: "Model Comparison", desc: "CAE vs DAE performance across MVTec categories" },
  { src: "/figures/thesis_fig3_generalization.png", title: "Cross-Dataset Generalization", desc: "MVTec to Kolektor transfer learning heatmap" },
  { src: "/figures/thesis_fig4_reconstructions.png", title: "Reconstruction Examples", desc: "Original, reconstruction, error map, and ground truth" },
];

const getAucColor = (auc: number) => {
  if (auc >= 0.8) return "text-emerald-400 bg-emerald-500/20";
  if (auc >= 0.6) return "text-amber-400 bg-amber-500/20";
  return "text-rose-400 bg-rose-500/20";
};

const getAucBg = (auc: number) => {
  if (auc >= 0.8) return "bg-emerald-500";
  if (auc >= 0.6) return "bg-amber-500";
  return "bg-rose-500";
};

export default function ResearchPage() {
  const [selectedModel, setSelectedModel] = useState<"CAE" | "VAE" | "DAE">("CAE");
  const [selectedFigure, setSelectedFigure] = useState<number | null>(null);

  const modelData = mvtecResults[selectedModel];
  const avgAuc = (modelData.reduce((sum, r) => sum + r.auc, 0) / modelData.length).toFixed(3);
  const bestCategory = modelData.reduce((best, r) => r.auc > best.auc ? r : best);
  const worstCategory = modelData.reduce((worst, r) => r.auc < worst.auc ? r : worst);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-12">
        <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
          <BarChart3 className="w-10 h-10 text-blue-400" />
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Research Results</span>
        </h1>
        <p className="text-xl text-slate-400">Complete experimental results from the thesis research</p>
      </motion.div>

      {/* Key Metrics */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { value: "15", label: "MVTec Categories", icon: Layers, color: "blue" },
            { value: "45+", label: "Trained Models", icon: Brain, color: "purple" },
            { value: "99%", label: "CNN Accuracy", icon: Award, color: "emerald" },
            { value: "0.69", label: "Best Cross-dataset", icon: Shuffle, color: "orange" },
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.1 }}
              whileHover={{ scale: 1.03 }}
              className="p-6 rounded-2xl bg-slate-800/50 border border-white/10 text-center"
            >
              <stat.icon className={`w-8 h-8 mx-auto mb-2 text-${stat.color}-400`} />
              <div className={`text-3xl font-black text-${stat.color}-400`}>{stat.value}</div>
              <div className="text-sm text-slate-400">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Figures Gallery */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="mb-12">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-blue-400" /> Thesis Figures
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          {figures.map((fig, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.1 }}
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelectedFigure(i)}
              className="cursor-pointer rounded-2xl overflow-hidden bg-slate-800/50 border border-white/10 group"
            >
              <div className="relative h-48 overflow-hidden">
                <Image src={fig.src} alt={fig.title} fill className="object-contain bg-slate-900 group-hover:scale-105 transition-transform" />
              </div>
              <div className="p-4">
                <h3 className="font-bold mb-1">{fig.title}</h3>
                <p className="text-sm text-slate-400">{fig.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Figure Modal */}
      <AnimatePresence>
        {selectedFigure !== null && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedFigure(null)}
            className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-8 cursor-pointer"
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="relative max-w-5xl w-full"
            >
              <Image src={figures[selectedFigure].src} alt={figures[selectedFigure].title} width={1200} height={800} className="rounded-xl" />
              <div className="mt-4 text-center">
                <h3 className="text-xl font-bold">{figures[selectedFigure].title}</h3>
                <p className="text-slate-400">{figures[selectedFigure].desc}</p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Model Performance Table */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mb-12">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-blue-400" /> MVTec AD Performance
          </h2>
          <div className="flex gap-2">
            {(["CAE", "VAE", "DAE"] as const).map((model) => (
              <button
                key={model}
                onClick={() => setSelectedModel(model)}
                className={`px-4 py-2 rounded-lg font-medium transition ${
                  selectedModel === model
                    ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white"
                    : "bg-slate-800/50 text-slate-400 hover:bg-slate-700"
                }`}
              >
                {model}
              </button>
            ))}
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="p-4 rounded-xl bg-slate-800/50 border border-white/5 text-center">
            <div className="text-2xl font-bold text-blue-400">{avgAuc}</div>
            <div className="text-sm text-slate-400">Mean AUC</div>
          </div>
          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center flex items-center justify-center gap-2">
            <CheckCircle className="w-5 h-5 text-emerald-400" />
            <div>
              <div className="text-lg font-bold text-emerald-400">{bestCategory.category}</div>
              <div className="text-xs text-slate-400">Best ({bestCategory.auc.toFixed(2)})</div>
            </div>
          </div>
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-center flex items-center justify-center gap-2">
            <XCircle className="w-5 h-5 text-rose-400" />
            <div>
              <div className="text-lg font-bold text-rose-400">{worstCategory.category}</div>
              <div className="text-xs text-slate-400">Worst ({worstCategory.auc.toFixed(2)})</div>
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="rounded-2xl overflow-hidden border border-white/10">
          <table className="w-full">
            <thead className="bg-slate-800/80">
              <tr>
                <th className="p-4 text-left text-slate-400">Category</th>
                <th className="p-4 text-center text-slate-400">ROC-AUC</th>
                <th className="p-4 text-left text-slate-400">Performance</th>
              </tr>
            </thead>
            <tbody>
              <AnimatePresence mode="wait">
                {modelData.map((row, i) => (
                  <motion.tr
                    key={`${selectedModel}-${row.category}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    transition={{ delay: i * 0.02 }}
                    className="border-t border-white/5 hover:bg-white/5"
                  >
                    <td className="p-4 font-medium">{row.category}</td>
                    <td className="p-4 text-center">
                      <span className={`px-3 py-1 rounded-lg text-sm font-bold ${getAucColor(row.auc)}`}>
                        {row.auc.toFixed(3)}
                      </span>
                    </td>
                    <td className="p-4">
                      <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${row.auc * 100}%` }}
                          transition={{ duration: 0.5, delay: i * 0.02 }}
                          className={`h-full ${getAucBg(row.auc)}`}
                        />
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        </div>
      </motion.section>

      {/* Cross-Dataset Table */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mb-12">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Shuffle className="w-6 h-6 text-blue-400" /> Cross-Dataset Evaluation (MVTec to Kolektor)
        </h2>
        <div className="rounded-2xl overflow-hidden border border-white/10">
          <table className="w-full">
            <thead className="bg-slate-800/80">
              <tr>
                <th className="p-4 text-left text-slate-400">Trained On</th>
                <th className="p-4 text-center text-blue-400">CAE</th>
                <th className="p-4 text-center text-orange-400">DAE</th>
                <th className="p-4 text-center text-purple-400">VAE</th>
              </tr>
            </thead>
            <tbody>
              {crossDatasetResults.map((row, i) => (
                <motion.tr
                  key={row.trained}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6 + i * 0.05 }}
                  className="border-t border-white/5 hover:bg-white/5"
                >
                  <td className="p-4 font-medium">{row.trained}</td>
                  <td className="p-4 text-center">
                    <span className={`px-3 py-1 rounded-lg text-sm font-bold ${getAucColor(row.cae)}`}>{row.cae.toFixed(3)}</span>
                  </td>
                  <td className="p-4 text-center">
                    <span className={`px-3 py-1 rounded-lg text-sm font-bold ${getAucColor(row.dae)}`}>{row.dae.toFixed(3)}</span>
                  </td>
                  <td className="p-4 text-center">
                    {row.vae ? <span className={`px-3 py-1 rounded-lg text-sm font-bold ${getAucColor(row.vae)}`}>{row.vae.toFixed(3)}</span> : <span className="text-slate-500">-</span>}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.section>

      {/* Model Comparison Cards */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}>
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Brain className="w-6 h-6 text-blue-400" /> Model Architecture Comparison
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              name: "CAE",
              full: "Convolutional Autoencoder",
              meanAuc: 0.617,
              color: "from-blue-500 to-cyan-500",
              pros: ["Simple architecture", "Stable training", "Best generalization"],
              cons: ["No probabilistic latent space"],
              bestOn: "Screw (0.98)",
            },
            {
              name: "VAE",
              full: "Variational Autoencoder",
              meanAuc: 0.476,
              color: "from-purple-500 to-pink-500",
              pros: ["Probabilistic encoding", "Latent space sampling"],
              cons: ["KL divergence instability", "Lower AUC"],
              bestOn: "Screw (1.00)*",
            },
            {
              name: "DAE",
              full: "Denoising Autoencoder",
              meanAuc: 0.621,
              color: "from-orange-500 to-red-500",
              pros: ["Robust to noise", "Good on textures", "Best mean AUC"],
              cons: ["Requires noise tuning"],
              bestOn: "Screw (0.99)",
            },
          ].map((model, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 + i * 0.1 }}
              whileHover={{ scale: 1.02 }}
              className="p-6 rounded-2xl bg-slate-800/30 border border-white/10 relative overflow-hidden group"
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${model.color} opacity-0 group-hover:opacity-10 transition-opacity`} />
              <div className="relative z-10">
                <div className={`inline-block px-4 py-2 rounded-xl bg-gradient-to-r ${model.color} text-white font-black text-xl mb-3`}>
                  {model.name}
                </div>
                <h3 className="font-semibold mb-1">{model.full}</h3>
                <div className="text-3xl font-black mb-4">{model.meanAuc.toFixed(3)}</div>
                <div className="space-y-2 text-sm">
                  <div className="text-emerald-400 flex items-start gap-1"><CheckCircle className="w-4 h-4 mt-0.5 shrink-0" /> {model.pros.join(" | ")}</div>
                  <div className="text-rose-400 flex items-start gap-1"><XCircle className="w-4 h-4 mt-0.5 shrink-0" /> {model.cons.join(" | ")}</div>
                  <div className="text-blue-400 flex items-center gap-1"><Award className="w-4 h-4" /> {model.bestOn}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
        <p className="text-sm text-slate-500 mt-4">* VAE screw result may be an outlier due to training instability</p>
      </motion.section>
    </div>
  );
}
