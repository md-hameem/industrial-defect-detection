"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import Link from "next/link";
import { useRef } from "react";
import { useTheme } from "@/context/ThemeContext";
import { Search, BarChart3, Image as ImageIcon, Zap, RefreshCw, History, Upload, Settings, Database, Cpu, BarChart2, Brain, Award, Shuffle, Layers, CheckCircle, XCircle } from "lucide-react";

const features = [
  { icon: Search, title: "Real-time Detection", desc: "Upload any industrial image for instant defect analysis using trained AI models" },
  { icon: Brain, title: "Multiple AI Models", desc: "Choose from CAE, VAE, or Denoising Autoencoder based on your needs" },
  { icon: BarChart3, title: "Visual Heatmaps", desc: "See exactly where defects are located with color-coded anomaly maps" },
  { icon: Zap, title: "Fast Processing", desc: "CPU-optimized inference delivers results in seconds, not minutes" },
  { icon: RefreshCw, title: "Model Comparison", desc: "Run all three models simultaneously to compare detection results" },
  { icon: History, title: "History Tracking", desc: "Keep track of all your predictions with filtering and export options" },
];

const stats = [
  { value: "15", label: "MVTec Categories", icon: Layers, color: "blue" },
  { value: "99%", label: "CNN Accuracy", icon: Award, color: "emerald" },
  { value: "0.69", label: "Cross-dataset AUC", icon: Shuffle, color: "purple" },
  { value: "45+", label: "Trained Models", icon: Brain, color: "cyan" },
];

const workflow = [
  { step: 1, title: "Upload", desc: "Drag & drop your industrial image", icon: Upload },
  { step: 2, title: "Select Model", desc: "Choose CAE, VAE, or DAE", icon: Settings },
  { step: 3, title: "Analyze", desc: "AI processes your image in seconds", icon: Zap },
  { step: 4, title: "Review", desc: "View heatmap and anomaly score", icon: BarChart2 },
];

const testimonials = [
  { quote: "The reconstruction-based approach eliminates the need for labeled defect data.", author: "Research Finding", role: "Unsupervised Learning" },
  { quote: "Cross-dataset evaluation shows the models generalize to unseen industrial domains.", author: "Kolektor Testing", role: "0.69 ROC-AUC" },
  { quote: "The lightweight CNN achieves near-perfect accuracy on supervised classification.", author: "NEU Results", role: "99% Accuracy" },
];

const colorMap: Record<string, string> = {
  blue: "text-blue-400",
  emerald: "text-emerald-400",
  purple: "text-purple-400",
  cyan: "text-cyan-400",
};

const glowMap: Record<string, string> = {
  blue: "shadow-blue-500/20",
  emerald: "shadow-emerald-500/20",
  purple: "shadow-purple-500/20",
  cyan: "shadow-cyan-500/20",
};

export default function HomePage() {
  const { darkMode } = useTheme();
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ["start start", "end start"] });
  const heroY = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
  const heroOpacity = useTransform(scrollYProgress, [0, 1], [1, 0]);

  // Theme-aware classes
  const textPrimary = darkMode ? "text-white" : "text-slate-900";
  const textSecondary = darkMode ? "text-slate-400" : "text-slate-600";
  const textMuted = darkMode ? "text-slate-500" : "text-slate-400";
  const cardClass = darkMode
    ? "glass inner-glow"
    : "bg-white/80 border border-slate-200 shadow-sm";

  return (
    <div className="overflow-hidden">
      {/* ===== HERO ===== */}
      <section ref={heroRef} className="relative min-h-screen flex items-center justify-center py-20">
        <motion.div style={{ y: heroY, opacity: heroOpacity }} className="relative z-10 text-center max-w-5xl mx-auto px-6">
          {/* Floating Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-sm mb-8 cursor-default ${
              darkMode
                ? "bg-blue-500/10 border border-blue-500/20 text-blue-300"
                : "bg-blue-50 border border-blue-200 text-blue-600"
            }`}
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
            </span>
            AI-Powered Quality Control
          </motion.div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 200 }}
            className="text-5xl md:text-7xl lg:text-8xl font-black mb-6 leading-[0.95] tracking-tight"
          >
            <span className="text-gradient-animated">Industrial Defect</span>
            <br />
            <span className={textPrimary}>Detection System</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className={`text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed ${textSecondary}`}
          >
            Harness the power of{" "}
            <span className={`font-semibold ${darkMode ? "text-blue-400 text-glow-blue" : "text-blue-600"}`}>
              deep learning autoencoders
            </span>{" "}
            to detect manufacturing defects in seconds. No labeled data required.
          </motion.p>

          {/* CTA */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-wrap gap-4 justify-center mb-16"
          >
            <Link href="/detect">
              <motion.button
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.97 }}
                className="px-10 py-5 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 text-white rounded-2xl font-bold text-lg shadow-2xl shadow-blue-500/25 flex items-center gap-3 btn-shimmer cursor-pointer"
              >
                <Search className="w-6 h-6" /> Start Detection
                <motion.span animate={{ x: [0, 5, 0] }} transition={{ duration: 1.5, repeat: Infinity }}>→</motion.span>
              </motion.button>
            </Link>
            <Link href="/research">
              <motion.button
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.97 }}
                className={`px-10 py-5 rounded-2xl font-bold text-lg flex items-center gap-3 cursor-pointer ${
                  darkMode
                    ? "glass text-white hover:bg-white/[0.08]"
                    : "bg-slate-100 border border-slate-200 text-slate-900 hover:bg-slate-200"
                }`}
              >
                <BarChart3 className="w-6 h-6" /> View Research
              </motion.button>
            </Link>
          </motion.div>

          {/* Demo Preview — 3D Glass Card */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, type: "spring", stiffness: 150 }}
            className="relative mx-auto max-w-4xl card-3d"
          >
            {/* Glow behind card */}
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-cyan-500/20 rounded-3xl blur-3xl opacity-50" />

            <div className={`relative rounded-3xl p-6 shadow-2xl ${
              darkMode
                ? "glass-strong"
                : "bg-white/90 border border-slate-200 shadow-xl"
            }`}>
              {/* Window dots */}
              <div className="flex gap-2 mb-5">
                <div className="w-3 h-3 rounded-full bg-rose-500/80" />
                <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
                <div className={`ml-4 flex-1 h-3 rounded-full ${darkMode ? "bg-white/[0.04]" : "bg-slate-100"}`} />
              </div>

              <div className="grid grid-cols-3 gap-4">
                {[
                  { icon: ImageIcon, label: "Original", color: "text-blue-400" },
                  { icon: RefreshCw, label: "Reconstruction", color: "text-purple-400" },
                  { icon: BarChart2, label: "Heatmap", color: "text-orange-400" },
                ].map((item, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 + i * 0.1 }}
                    className={`rounded-2xl p-6 text-center card-3d-subtle ${
                      darkMode ? "bg-white/[0.03] border border-white/[0.06]" : "bg-slate-50 border border-slate-200"
                    }`}
                  >
                    <item.icon className={`w-10 h-10 mx-auto mb-3 ${item.color}`} />
                    <div className={`text-sm font-medium ${textSecondary}`}>{item.label}</div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className={`flex flex-col items-center gap-2 ${textMuted}`}
          >
            <span className="text-xs tracking-widest uppercase">Scroll</span>
            <div className={`w-5 h-8 rounded-full border-2 flex justify-center pt-1.5 ${darkMode ? "border-white/20" : "border-slate-300"}`}>
              <motion.div
                animate={{ y: [0, 8, 0], opacity: [1, 0, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                className={`w-1 h-1 rounded-full ${darkMode ? "bg-white/40" : "bg-slate-400"}`}
              />
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* ===== STATS ===== */}
      <section className="py-20 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="grid grid-cols-2 md:grid-cols-4 gap-5"
          >
            {stats.map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ y: -8, scale: 1.02 }}
                className={`p-8 rounded-3xl text-center card-3d-subtle cursor-default ${cardClass}`}
              >
                <div className={`w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center ${
                  darkMode ? "bg-white/[0.05]" : "bg-slate-100"
                }`}>
                  <stat.icon className={`w-7 h-7 ${colorMap[stat.color]}`} />
                </div>
                <div className={`text-5xl font-black mb-2 ${colorMap[stat.color]}`}>
                  {stat.value}
                </div>
                <div className={`text-sm ${textSecondary}`}>{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section className="py-20 px-6 relative">
        <div className="max-w-7xl mx-auto relative">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="text-gradient-animated">How It Works</span>
            </h2>
            <p className={`text-lg ${textSecondary}`}>Simple 4-step process to detect defects</p>
          </motion.div>

          <div className="grid md:grid-cols-4 gap-6">
            {workflow.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.12 }}
                className="relative"
              >
                {/* Connecting line */}
                {i < workflow.length - 1 && (
                  <div className={`hidden md:block absolute top-14 left-full w-full h-px ${
                    darkMode
                      ? "bg-gradient-to-r from-blue-500/30 to-transparent"
                      : "bg-gradient-to-r from-blue-300/50 to-transparent"
                  }`} />
                )}

                <motion.div
                  whileHover={{ y: -6, scale: 1.02 }}
                  className={`p-6 rounded-2xl text-center relative card-3d-subtle cursor-default ${cardClass}`}
                >
                  {/* Step number */}
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                    <div className="relative">
                      <div className="absolute inset-0 bg-blue-500/30 rounded-full blur-lg" />
                      <div className="relative w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-sm font-bold text-white shadow-lg">
                        {item.step}
                      </div>
                    </div>
                  </div>

                  <item.icon className="w-10 h-10 mx-auto mb-4 mt-6 text-blue-400" />
                  <h3 className={`text-lg font-bold mb-2 ${textPrimary}`}>{item.title}</h3>
                  <p className={`text-sm ${textSecondary}`}>{item.desc}</p>
                </motion.div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FEATURES GRID ===== */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="text-gradient-animated">Powerful Features</span>
            </h2>
            <p className={`text-lg ${textSecondary}`}>Everything you need for industrial quality control</p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.06 }}
                whileHover={{ y: -6 }}
                className={`p-8 rounded-2xl group card-3d-subtle cursor-default ${cardClass}`}
              >
                <div className={`w-14 h-14 rounded-2xl mb-5 flex items-center justify-center transition-transform group-hover:scale-110 ${
                  darkMode ? "bg-blue-500/10" : "bg-blue-50"
                }`}>
                  <feature.icon className="w-7 h-7 text-blue-400" />
                </div>
                <h3 className={`text-xl font-bold mb-3 ${textPrimary}`}>{feature.title}</h3>
                <p className={`leading-relaxed ${textSecondary}`}>{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== AI MODELS ===== */}
      <section className="py-20 px-6 relative">
        <div className="max-w-7xl mx-auto relative">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="text-gradient-animated">AI Models</span>
            </h2>
            <p className={`text-lg ${textSecondary}`}>Five specialized architectures for different use cases</p>
          </motion.div>

          <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-5">
            {[
              { name: "CAE", full: "Convolutional Autoencoder", score: "0.62 AUC", gradient: "from-blue-500 to-cyan-500", glow: "blue" },
              { name: "VAE", full: "Variational Autoencoder", score: "0.53 AUC", gradient: "from-purple-500 to-pink-500", glow: "purple" },
              { name: "DAE", full: "Denoising Autoencoder", score: "0.62 AUC", gradient: "from-orange-500 to-red-500", glow: "orange" },
              { name: "Skip-CAE", full: "U-Net Style AE", score: "Better Maps", gradient: "from-teal-500 to-cyan-500", glow: "teal" },
              { name: "PatchCore", full: "Feature-Based (SOTA)", score: "0.85+ AUC", gradient: "from-rose-500 to-pink-600", glow: "rose" },
            ].map((model, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ y: -8, scale: 1.03 }}
                className={`p-6 rounded-3xl text-center relative overflow-hidden group card-3d-subtle cursor-default ${cardClass}`}
              >
                {/* Hover glow */}
                <div className={`absolute inset-0 bg-gradient-to-br ${model.gradient} opacity-0 group-hover:opacity-[0.08] transition-opacity duration-500`} />

                <div className={`inline-block px-4 py-2 rounded-xl bg-gradient-to-r ${model.gradient} text-white font-black text-xl mb-4 shadow-lg`}>
                  {model.name}
                </div>
                <h3 className={`text-sm font-semibold mb-1 relative z-10 ${textPrimary}`}>{model.full}</h3>
                <div className={`text-xl font-bold mt-3 relative z-10 ${textPrimary}`}>{model.score}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== RESEARCH INSIGHTS ===== */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="text-gradient-animated">Research Insights</span>
            </h2>
          </motion.div>

          <div className="space-y-5">
            {testimonials.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: i % 2 === 0 ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className={`p-8 rounded-2xl card-3d-subtle cursor-default ${cardClass}`}
              >
                <p className={`text-xl mb-5 leading-relaxed ${darkMode ? "text-slate-300" : "text-slate-700"}`}>
                  &ldquo;{item.quote}&rdquo;
                </p>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                    <BarChart3 className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className={`font-bold ${textPrimary}`}>{item.author}</div>
                    <div className={`text-sm ${textSecondary}`}>{item.role}</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FINAL CTA ===== */}
      <section className="py-20 px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto"
        >
          <div className={`relative p-12 md:p-16 rounded-3xl overflow-hidden text-center ${
            darkMode ? "glass-strong" : "bg-white border border-slate-200 shadow-xl"
          }`}>
            {/* Background glow */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 rounded-full blur-3xl" />
            </div>

            <div className="relative z-10">
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              >
                <Cpu className={`w-16 h-16 mx-auto mb-6 ${darkMode ? "text-blue-400" : "text-blue-500"}`} />
              </motion.div>
              <h2 className={`text-4xl md:text-5xl font-bold mb-6 ${textPrimary}`}>Ready to detect defects?</h2>
              <p className={`text-xl mb-10 max-w-2xl mx-auto ${textSecondary}`}>
                Upload your industrial images and get instant AI-powered analysis with visual heatmaps
              </p>
              <Link href="/detect">
                <motion.button
                  whileHover={{ scale: 1.05, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-12 py-5 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 text-white rounded-2xl font-bold text-xl shadow-2xl shadow-blue-500/25 btn-shimmer cursor-pointer"
                >
                  Get Started Free →
                </motion.button>
              </Link>
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
}
