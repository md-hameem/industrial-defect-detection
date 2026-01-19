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
  { value: "15", label: "MVTec Categories", icon: Layers },
  { value: "99%", label: "CNN Accuracy", icon: Award },
  { value: "0.69", label: "Cross-dataset AUC", icon: Shuffle },
  { value: "45+", label: "Trained Models", icon: Brain },
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

export default function HomePage() {
  const { darkMode } = useTheme();
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ["start start", "end start"] });
  const heroY = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
  const heroOpacity = useTransform(scrollYProgress, [0, 1], [1, 0]);

  // Theme-aware classes
  const textPrimary = darkMode ? "text-white" : "text-slate-900";
  const textSecondary = darkMode ? "text-slate-400" : "text-slate-600";
  const cardBg = darkMode ? "bg-slate-800/30 border-white/5" : "bg-white/80 border-slate-200 shadow-sm";
  const demoBg = darkMode ? "bg-slate-900/80 border-white/10" : "bg-white/90 border-slate-200";
  const demoItemBg = darkMode ? "bg-slate-800/50" : "bg-slate-100";
  const secondaryBtnBg = darkMode ? "bg-white/5 border-white/10 text-white hover:bg-white/10" : "bg-slate-100 border-slate-200 text-slate-900 hover:bg-slate-200";

  return (
    <div className="overflow-hidden">
      {/* Hero Section with Parallax */}
      <section ref={heroRef} className="relative min-h-screen flex items-center justify-center py-20">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div animate={{ rotate: 360 }} transition={{ duration: 50, repeat: Infinity, ease: "linear" }} className="absolute -top-1/2 -left-1/2 w-full h-full">
            <div className={`absolute top-1/2 left-1/2 w-96 h-96 rounded-full blur-3xl ${darkMode ? "bg-blue-500/10" : "bg-blue-300/20"}`} />
          </motion.div>
          <motion.div animate={{ rotate: -360 }} transition={{ duration: 70, repeat: Infinity, ease: "linear" }} className="absolute -bottom-1/2 -right-1/2 w-full h-full">
            <div className={`absolute top-1/2 left-1/2 w-96 h-96 rounded-full blur-3xl ${darkMode ? "bg-purple-500/10" : "bg-purple-300/20"}`} />
          </motion.div>
        </div>

        <motion.div style={{ y: heroY, opacity: heroOpacity }} className="relative z-10 text-center max-w-5xl mx-auto px-6">
          {/* Floating Badge */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm mb-8">
            <span className="animate-pulse w-2 h-2 rounded-full bg-red-500" /> AI-Powered Quality Control
          </motion.div>

          {/* Main Title */}
          <motion.h1 initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="text-5xl md:text-7xl font-black mb-6 leading-tight">
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">Industrial Defect</span>
            <br />
            <span className={textPrimary}>Detection System</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className={`text-xl md:text-2xl max-w-3xl mx-auto mb-10 leading-relaxed ${textSecondary}`}>
            Harness the power of <span className="text-blue-400 font-semibold">deep learning autoencoders</span> to detect manufacturing defects in seconds. No labeled data required.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="flex flex-wrap gap-4 justify-center mb-16">
            <Link href="/detect">
              <motion.button whileHover={{ scale: 1.05, boxShadow: "0 0 60px rgba(99, 102, 241, 0.4)" }} whileTap={{ scale: 0.95 }} className="px-10 py-5 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 text-white rounded-2xl font-bold text-lg shadow-2xl flex items-center gap-3">
                <Search className="w-6 h-6" /> Start Detection
                <motion.span animate={{ x: [0, 5, 0] }} transition={{ duration: 1.5, repeat: Infinity }}>→</motion.span>
              </motion.button>
            </Link>
            <Link href="/research">
              <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className={`px-10 py-5 border rounded-2xl font-bold text-lg backdrop-blur flex items-center gap-3 ${secondaryBtnBg}`}>
                <BarChart3 className="w-6 h-6" /> View Research
              </motion.button>
            </Link>
          </motion.div>

          {/* Demo Preview */}
          <motion.div initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="relative mx-auto max-w-4xl">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl blur-2xl opacity-20" />
            <div className={`relative backdrop-blur-xl border rounded-3xl p-6 shadow-2xl ${demoBg}`}>
              <div className="flex gap-2 mb-4">
                <div className="w-3 h-3 rounded-full bg-rose-500" />
                <div className="w-3 h-3 rounded-full bg-amber-500" />
                <div className="w-3 h-3 rounded-full bg-emerald-500" />
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className={`rounded-xl p-4 text-center ${demoItemBg}`}>
                  <ImageIcon className="w-10 h-10 mx-auto mb-2 text-blue-400" />
                  <div className={`text-sm ${textSecondary}`}>Original</div>
                </div>
                <div className={`rounded-xl p-4 text-center ${demoItemBg}`}>
                  <RefreshCw className="w-10 h-10 mx-auto mb-2 text-purple-400" />
                  <div className={`text-sm ${textSecondary}`}>Reconstruction</div>
                </div>
                <div className={`rounded-xl p-4 text-center ${demoItemBg}`}>
                  <BarChart2 className="w-10 h-10 mx-auto mb-2 text-orange-400" />
                  <div className={`text-sm ${textSecondary}`}>Heatmap</div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }} className="absolute bottom-8 left-1/2 -translate-x-1/2">
          <motion.div animate={{ y: [0, 10, 0] }} transition={{ duration: 2, repeat: Infinity }} className={`flex flex-col items-center gap-2 ${textSecondary}`}>
            <span className="text-sm">Scroll to explore</span>
            <span className="text-2xl">↓</span>
          </motion.div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {stats.map((stat, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }} whileHover={{ scale: 1.05, y: -5 }} className={`p-8 rounded-3xl border text-center backdrop-blur ${cardBg}`}>
                <stat.icon className="w-10 h-10 mx-auto mb-3 text-blue-400" />
                <div className="text-5xl font-black bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">{stat.value}</div>
                <div className={textSecondary}>{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 px-6 relative">
        <div className={`absolute inset-0 ${darkMode ? "bg-gradient-to-b from-transparent via-blue-500/5 to-transparent" : "bg-gradient-to-b from-transparent via-blue-100/50 to-transparent"}`} />
        <div className="max-w-7xl mx-auto relative">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">How It Works</span>
            </h2>
            <p className={`text-lg ${textSecondary}`}>Simple 4-step process to detect defects</p>
          </motion.div>

          <div className="grid md:grid-cols-4 gap-8">
            {workflow.map((item, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }} className="relative">
                {i < workflow.length - 1 && (
                  <div className="hidden md:block absolute top-12 left-full w-full h-0.5 bg-gradient-to-r from-blue-500/50 to-transparent" />
                )}
                <motion.div whileHover={{ scale: 1.05 }} className={`p-6 rounded-2xl border text-center relative ${cardBg}`}>
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-sm font-bold text-white">{item.step}</div>
                  <item.icon className="w-12 h-12 mx-auto mb-4 mt-4 text-blue-400" />
                  <h3 className={`text-xl font-bold mb-2 ${textPrimary}`}>{item.title}</h3>
                  <p className={`text-sm ${textSecondary}`}>{item.desc}</p>
                </motion.div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Powerful Features</span>
            </h2>
            <p className={`text-lg ${textSecondary}`}>Everything you need for industrial quality control</p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.05 }} whileHover={{ scale: 1.02, y: -5 }} className={`p-8 rounded-2xl border backdrop-blur group ${cardBg}`}>
                <feature.icon className="w-12 h-12 mb-4 text-blue-400 group-hover:scale-110 transition-transform" />
                <h3 className={`text-xl font-bold mb-3 ${textPrimary}`}>{feature.title}</h3>
                <p className={`leading-relaxed ${textSecondary}`}>{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Models Section */}
      <section className="py-20 px-6 relative">
        <div className={`absolute inset-0 ${darkMode ? "bg-gradient-to-b from-transparent via-purple-500/5 to-transparent" : "bg-gradient-to-b from-transparent via-purple-100/50 to-transparent"}`} />
        <div className="max-w-7xl mx-auto relative">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">AI Models</span>
            </h2>
            <p className={`text-lg ${textSecondary}`}>Three specialized architectures for different use cases</p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { name: "CAE", full: "Convolutional Autoencoder", score: "0.62 AUC", color: "from-blue-500 to-cyan-500", desc: "Simple, stable, and effective baseline architecture" },
              { name: "VAE", full: "Variational Autoencoder", score: "0.53 AUC", color: "from-purple-500 to-pink-500", desc: "Probabilistic encoding with latent space sampling" },
              { name: "DAE", full: "Denoising Autoencoder", score: "0.62 AUC", color: "from-orange-500 to-red-500", desc: "Robust feature learning through noise injection" },
            ].map((model, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }} whileHover={{ scale: 1.03 }} className={`p-8 rounded-3xl border text-center relative overflow-hidden group ${cardBg}`}>
                <div className={`absolute inset-0 bg-gradient-to-br ${model.color} opacity-0 group-hover:opacity-10 transition-opacity`} />
                <div className={`inline-block px-6 py-3 rounded-xl bg-gradient-to-r ${model.color} text-white font-black text-3xl mb-4`}>{model.name}</div>
                <h3 className={`text-lg font-semibold mb-2 ${textPrimary}`}>{model.full}</h3>
                <p className={`text-sm mb-4 ${textSecondary}`}>{model.desc}</p>
                <div className={`text-2xl font-bold ${textPrimary}`}>{model.score}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Research Insights</span>
            </h2>
          </motion.div>

          <div className="space-y-6">
            {testimonials.map((item, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: i % 2 === 0 ? -30 : 30 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }} className={`p-8 rounded-2xl border ${cardBg}`}>
                <p className={`text-xl mb-4 leading-relaxed ${darkMode ? "text-slate-300" : "text-slate-700"}`}>&ldquo;{item.quote}&rdquo;</p>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
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

      {/* Final CTA */}
      <section className="py-20 px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="max-w-4xl mx-auto">
          <div className="relative p-12 md:p-16 rounded-3xl overflow-hidden text-center">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-pink-600/20" />
            <div className="absolute inset-0 backdrop-blur-xl" />
            <div className={`absolute inset-0 border rounded-3xl ${darkMode ? "border-white/10" : "border-slate-200"}`} />
            
            <div className="relative z-10">
              <motion.div animate={{ y: [0, -10, 0] }} transition={{ duration: 3, repeat: Infinity }}>
                <Cpu className="w-16 h-16 mx-auto mb-6 text-blue-400" />
              </motion.div>
              <h2 className={`text-4xl md:text-5xl font-bold mb-6 ${textPrimary}`}>Ready to detect defects?</h2>
              <p className={`text-xl mb-10 max-w-2xl mx-auto ${textSecondary}`}>
                Upload your industrial images and get instant AI-powered analysis with visual heatmaps
              </p>
              <Link href="/detect">
                <motion.button whileHover={{ scale: 1.05, boxShadow: "0 0 60px rgba(99, 102, 241, 0.4)" }} whileTap={{ scale: 0.95 }} className="px-12 py-5 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 text-white rounded-2xl font-bold text-xl shadow-2xl">
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
