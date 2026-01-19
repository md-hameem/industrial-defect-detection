"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { useTheme } from "@/context/ThemeContext";
import { 
  Search, Brain, Thermometer, Cpu, Shuffle, Globe,
  Database, FlaskConical, BarChart3, Layers, Zap,
  Github, Linkedin, User, Code
} from "lucide-react";

const projectFeatures = [
  { icon: Search, title: "Unsupervised Detection", desc: "Train on normal samples only - no labeled defects required" },
  { icon: Brain, title: "Multiple Architectures", desc: "CAE, VAE, and Denoising AE implementations" },
  { icon: Thermometer, title: "Visual Heatmaps", desc: "Pixel-level anomaly localization with color overlays" },
  { icon: Cpu, title: "CPU Optimized", desc: "Designed for training without GPU hardware" },
  { icon: Shuffle, title: "Cross-Dataset Testing", desc: "Generalization evaluation on unseen domains" },
  { icon: Globe, title: "Web Interface", desc: "Full-stack application with FastAPI + Next.js" },
];

const architectures = [
  { name: "CAE", full: "Convolutional Autoencoder", desc: "Encoder-decoder architecture that learns compressed representations of normal images. Anomalies produce high reconstruction error.", layers: "Encoder: 4 Conv layers (32→64→128→256) | Decoder: 4 ConvTranspose layers", auc: "0.617" },
  { name: "VAE", full: "Variational Autoencoder", desc: "Probabilistic autoencoder with latent space sampling using reparameterization trick. Combines reconstruction and KL divergence loss.", layers: "Same architecture + μ/σ layers for latent sampling", auc: "0.476" },
  { name: "DAE", full: "Denoising Autoencoder", desc: "Trained to reconstruct clean images from noisy inputs, learning robust features that generalize better to anomalies.", layers: "CAE architecture + Gaussian noise injection (σ=0.3)", auc: "0.621" },
];

const datasets = [
  { name: "MVTec AD", role: "Primary Benchmark", desc: "Industrial anomaly detection benchmark with 15 categories of textures and objects. Used for training all autoencoder models.", stats: ["5,354 images", "15 categories", "73 defect types"], url: "https://www.mvtec.com/company/research/datasets/mvtec-ad" },
  { name: "KolektorSDD2", role: "Generalization Testing", desc: "Real-world electrical commutator surface defects. Used to evaluate cross-dataset transfer without retraining.", stats: ["3,339 images", "Surface cracks", "JSON annotations"], url: "https://www.vicos.si/resources/kolektorsdd2/" },
  { name: "NEU Surface Defect", role: "Supervised Baseline", desc: "Steel surface defects with 6 classes. Used to train CNN classifier as supervised comparison (achieved 99% accuracy).", stats: ["1,800 images", "6 classes", "Classification task"], url: "http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/" },
];

const techStack = [
  { name: "PyTorch", icon: Zap, desc: "Deep learning framework" },
  { name: "FastAPI", icon: Zap, desc: "Backend REST API" },
  { name: "Next.js", icon: Code, desc: "React frontend" },
  { name: "Tailwind", icon: Layers, desc: "CSS styling" },
  { name: "NumPy", icon: BarChart3, desc: "Numerical computing" },
  { name: "Matplotlib", icon: BarChart3, desc: "Visualization" },
];

const methodology = [
  { step: 1, title: "Data Preparation", desc: "Load and preprocess images (256×256, ImageNet normalization)" },
  { step: 2, title: "Model Training", desc: "Train autoencoders on normal samples only (15 epochs)" },
  { step: 3, title: "Inference", desc: "Compute reconstruction and calculate pixel-wise error" },
  { step: 4, title: "Anomaly Scoring", desc: "Aggregate error maps to produce image-level anomaly scores" },
  { step: 5, title: "Evaluation", desc: "Calculate ROC-AUC using ground truth labels" },
];

export default function AboutPage() {
  const { darkMode } = useTheme();
  
  // Theme-aware classes
  const cardBg = darkMode ? "bg-slate-800/30 border-white/10" : "bg-white/80 border-slate-200 shadow-sm";
  const textPrimary = darkMode ? "text-white" : "text-slate-900";
  const textSecondary = darkMode ? "text-slate-300" : "text-slate-700";
  const textMuted = darkMode ? "text-slate-400" : "text-slate-500";
  const tagBg = darkMode ? "bg-slate-700/50 text-slate-400" : "bg-slate-200 text-slate-600";
  const buttonBg = darkMode ? "bg-slate-700 hover:bg-slate-600" : "bg-slate-200 hover:bg-slate-300";
  const ctaBg = darkMode ? "bg-slate-800 hover:bg-slate-700" : "bg-slate-100 hover:bg-slate-200";

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-16">
        <h1 className="text-4xl font-bold mb-4">
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">About This Project</span>
        </h1>
        <p className={`text-xl max-w-3xl mx-auto ${textMuted}`}>
          A comprehensive deep learning system for industrial defect detection using unsupervised anomaly detection methods
        </p>
      </motion.div>

      {/* Project Overview */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }} className="mb-16">
        <div className={`p-8 rounded-3xl bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10 border ${darkMode ? "border-white/10" : "border-slate-200"}`}>
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
              <FlaskConical className="w-6 h-6 text-white" />
            </div>
            <h2 className={`text-2xl font-bold ${textPrimary}`}>Project Goal</h2>
          </div>
          <p className={`text-lg leading-relaxed text-center max-w-4xl mx-auto ${textSecondary}`}>
            Develop an <strong className={textPrimary}>unsupervised anomaly detection system</strong> that can identify manufacturing 
            defects by training only on normal (defect-free) samples. This approach eliminates the need for expensive labeled 
            defect data, making it practical for real-world industrial quality control applications.
          </p>
        </div>
      </motion.section>

      {/* Key Features */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="mb-16">
        <h2 className="text-2xl font-bold mb-8 text-center">
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Key Features</span>
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          {projectFeatures.map((feature, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 + i * 0.05 }} whileHover={{ scale: 1.03, y: -5 }} className={`p-6 rounded-2xl border ${cardBg}`}>
              <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-blue-500/20 to-purple-500/20 flex items-center justify-center mb-4">
                <feature.icon className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className={`text-lg font-bold mb-2 ${textPrimary}`}>{feature.title}</h3>
              <p className={`text-sm ${textMuted}`}>{feature.desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Model Architectures */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="mb-16">
        <h2 className="text-2xl font-bold mb-8 text-center">
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Model Architectures</span>
        </h2>
        <div className="space-y-6">
          {architectures.map((arch, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 + i * 0.1 }} className={`p-6 rounded-2xl border ${cardBg}`}>
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-3 py-1 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold">{arch.name}</span>
                    <span className={`text-lg font-semibold ${textPrimary}`}>{arch.full}</span>
                  </div>
                  <p className={`mb-3 ${textSecondary}`}>{arch.desc}</p>
                  <p className={`text-sm ${textMuted}`}><strong className={textSecondary}>Architecture:</strong> {arch.layers}</p>
                </div>
                <div className={`text-right px-4 py-2 rounded-xl ${darkMode ? "bg-slate-700/50" : "bg-slate-100"}`}>
                  <div className="text-2xl font-black text-blue-400">{arch.auc}</div>
                  <div className={`text-xs ${textMuted}`}>Mean AUC</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Methodology */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mb-16">
        <h2 className="text-2xl font-bold mb-8 text-center">
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Methodology</span>
        </h2>
        <div className="grid md:grid-cols-5 gap-4">
          {methodology.map((step, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 + i * 0.1 }} className={`relative p-4 rounded-xl text-center border ${cardBg}`}>
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-sm font-bold text-white mx-auto mb-3">
                {step.step}
              </div>
              <h3 className={`font-bold text-sm mb-1 ${textPrimary}`}>{step.title}</h3>
              <p className={`text-xs ${textMuted}`}>{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Datasets */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mb-16">
        <h2 className="text-2xl font-bold mb-8 text-center flex items-center justify-center gap-2">
          <Database className="w-6 h-6 text-blue-400" />
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Datasets</span>
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          {datasets.map((ds, i) => (
            <motion.a key={i} href={ds.url} target="_blank" rel="noopener noreferrer" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 + i * 0.1 }} whileHover={{ scale: 1.03, y: -5 }} className={`p-6 rounded-2xl block hover:border-blue-500/50 transition-colors border ${cardBg}`}>
              <div className="text-xs font-bold text-blue-400 mb-1">{ds.role}</div>
              <h3 className={`text-xl font-bold mb-2 ${textPrimary}`}>{ds.name}</h3>
              <p className={`text-sm mb-4 ${textSecondary}`}>{ds.desc}</p>
              <div className="flex flex-wrap gap-2">
                {ds.stats.map((stat, j) => (
                  <span key={j} className={`text-xs px-2 py-1 rounded ${tagBg}`}>{stat}</span>
                ))}
              </div>
            </motion.a>
          ))}
        </div>
      </motion.section>

      {/* Tech Stack */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="mb-16">
        <h2 className="text-2xl font-bold mb-8 text-center">
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Technology Stack</span>
        </h2>
        <div className="flex flex-wrap justify-center gap-4">
          {techStack.map((tech, i) => (
            <motion.div key={i} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.7 + i * 0.05 }} whileHover={{ scale: 1.05 }} className={`flex items-center gap-3 px-5 py-3 rounded-xl border ${cardBg}`}>
              <tech.icon className="w-5 h-5 text-blue-400" />
              <div>
                <div className={`font-bold text-sm ${textPrimary}`}>{tech.name}</div>
                <div className={`text-xs ${textMuted}`}>{tech.desc}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Author & Supervisor Section */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }} className="mb-16">
        <div className={`p-6 rounded-2xl border ${cardBg}`}>
          <div className="grid md:grid-cols-2 gap-6">
            {/* Author */}
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shrink-0">
                <User className="w-8 h-8 text-white" />
              </div>
              <div>
                <div className={`text-xs uppercase tracking-wide ${textMuted}`}>Author</div>
                <h3 className={`text-xl font-bold ${textPrimary}`}>Mohammad Hamim</h3>
                <p className={`text-sm ${textMuted}`}>Bachelor&apos;s Thesis - 2026</p>
                <div className="flex gap-2 mt-2">
                  <a href="https://github.com/md-hameem" target="_blank" rel="noopener noreferrer" className={`flex items-center gap-1 px-3 py-1 text-sm rounded-lg transition ${buttonBg}`}>
                    <Github className="w-3 h-3" /> GitHub
                  </a>
                  <a href="https://linkedin.com/in/md-hameem" target="_blank" rel="noopener noreferrer" className={`flex items-center gap-1 px-3 py-1 text-sm rounded-lg transition ${buttonBg}`}>
                    <Linkedin className="w-3 h-3" /> LinkedIn
                  </a>
                </div>
              </div>
            </div>
            
            {/* Supervisor */}
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shrink-0">
                <User className="w-8 h-8 text-white" />
              </div>
              <div>
                <div className={`text-xs uppercase tracking-wide ${textMuted}`}>Supervisor</div>
                <h3 className={`text-xl font-bold ${textPrimary}`}>Lu Yang (卢洋)</h3>
                <p className={`text-sm ${textMuted}`}>Zhengzhou University</p>
                <p className={`text-sm ${textMuted}`}>School of Computer Science</p>
                <a href="mailto:ieylu@zzu.edu.cn" className="text-sm text-blue-400 hover:text-blue-300 transition">ieylu@zzu.edu.cn</a>
              </div>
            </div>
          </div>
        </div>
      </motion.section>

      {/* CTA */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}>
        <div className="flex flex-wrap gap-4 justify-center">
          <Link href="/detect">
            <motion.div whileHover={{ scale: 1.05 }} className="flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 text-white">
              <Search className="w-6 h-6" />
              <div>
                <div className="font-bold">Try Detection</div>
                <div className="text-sm text-white/80">Upload an image</div>
              </div>
            </motion.div>
          </Link>
          <Link href="/research">
            <motion.div whileHover={{ scale: 1.05 }} className={`flex items-center gap-3 px-8 py-4 rounded-xl transition ${ctaBg}`}>
              <BarChart3 className="w-6 h-6 text-blue-400" />
              <div>
                <div className={`font-bold ${textPrimary}`}>View Results</div>
                <div className={`text-sm ${textMuted}`}>Research data</div>
              </div>
            </motion.div>
          </Link>
          <Link href="https://github.com/md-hameem/industrial-defect-detection" target="_blank">
            <motion.div whileHover={{ scale: 1.05 }} className={`flex items-center gap-3 px-8 py-4 rounded-xl transition ${ctaBg}`}>
              <Github className="w-6 h-6" />
              <div>
                <div className={`font-bold ${textPrimary}`}>Source Code</div>
                <div className={`text-sm ${textMuted}`}>GitHub repo</div>
              </div>
            </motion.div>
          </Link>
        </div>
      </motion.section>
    </div>
  );
}
