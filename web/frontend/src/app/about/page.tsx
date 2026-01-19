"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { 
  Search, Brain, Thermometer, Cpu, Shuffle, Globe,
  Database, FlaskConical, BarChart3, Layers, Zap,
  Github, Linkedin, User, ArrowRight, BookOpen, Code
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
  {
    name: "CAE",
    full: "Convolutional Autoencoder",
    desc: "Encoder-decoder architecture that learns compressed representations of normal images. Anomalies produce high reconstruction error.",
    layers: "Encoder: 4 Conv layers (32→64→128→256) | Decoder: 4 ConvTranspose layers",
    auc: "0.617",
  },
  {
    name: "VAE",
    full: "Variational Autoencoder",
    desc: "Probabilistic autoencoder with latent space sampling using reparameterization trick. Combines reconstruction and KL divergence loss.",
    layers: "Same architecture + μ/σ layers for latent sampling",
    auc: "0.476",
  },
  {
    name: "DAE",
    full: "Denoising Autoencoder",
    desc: "Trained to reconstruct clean images from noisy inputs, learning robust features that generalize better to anomalies.",
    layers: "CAE architecture + Gaussian noise injection (σ=0.3)",
    auc: "0.621",
  },
];

const datasets = [
  {
    name: "MVTec AD",
    role: "Primary Benchmark",
    desc: "Industrial anomaly detection benchmark with 15 categories of textures and objects. Used for training all autoencoder models.",
    stats: ["5,354 images", "15 categories", "73 defect types"],
    url: "https://www.mvtec.com/company/research/datasets/mvtec-ad",
    color: "blue",
  },
  {
    name: "KolektorSDD2",
    role: "Generalization Testing",
    desc: "Real-world electrical commutator surface defects. Used to evaluate cross-dataset transfer without retraining.",
    stats: ["3,339 images", "Surface cracks", "JSON annotations"],
    url: "https://www.vicos.si/resources/kolektorsdd2/",
    color: "purple",
  },
  {
    name: "NEU Surface Defect",
    role: "Supervised Baseline",
    desc: "Steel surface defects with 6 classes. Used to train CNN classifier as supervised comparison (achieved 99% accuracy).",
    stats: ["1,800 images", "6 classes", "Classification task"],
    url: "http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/",
    color: "orange",
  },
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
  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-16">
        <h1 className="text-4xl font-bold mb-4">
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">About This Project</span>
        </h1>
        <p className="text-xl text-slate-400 max-w-3xl mx-auto">
          A comprehensive deep learning system for industrial defect detection using unsupervised anomaly detection methods
        </p>
      </motion.div>

      {/* Project Overview */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }} className="mb-16">
        <div className="p-8 rounded-3xl bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10 border border-white/10">
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
              <FlaskConical className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-bold">Project Goal</h2>
          </div>
          <p className="text-lg text-slate-300 leading-relaxed text-center max-w-4xl mx-auto">
            Develop an <strong className="text-white">unsupervised anomaly detection system</strong> that can identify manufacturing 
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
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.05 }}
              whileHover={{ scale: 1.03, y: -5 }}
              className="p-6 rounded-2xl bg-slate-800/30 border border-white/5"
            >
              <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-blue-500/20 to-purple-500/20 flex items-center justify-center mb-4">
                <feature.icon className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-lg font-bold mb-2">{feature.title}</h3>
              <p className="text-slate-400 text-sm">{feature.desc}</p>
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
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 + i * 0.1 }}
              className="p-6 rounded-2xl bg-slate-800/30 border border-white/10"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-3 py-1 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold">{arch.name}</span>
                    <span className="text-lg font-semibold">{arch.full}</span>
                  </div>
                  <p className="text-slate-300 mb-3">{arch.desc}</p>
                  <p className="text-sm text-slate-500"><strong className="text-slate-400">Architecture:</strong> {arch.layers}</p>
                </div>
                <div className="text-right px-4 py-2 rounded-xl bg-slate-700/50">
                  <div className="text-2xl font-black text-blue-400">{arch.auc}</div>
                  <div className="text-xs text-slate-500">Mean AUC</div>
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
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + i * 0.1 }}
              className="relative p-4 rounded-xl bg-slate-800/30 border border-white/5 text-center"
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-sm font-bold mx-auto mb-3">
                {step.step}
              </div>
              <h3 className="font-bold text-sm mb-1">{step.title}</h3>
              <p className="text-xs text-slate-400">{step.desc}</p>
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
            <motion.a
              key={i}
              href={ds.url}
              target="_blank"
              rel="noopener noreferrer"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + i * 0.1 }}
              whileHover={{ scale: 1.03, y: -5 }}
              className="p-6 rounded-2xl bg-slate-800/30 border border-white/10 block hover:border-blue-500/50 transition-colors"
            >
              <div className="text-xs font-bold text-blue-400 mb-1">{ds.role}</div>
              <h3 className="text-xl font-bold mb-2">{ds.name}</h3>
              <p className="text-slate-300 text-sm mb-4">{ds.desc}</p>
              <div className="flex flex-wrap gap-2">
                {ds.stats.map((stat, j) => (
                  <span key={j} className="text-xs px-2 py-1 rounded bg-slate-700/50 text-slate-400">{stat}</span>
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
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.7 + i * 0.05 }}
              whileHover={{ scale: 1.05 }}
              className="flex items-center gap-3 px-5 py-3 rounded-xl bg-slate-800/50 border border-white/5"
            >
              <tech.icon className="w-5 h-5 text-blue-400" />
              <div>
                <div className="font-bold text-sm">{tech.name}</div>
                <div className="text-xs text-slate-500">{tech.desc}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Author Section (Compact) */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }} className="mb-16">
        <div className="p-6 rounded-2xl bg-slate-800/30 border border-white/10">
          <div className="flex items-center gap-6 flex-wrap">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shrink-0">
              <User className="w-10 h-10" />
            </div>
            <div className="flex-1 min-w-[200px]">
              <h3 className="text-xl font-bold">Mohammad Hamim</h3>
              <p className="text-blue-400">Bachelors Thesis - Zhengzhou University - 2026</p>
              <p className="text-sm text-slate-400 mt-2">
                Computer Science student specializing in AI and deep learning for industrial applications.
              </p>
            </div>
            <div className="flex gap-3">
              <a href="https://github.com/md-hameem" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition">
                <Github className="w-4 h-4" /> GitHub
              </a>
              <a href="https://linkedin.com/in/md-hameem" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition">
                <Linkedin className="w-4 h-4" /> LinkedIn
              </a>
            </div>
          </div>
        </div>
      </motion.section>

      {/* CTA */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}>
        <div className="flex flex-wrap gap-4 justify-center">
          <Link href="/detect">
            <motion.div whileHover={{ scale: 1.05 }} className="flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 transition">
              <Search className="w-6 h-6" />
              <div>
                <div className="font-bold">Try Detection</div>
                <div className="text-sm text-white/80">Upload an image</div>
              </div>
            </motion.div>
          </Link>
          <Link href="/research">
            <motion.div whileHover={{ scale: 1.05 }} className="flex items-center gap-3 px-8 py-4 rounded-xl bg-slate-800 hover:bg-slate-700 transition">
              <BarChart3 className="w-6 h-6" />
              <div>
                <div className="font-bold">View Results</div>
                <div className="text-sm text-slate-400">Research data</div>
              </div>
            </motion.div>
          </Link>
          <Link href="https://github.com/md-hameem/industrial-defect-detection" target="_blank">
            <motion.div whileHover={{ scale: 1.05 }} className="flex items-center gap-3 px-8 py-4 rounded-xl bg-slate-800 hover:bg-slate-700 transition">
              <Github className="w-6 h-6" />
              <div>
                <div className="font-bold">Source Code</div>
                <div className="text-sm text-slate-400">GitHub repo</div>
              </div>
            </motion.div>
          </Link>
        </div>
      </motion.section>
    </div>
  );
}
