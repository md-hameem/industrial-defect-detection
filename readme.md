# Industrial Defect Detection Using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Bachelor's Graduation Thesis** - Research on Industrial Defect Detection Methods Based on Deep Learning

## Overview

This project implements **unsupervised anomaly detection** using autoencoder-based deep learning methods for industrial defect detection and localization.

### Key Features

- 🔍 **Unsupervised Learning** - Train on normal samples only
- 🏭 **Industrial Focus** - MVTec AD, KolektorSDD2, NEU datasets
- 🧠 **Multiple Models** - CAE, VAE, Denoising AE
- 📊 **Rich Visualizations** - Heatmaps, ROC curves, latent space
- 💻 **CPU Optimized** - Designed for training without GPU

## Project Structure

```
├── src/
│   ├── config.py           # Configuration settings
│   ├── data/               # Dataset loaders
│   ├── models/             # Model architectures
│   ├── training/           # Training utilities
│   └── evaluation/         # Metrics & visualization
├── notebooks/              # Jupyter notebooks
│   ├── 00_data_exploration.ipynb
│   ├── 01_train_cae.ipynb
│   ├── 02_train_vae.ipynb
│   ├── 03_train_denoising_ae.ipynb
│   ├── 04_train_cnn_classifier.ipynb
│   └── 05_analysis_visualization.ipynb
├── datasets/               # Dataset storage
├── outputs/                # Models, logs, figures
└── tests/                  # Unit tests
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/industrial-defect-detection.git
cd industrial-defect-detection
```

### 2. Setup Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Download Datasets
See `datasets/README.md` for download links.

### 4. Run Notebooks
```bash
jupyter notebook
```
Start with `00_data_exploration.ipynb` → then training notebooks.

## Models

| Model | Type | Key Feature |
|-------|------|-------------|
| **CAE** | Convolutional Autoencoder | Simple, effective baseline |
| **VAE** | Variational Autoencoder | Probabilistic latent space |
| **Denoising AE** | Noise injection | Robust feature learning |
| **CNN** | Classifier | Supervised comparison |

## Results

Results are saved to `outputs/`:
- `models/` - Trained model checkpoints
- `logs/` - Training history JSON
- `figures/` - Visualizations (ROC, heatmaps, etc.)

## Hardware Requirements

Designed for **CPU-only training**:
- Python 3.10+
- 16GB RAM recommended
- ~10GB disk for datasets

## Citation

If you use this code, please cite:
```bibtex
@thesis{defect_detection_2026,
  author = {Mohammad Hamim},
  title = {Research on Industrial Defect Detection Methods Based on Deep Learning},
  school = {Zhengzhou University},
  year = {2026}
}
```

## License

MIT License - see [LICENSE](LICENSE)
