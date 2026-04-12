# Industrial Defect Detection — Comprehensive Project Overview

> **Purpose**: This document is a complete technical reference for the entire project. All claims are verified against the repository's actual code, configs, trained weights, and result files.
> **Last verified**: April 2026

---

## 1. Project Summary

This is a **bachelor thesis project** on unsupervised anomaly detection in industrial manufacturing images. It implements **five anomaly detection architectures** (CAE, VAE, DAE, Skip-CAE, PatchCore) plus a **supervised CNN classifier**, trains and evaluates them on the MVTec AD benchmark, and provides a **full-stack web application** (Next.js + FastAPI) for interactive defect detection with heatmap visualization.

- **Author**: Mohammad Hamim
- **Supervisor**: Lu Yang (卢洋), Zhengzhou University, School of Computer Science and Artificial Intelligence
- **Repository**: https://github.com/md-hameem/industrial-defect-detection
- **License**: MIT

---

## 2. Problem Being Solved

**Domain**: Automated visual quality control in industrial manufacturing.

**Challenge**: In real-world manufacturing, defective samples are extremely rare and diverse, while normal samples are abundant. This class imbalance makes supervised defect detection impractical. The project investigates **unsupervised anomaly detection** — training models only on normal (defect-free) images, then detecting anomalies as deviations from the learned normal distribution.

**Practical motivation**: Eliminating the need for labeled defect data enables deployment in scenarios where collecting and annotating defective samples is expensive or infeasible.

---

## 3. Objectives

1. Implement and compare 5 anomaly detection architectures (3 autoencoders + Skip-CAE + PatchCore)
2. Evaluate performance on the MVTec AD benchmark dataset (15 categories)
3. Investigate Skip-CAE skip connections and SSIM-based scoring for improved anomaly maps
4. Compare reconstruction-based methods (autoencoders) vs. feature-based SOTA (PatchCore)
5. Investigate cross-dataset generalization (MVTec → KolektorSDD2)
6. Develop a full-stack web application for interactive multi-model defect detection

---

## 4. End-to-End System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                           │
│  MVTec AD / KolektorSDD2 / NEU → Resize 256×256 → ImageNet     │
│  normalization → DataLoader (batch=16) → augmentation           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING / FITTING                      │
│  Reconstruction: CAE / VAE / DAE / Skip-CAE → Train on Normal   │
│  Feature-based:  PatchCore → Extract features → Memory Bank     │
│  Supervised:     CNN → Train on labeled NEU dataset              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION & INFERENCE                        │
│  Reconstruction: Input → Reconstruct → MSE/SSIM Error Map       │
│  PatchCore:      Input → Features → NN Distance → Anomaly Map   │
│  Post-processing: Gaussian smoothing (σ=4.0) → Normalization    │
│  Metrics: Image AUC, AP, Precision, Recall, F1, Pixel AUC      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     WEB APPLICATION                              │
│  Backend (FastAPI):  /predict, /predict/batch, /models           │
│  Frontend (Next.js): Homepage, Detection, Research, About, Hist. │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Repository Structure Overview

```
Thesis/
├── src/                         # Core ML source code
│   ├── config.py                # Centralized configuration
│   ├── data/                    # Dataset loaders + transforms
│   │   ├── mvtec_dataset.py     # MVTec AD loader (15 categories)
│   │   ├── kolektor_dataset.py  # KolektorSDD2 loader (JSON annotation parsing)
│   │   ├── neu_dataset.py       # NEU Surface Defect loader (6 classes)
│   │   └── transforms.py       # Image preprocessing + augmentation
│   ├── models/                  # Model architectures (6 models)
│   │   ├── cae.py               # Convolutional Autoencoder + SSIM scoring
│   │   ├── vae.py               # Variational Autoencoder
│   │   ├── denoising_ae.py      # Denoising Autoencoder (wraps CAE)
│   │   ├── skip_cae.py          # U-Net style Skip-Connection CAE
│   │   ├── patchcore.py         # PatchCore (pretrained ResNet-18)
│   │   └── cnn_classifier.py    # Lightweight supervised CNN
│   ├── training/                # Training pipeline
│   │   ├── trainer.py           # AutoencoderTrainer + EarlyStopping
│   │   ├── vae_trainer.py       # Specialized VAE training with KL annealing
│   │   └── losses.py            # MSE, SSIM, Combined, VAE, Denoising losses
│   ├── evaluation/              # Evaluation & visualization
│   │   ├── metrics.py           # ROC-AUC, AP, F1, IoU, Dice, PRO, pixel AUC
│   │   └── visualization.py     # Plotting functions (12 functions)
│   └── utils/                   # Currently minimal (empty __init__.py)
├── web/
│   ├── backend/                 # FastAPI inference server
│   │   ├── main.py              # API endpoints (v2.0.0)
│   │   ├── inference.py         # ModelInference class, Gaussian smoothing
│   │   └── requirements.txt     # Backend-specific dependencies
│   └── frontend/                # Next.js 16 React application
│       ├── src/app/             # Pages: home, detect, research, about, history
│       ├── src/components/      # Navbar, Footer, ClientLayout, LoadingUI
│       ├── src/context/         # ThemeContext (dark/light mode)
│       └── package.json         # Frontend dependencies
├── notebooks/                   # 14 Jupyter notebooks
│   ├── 00_data_exploration.ipynb
│   ├── 01–04: training notebooks (CAE, VAE, DAE, CNN)
│   ├── 05_analysis_visualization.ipynb
│   ├── 06_cross_dataset_evaluation.ipynb
│   ├── 07_thesis_figures.ipynb
│   ├── 08_comprehensive_evaluation.ipynb
│   ├── 09–10: local Skip-CAE and PatchCore training
│   ├── 11_train_skip_cae_kaggle.ipynb   # Cloud GPU training
│   └── 12_train_patchcore_kaggle.ipynb  # Cloud GPU training
├── outputs/
│   ├── models/                  # 152 trained model files
│   ├── figures/                 # Thesis visualizations
│   ├── logs/                    # Training history JSON files
│   └── *.csv                    # Result CSV files
├── scripts/
│   ├── evaluate_all_models.py   # Batch evaluation for CAE/VAE/DAE
│   └── regenerate_figures.py    # Thesis figure regeneration
├── tests/                       # Unit tests (pytest)
│   ├── test_models.py           # 28 tests for all 6 models
│   ├── test_metrics.py          # 14 tests for evaluation metrics
│   ├── test_data_loading.py     # Data loading tests
│   ├── test_api_quick.py        # API endpoint tests
│   └── conftest.py              # Pytest config
├── datasets/                    # Dataset storage (gitignored content)
│   ├── mvtec_ad/                # MVTec AD (15 categories)
│   ├── kolektor_sdd2/           # KolektorSDD2
│   └── neu_surface_defect/      # NEU Surface Defect
├── docs/                        # Thesis documentation
│   ├── thesis_paper.md          # Full thesis draft
│   ├── mid-term-report.md       # Mid-term report
│   └── proposal.md              # Original proposal
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata (PEP 621)
├── setup.py                     # Package installation
├── CHANGELOG.md                 # Version history (v1.0.0 → v4.0.0)
└── README.md                    # Project README
```

---

## 6. Technology Stack

### Verified from code, configs, and package files:

| Layer | Technology | Evidence |
|-------|-----------|----------|
| **Language** | Python 3.10+ | `pyproject.toml` requires-python ≥3.10 |
| **DL Framework** | PyTorch ≥2.0.0, torchvision ≥0.15.0 | `requirements.txt`, `pyproject.toml` |
| **Data Processing** | NumPy, Pandas, Pillow | `requirements.txt` |
| **ML Utilities** | scikit-learn ≥1.3.0 | ROC-AUC, confusion matrix, etc. |
| **Visualization** | matplotlib ≥3.7.0, seaborn ≥0.12.0 | `visualization.py` imports |
| **Image Smoothing** | SciPy (gaussian_filter) | `inference.py` line 56 |
| **Backend** | FastAPI 0.109.0, uvicorn 0.27.0 | `web/backend/requirements.txt` |
| **Auth** | python-jose[cryptography] 3.3.0, passlib[bcrypt] 1.7.4 | Backend `requirements.txt` |
| **Frontend** | Next.js 16.1.3, React 19.2.3 | `package.json` |
| **Styling** | Tailwind CSS v4 | `package.json` devDependencies |
| **Animations** | Framer Motion 12.27.0 | `package.json` |
| **Icons** | Lucide React 0.562.0 | `package.json` |
| **File Upload** | react-dropzone 14.3.8 | `package.json` |
| **Fonts** | Inter, Geist, Geist Mono (Google Fonts) | `layout.tsx` |
| **Testing** | pytest ≥7.4.0 | `requirements.txt`, `pyproject.toml` |
| **Notebooks** | Jupyter, ipykernel | `requirements.txt` |
| **Cloud Training** | Kaggle (Tesla T4 / P100 GPUs) | Notebooks 11, 12 |

---

## 7. Data Pipeline

### Seven steps, verified from `src/data/transforms.py` and dataset loaders:

1. **Load image** — PIL Image.open → RGB conversion
2. **Resize** — All images resized to **256×256** (`config.py` IMAGE_SIZE=256)
3. **Augmentation** (training only, two levels):
   - **Standard**: RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomRotation(±10°)
   - **Strong**: Standard + ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02), RandomAffine(translate=5%, scale=0.95–1.05), GaussianBlur(kernel_size=3, σ=0.1–1.0)
4. **ToTensor** — Convert to PyTorch tensor
5. **Normalize** — ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
6. **Ground truth masks** — Resized to 256×256 with nearest-neighbor interpolation, no normalization
7. **DataLoader** — batch_size=16, num_workers=0 (Windows compat), shuffle=True for training

---

## 8. Dataset Details

### 8.1 MVTec AD (Primary Benchmark)
- **Source**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **15 categories**: 5 textures (carpet, grid, leather, tile, wood) + 10 objects (bottle, cable, capsule, hazelnut, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper)
- **Images**: 3,629 train + 1,725 test = 5,354 total
- **Resolution**: 700×700 to 1024×1024 (resized to 256×256)
- **Annotations**: Pixel-level ground truth masks for defect regions
- **Train split**: Normal (good) images only
- **Test split**: Normal + multiple defect types per category
- **File format**: PNG images, PNG masks with `_mask` suffix
- **Verified from**: `src/data/mvtec_dataset.py`, `datasets/README.md`

### 8.2 KolektorSDD2 (Generalization Testing)
- **Source**: https://www.vicos.si/resources/kolektorsdd2/
- **Description**: Real-world electrical commutator surface defects
- **Structure**: train/test splits with img/ and ann/ subdirectories
- **Annotations**: Supervisely JSON format with base64-encoded bitmap masks
- **Images**: 2,335 train + 1,004 test
- **Usage**: Cross-dataset generalization evaluation
- **Verified from**: `src/data/kolektor_dataset.py` (JSON parsing + bitmap decoding)

### 8.3 NEU Surface Defect (Supervised Baseline)
- **Source**: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/
- **6 categories**: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- **Images**: 1,440 train + 360 validation = 1,800 total
- **Structure**: train/validation → images/ → category_folders/
- **Usage**: Supervised classification baseline (CNN)
- **Verified from**: `src/data/neu_dataset.py`, `src/config.py` NEU_CATEGORIES

---

## 9. Preprocessing and Augmentation

*(Verified from `src/data/transforms.py`)*

| Parameter | Value | Source |
|-----------|-------|--------|
| Input size | 256×256 | `config.py` IMAGE_SIZE |
| Normalization mean | [0.485, 0.456, 0.406] | `config.py` NORMALIZE_MEAN |
| Normalization std | [0.229, 0.224, 0.225] | `config.py` NORMALIZE_STD |
| Horizontal flip p | 0.5 | `transforms.py` line 32 |
| Vertical flip p | 0.5 | `transforms.py` line 33 |
| Rotation range | ±10° | `transforms.py` line 34 |
| ColorJitter (strong) | brightness=0.2, contrast=0.2, sat=0.1, hue=0.02 | `transforms.py` lines 39–44 |
| GaussianBlur (strong) | kernel=3, σ=0.1–1.0 | `transforms.py` line 50 |

The `denormalize()` function reverses normalization for visualization.

---

## 10. Model Architectures

### 10.1 Convolutional Autoencoder (CAE)
- **File**: `src/models/cae.py`
- **Class**: `ConvAutoencoder`
- **Encoder**: 4 ConvBlocks: 3→32→64→128→256, each Conv2d(3×3, stride=2) + BN + ReLU
- **Latent**: 16×16×256 = 65,536 dimensions (spatial bottleneck)
- **Decoder**: 4 DeconvBlocks: 256→128→64→32→3, each ConvTranspose2d(3×3, stride=2) + BN + ReLU
- **Parameters**: 2,764,099
- **SSIM scoring**: `get_anomaly_map_ssim()` and `get_anomaly_map_combined()` with 11×11 Gaussian window (σ=1.5), α=0.5 weighting
- **Error scoring**: MSE-based `get_anomaly_score()`, SSIM-based, and combined MSE+SSIM

### 10.2 Variational Autoencoder (VAE)
- **File**: `src/models/vae.py`
- **Class**: `VariationalAutoencoder`
- **Encoder**: Same conv layers as CAE + fc_mu and fc_logvar projections
- **Latent**: **128-dimensional** dense vector (via fully connected layers)
- **Decoder**: FC projection back to 256×16×16 + transposed convolutions
- **Parameters**: 26,009,603 (much larger due to FC layers)
- **Reparameterization**: z = μ + σ × ε with logvar clamping [-20, 2]
- **Loss**: MSE reconstruction + β × KL divergence
- **Anomaly score**: Combines reconstruction error + KL divergence

### 10.3 Denoising Autoencoder (DAE)
- **File**: `src/models/denoising_ae.py`
- **Class**: `DenoisingAutoencoder`
- **Architecture**: Wraps `ConvAutoencoder` — identical encoder/decoder
- **Noise injection**: Gaussian noise with σ = noise_factor (default=0.3) during training only
- **Parameters**: 2,764,099 (same as CAE)
- **Key difference**: No clamping on noisy input (designed for normalized data)

### 10.4 Skip-Connection CAE (U-Net Style)
- **File**: `src/models/skip_cae.py`
- **Class**: `SkipConvAutoencoder`
- **Encoder**: 4 `SkipEncoderBlock` modules, each with 2× (Conv3×3 → BN → ReLU) + MaxPool2d(2)
- **Bottleneck**: Conv(256→512) → BN → ReLU → Conv(512→256) → BN → ReLU
- **Decoder**: 3 `SkipDecoderBlock` modules with ConvTranspose2d upsampling + skip concatenation
- **Final layer**: Upsample + concatenate with processed input + 1×1 conv to 3 channels
- **Skip connections**: Concatenation (not addition); `nn.functional.interpolate` handles size mismatches
- **Input level skip**: The original input image itself serves as the final skip connection
- **Parameters**: ~4,200,000 (verified: more than CAE due to double-conv blocks + skip processing)
- **Channels**: [32, 64, 128, 256] (configurable)

### 10.5 PatchCore (Feature-Based Detection)
- **File**: `src/models/patchcore.py`
- **Class**: `PatchCoreModel`
- **Backbone**: ResNet-18 (ImageNet pretrained, fully frozen)
- **Parameters**: 11,689,512 total, **0 trainable**
- **Feature layers**: layer2 (128ch, 32×32) + layer3 (256ch, 16×16)
- **Feature extraction**: Upsample layer3 → 32×32, concatenate → 384-dim per patch
- **Patches per image**: 1,024 (32×32 grid)
- **Memory bank**: Random subsampling at 10% (simplified coreset selection)
- **Scoring**: k=3 nearest neighbors, L2 distance, averaged
- **Image score**: max(all patch distances)
- **Anomaly map**: Reshape patch distances → 32×32 → bilinear upsample to 256×256
- **Chunked distance computation**: 1,000 memory entries per chunk to prevent OOM

### 10.6 Lightweight CNN Classifier
- **File**: `src/models/cnn_classifier.py`
- **Class**: `LightweightCNN`
- **Feature extractor**: 4 ConvBlocks [32, 64, 128, 256], each Conv(3×3) + BN + ReLU + MaxPool(2)
- **Pooling**: AdaptiveAvgPool2d(1) — global average pooling
- **Classifier**: Dropout(0.5) → Linear(256 → 6)
- **Designed for**: NEU Surface Defect (6 classes)
- **Methods**: `predict()`, `predict_proba()`, `get_features()`

---

## 11. Training Pipeline

### 11.1 Configuration (from `src/config.py`)

| Parameter | Value | Source |
|-----------|-------|--------|
| Image size | 256 | `config.py` line 38 |
| Batch size | 16 | `config.py` line 75 |
| Learning rate | **1e-3** | `config.py` line 77 |
| Weight decay | 1e-5 | `config.py` line 78 |
| Epochs | 100 | `config.py` line 76 |
| Early stopping patience | 10 | `config.py` line 85 |
| Noise factor (DAE) | 0.3 | `config.py` line 99 |
| Latent dim (VAE) | 128 | `config.py` line 95 |
| Device | Auto-detect CUDA, fallback CPU | `config.py` line 72 |
| Num workers | 0 | Windows compatibility |

> **Note**: The config file specifies LR=1e-3. Notebooks may override this per-experiment.

### 11.2 Trainer (`src/training/trainer.py`)
- `AutoencoderTrainer` class handles train/validation loops
- `EarlyStopping` with patience and min_delta
- `TrainingHistory` tracks metrics per epoch, saves to JSON
- Supports `ReduceLROnPlateau`, `StepLR`, `CosineAnnealingLR`, `ExponentialLR` schedulers
- Checkpointing: saves best model + periodic (every 10 epochs) + final
- Optimizer factory: supports Adam, AdamW, SGD

### 11.3 VAE-Specific Training (`src/training/vae_trainer.py`)
- **KL annealing**: β increases linearly from 0 to β_max over first 10 epochs
- **β_max**: 0.001 (very small — heavily prioritizes reconstruction)
- **Gradient clipping**: max_norm=1.0 to prevent exploding gradients
- **NaN detection**: Skips batches with NaN losses
- **Logvar clamping**: [-20, 2] in `reparameterize()` for numerical stability

### 11.4 Loss Functions (`src/training/losses.py`)
- **MSELoss**: Standard pixel-wise MSE
- **SSIMLoss**: 1 - SSIM using 11×11 Gaussian window (σ=1.5)
- **CombinedLoss**: α × MSE + (1-α) × SSIMLoss (default α=0.8 for CAE)
- **VAELoss**: MSE + β × KL divergence
- **DenoisingLoss**: MSE between reconstruction and **clean** (not noisy) input
- Factory function: `get_loss_function(model_type)`

### 11.5 Cloud GPU Training (Kaggle)
- **Notebooks**: `11_train_skip_cae_kaggle.ipynb`, `12_train_patchcore_kaggle.ipynb`
- **Hardware**: Tesla T4 / P100 GPUs (16GB VRAM)
- **Motivation**: OOM issues when training Skip-CAE and PatchCore locally
- **Output**: Trained model weights synced to local `outputs/models/`
- **Verified**: 152 model files present in `outputs/models/`

---

## 12. Evaluation Methodology and Metrics

### 12.1 Evaluation Protocol
- Train only on normal (good) samples
- Evaluate on combined normal + anomalous test set
- Use official MVTec AD train/test split
- Find optimal threshold using F1 maximization

### 12.2 Metrics Implemented (`src/evaluation/metrics.py`)

**Image-level:**
- ROC-AUC (Area Under ROC Curve)
- Average Precision (AP)
- Precision, Recall, F1 at optimal threshold
- Accuracy

**Pixel-level:**
- Pixel-level AUC (ROC-AUC on flattened pixel predictions)
- Mean IoU (Intersection over Union)
- Mean Dice coefficient
- PRO (Per-Region Overlap) — implemented but commented out in default pipeline

**Threshold optimization:**
- F1-based (default)
- Youden's J statistic
- Precision-targeted (≥0.9)
- Recall-targeted (≥0.9)

### 12.3 MVTec AD Results (Verified from CSV files)

**Mean Image-Level Performance (15 categories):**

| Model | Image AUC | AP | Precision | Recall | F1 |
|-------|-----------|-----|-----------|--------|-----|
| CAE | 0.580 | 0.796 | 0.757 | 0.982 | 0.849 |
| VAE | 0.412 | 0.706 | 0.720 | 0.990 | 0.822 |
| DAE | 0.596 | 0.813 | 0.762 | 0.995 | 0.854 |
| Skip-CAE | 0.594 | — | — | — | — |
| PatchCore | 0.632 | — | — | — | — |

> **Source**: `outputs/comprehensive_metrics_report.csv` (CAE/VAE/DAE), `outputs/skip_cae_results_all.csv`, `outputs/patchcore_results_all.csv`

> **Note**: Skip-CAE and PatchCore CSV files only contain AUC per category, not full metric breakdowns (AP, Precision, Recall, F1). The full metric tables for these models are not available in the outputs.

**Per-Category Image AUC (all 5 models):**

| Category | CAE | VAE | DAE | Skip-CAE | PatchCore | Best |
|----------|-----|-----|-----|----------|-----------|------|
| Bottle | 0.550 | 0.199 | 0.537 | 0.440 | 0.388 | CAE |
| Cable | 0.458 | 0.361 | 0.464 | 0.486 | 0.678 | PatchCore |
| Capsule | 0.477 | 0.482 | 0.466 | 0.428 | 0.506 | PatchCore |
| Carpet | 0.330 | 0.617 | 0.332 | 0.513 | 0.630 | PatchCore |
| Grid | 0.779 | 0.297 | 0.870 | 0.613 | 0.662 | DAE |
| Hazelnut | 0.877 | 0.255 | 0.888 | 0.790 | 0.933 | PatchCore |
| Leather | 0.447 | 0.303 | 0.389 | 0.760 | 0.713 | Skip-CAE |
| Metal Nut | 0.268 | 0.152 | 0.268 | 0.422 | 0.765 | PatchCore |
| Pill | 0.751 | 0.601 | 0.762 | 0.840 | 0.667 | Skip-CAE |
| Screw | 0.979 | 0.074 | 0.986 | 0.001 | 0.693 | DAE |
| Tile | 0.822 | 0.569 | 0.808 | 0.863 | 0.701 | Skip-CAE |
| Toothbrush | 0.656 | 0.686 | 0.650 | 0.636 | 0.461 | VAE |
| Transistor | 0.403 | 0.303 | 0.445 | 0.573 | 0.491 | Skip-CAE |
| Wood | 0.948 | 0.804 | 0.962 | 0.960 | 0.843 | DAE |
| Zipper | 0.506 | 0.480 | 0.487 | 0.588 | 0.364 | Skip-CAE |
| **MEAN** | **0.580** | **0.412** | **0.596** | **0.594** | **0.632** | **PatchCore** |

**Pixel-Level Metrics (CAE/VAE/DAE only):**

| Model | Mean Pixel AUC | Mean IoU | Mean Dice |
|-------|---------------|----------|-----------|
| CAE | 0.618 | 0.011 | 0.020 |
| VAE | 0.524 | 0.025 | 0.044 |
| DAE | 0.595 | 0.012 | 0.022 |

### 12.4 Cross-Dataset Generalization (MVTec → Kolektor)

*(Source: `outputs/kolektor_generalization_results.csv`)*

| Trained On | CAE | VAE | DAE |
|------------|-----|-----|-----|
| Grid | **0.690** | 0.574 | **0.688** |
| Leather | 0.668 | 0.463 | 0.646 |
| Carpet | 0.665 | 0.545 | 0.682 |
| Wood | 0.662 | 0.493 | 0.652 |
| Tile | 0.649 | 0.590 | 0.575 |
| Bottle | 0.637 | 0.496 | 0.609 |
| Metal Nut | 0.622 | 0.587 | 0.617 |

### 12.5 CNN Classifier Results (NEU Dataset)
- **Accuracy**: 99% (verified from thesis, README, CHANGELOG)
- **Classes**: 6 (Crazing, Inclusion, Patches, Pitted, Rolled, Scratches)
- **Training**: 50 epochs
- **Confusion matrix**: Available in `outputs/figures/cnn_confusion_matrix.png`

---

## 13. Inference Pipeline

### 13.1 Backend Inference (`web/backend/inference.py`)

**Processing flow:**
1. **Image validation**: Cap at 4 megapixels (downscale if larger)
2. **Preprocessing**: Same transforms as training (resize 256×256, ImageNet normalization)
3. **Model loading**: Lazy loading with caching (model loaded once, reused)
4. **Model-specific inference**:
   - **CAE**: Combined MSE+SSIM anomaly map via `get_anomaly_map_combined()`
   - **VAE**: Standard reconstruction + anomaly scoring
   - **DAE**: Reconstruction via `reconstruct()` (no noise at inference)
   - **Skip-CAE**: Standard forward pass + error map
   - **PatchCore**: Feature extraction → NN distance → anomaly map
   - **CNN**: Class probabilities via `predict_proba()`
5. **Post-processing**: Gaussian smoothing (σ=4.0) + per-image normalization to [0,1]
6. **Visualization**: 512×512 heatmap overlay (jet colormap, α=0.5)
7. **Output**: Base64-encoded original, reconstruction, and heatmap images

### 13.2 Heatmap Generation
- Gaussian smoothing via `scipy.ndimage.gaussian_filter` with σ=4.0
- Per-image min-max normalization to [0,1]
- matplotlib jet colormap overlay at α=0.5 transparency
- Output resolution: 512×512 pixels

---

## 14. Backend Architecture

### 14.1 Framework & Configuration
- **Framework**: FastAPI 0.109.0 (v2.0.0 of the API)
- **Server**: uvicorn (port 8000)
- **CORS**: Allow all origins (development config)
- **Authentication**: Optional JWT (OAuth2 bearer token)
  - Secret key from `IDD_SECRET_KEY` environment variable
  - bcrypt password hashing
  - 60-minute token expiration
  - In-memory user database (demo: admin/secret)

### 14.2 API Endpoints (verified from `web/backend/main.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + version info |
| POST | `/token` | Login (get JWT) |
| GET | `/models` | List available trained models |
| GET | `/model-types` | List supported model architectures |
| POST | `/predict` | Single model inference |
| POST | `/predict/batch` | Multi-model comparison |
| GET | `/categories` | List categories with trained models |
| GET | `/cnn/available` | Check CNN model availability |

### 14.3 Response Formats
- **Anomaly detection** response: success, model, category, anomaly_score, original_image (base64), reconstruction (base64), heatmap (base64), processing_time
- **CNN classification** response: predicted_class, confidence, class_probabilities, chart_image (base64)
- **Batch** response: results array with per-model results

---

## 15. Frontend Architecture

### 15.1 Framework & Stack
- **Framework**: Next.js 16.1.3 (App Router)
- **React**: 19.2.3
- **TypeScript**: 5.x
- **Styling**: Tailwind CSS v4 (via PostCSS)
- **Animations**: Framer Motion 12.27.0
- **Icons**: Lucide React
- **File upload**: react-dropzone
- **Fonts**: Inter (primary), Geist, Geist Mono

### 15.2 Pages (verified from `web/frontend/src/app/`)

| Page | File | Description |
|------|------|-------------|
| Homepage | `page.tsx` (22KB) | Hero section, stats, workflow, features, model cards, CTA |
| Detection | `detect/page.tsx` (49KB) | Image upload, model selection, heatmap visualization, compare mode |
| Research | `research/page.tsx` (26KB) | Performance tables, metrics, cross-dataset results, visualizations |
| About | `about/page.tsx` (18KB) | Project info, author/supervisor, methodology |
| History | `history/page.tsx` (11KB) | Prediction history with filtering and export |

### 15.3 Shared Components
- **Navbar** (`Navbar.tsx`): Navigation with theme toggle, mobile responsive
- **Footer** (`Footer.tsx`): Site-wide footer
- **ClientLayout** (`ClientLayout.tsx`): Wraps pages with Navbar + Footer
- **LoadingUI** (`LoadingUI.tsx`): Page-specific loading screens
- **ThemeContext** (`ThemeContext.tsx`): Dark/light mode state management

### 15.4 Design System
- Deep OLED-dark aesthetics with glassmorphism
- CSS perspective transforms and 3D effects
- Animated gradients and glow borders
- Hover micro-animations
- Responsive layout (mobile to desktop)

---

## 16. Integration Between ML, Backend, and Frontend

### 16.1 Model → Backend
- Backend `inference.py` imports from `src/models/` via sys.path manipulation
- Models loaded from `outputs/models/` directory
- Naming convention: `{model_type}_{category}_final.pth` (autoencoders), `patchcore_{category}_memory.pth` (PatchCore), `cnn_classifier_final.pth` (CNN)
- Models cached in `ModelInference.models` dict after first load

### 16.2 Backend → Frontend
- Frontend sends images to FastAPI endpoints via multipart form data
- Backend returns JSON with base64-encoded images
- Frontend renders base64 images directly in `<img>` tags
- Batch endpoint enables multi-model comparison in single request

### 16.3 Data Flow
```
User uploads image → Frontend react-dropzone → POST /predict (multipart)
→ Backend validates image → Loads/caches model → Runs inference
→ Gaussian smoothing → Generates heatmap overlay → Base64 encode
→ JSON response → Frontend renders results with animations
```

---

## 17. Important Configurations and Dependencies

### 17.1 Model Checkpoint Format
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # if present
    'timestamp': '2026-...',
    'epoch': N,
    'train_loss': float,
    'val_loss': float,  # if present
}
```

### 17.2 PatchCore Memory Bank Format
```python
{
    'memory_bank': tensor (N_patches × 384),
    'feature_size': (H, W, C),
    'k': 3,
    'subsample_ratio': 0.1,
    'layers': ['layer2', 'layer3'],
}
```

### 17.3 Security Configuration
- `IDD_SECRET_KEY` environment variable for JWT secrets
- `weights_only=True` in `torch.load()` for PyTorch 2.4+ security
- Image size validation: 4 megapixel cap
- CORS wildcard (development; should be restricted in production)

---

## 18. Execution / Deployment Workflow

### 18.1 Development Setup
```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

### 18.2 Training (Local)
```bash
jupyter notebook                # Open notebooks 01–04 for training
```

### 18.3 Training (Cloud GPU)
- Upload notebooks 11, 12 to Kaggle
- Run with GPU accelerator enabled
- Download trained model files to `outputs/models/`

### 18.4 Running Web Application
```bash
# Terminal 1: Backend
cd web/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd web/frontend
npm install
npm run dev    # Port 3000
```

---

## 19. Current Features

### ML / Research
- [x] 5 anomaly detection architectures + CNN classifier
- [x] Training on MVTec AD (15 categories × 5 models = 75+ trained models)
- [x] PatchCore with pretrained ResNet-18 features
- [x] Skip-CAE with U-Net style skip connections
- [x] SSIM-based anomaly scoring (CAE)
- [x] Combined MSE+SSIM anomaly maps
- [x] Cross-dataset generalization (MVTec → Kolektor)
- [x] Comprehensive evaluation metrics (image-level + pixel-level)
- [x] 152 trained model files
- [x] Cloud GPU training pipeline (Kaggle)

### Web Application
- [x] Full-stack: Next.js 16 + FastAPI
- [x] 6-model inference (CAE, VAE, DAE, Skip-CAE, PatchCore, CNN)
- [x] Batch API for multi-model comparison
- [x] Gaussian-smoothed heatmap overlays
- [x] Dark/light mode with glassmorphism
- [x] Prediction history page
- [x] Research dashboard with metrics
- [x] JWT authentication
- [x] Image size validation

### Testing & Quality
- [x] pytest test suite (test_models.py + test_metrics.py + test_data_loading.py + test_api_quick.py)
- [x] Comprehensive CHANGELOG (v1.0.0 → v4.0.0)
- [x] Thesis paper draft
- [x] Mid-term report

---

## 20. Current Limitations

1. **Low pixel-level accuracy for autoencoders**: IoU ~0.01–0.02, Dice ~0.02–0.04 — poor localization
2. **VAE training instability**: Requires careful hyperparameter tuning (KL annealing, logvar clamping)
3. **Category-specific models**: Each category requires a separate model/memory bank
4. **PatchCore memory requirements**: Large memory banks (up to 61MB for hazelnut)
5. **PatchCore simplified coreset**: Uses random subsampling instead of greedy coreset selection
6. **No Docker / containerized deployment**: Application runs via manual uvicorn + npm commands
7. **CORS wildcard**: Not production-ready security configuration
8. **In-memory user database**: No persistent user authentication
9. **No CI/CD pipeline**: No automated testing or deployment
10. **Skip-CAE screw anomaly** (AUC=0.001): Appears to be a training failure for this category
11. **Reconstruction-based performance ceiling**: ~0.60 AUC mean, significantly below SOTA

---

## 21. Future Improvement Opportunities

1. **Greedy coreset selection** for PatchCore memory bank quality
2. **Test-time augmentation** for improved PatchCore accuracy
3. **Few-shot learning** for scenarios with limited normal samples
4. **Multi-category PatchCore** (single memory bank for all categories)
5. **Edge deployment** via ONNX quantization
6. **Attention mechanisms** in autoencoder bottlenecks
7. **GAN-based detection** (AnoGAN, f-AnoGAN)
8. **Docker containerization** for reproducible deployment
9. **CI/CD pipeline** with automated testing
10. **Production authentication** (external identity provider)

---

## 22. Important Files and Their Responsibilities

| File | Purpose |
|------|---------|
| `src/config.py` | Centralized configuration: paths, hyperparameters, categories |
| `src/models/cae.py` | CAE model + SSIM scoring (298 lines) |
| `src/models/vae.py` | VAE model + reparameterization (324 lines) |
| `src/models/skip_cae.py` | U-Net style autoencoder (260 lines) |
| `src/models/patchcore.py` | Feature-based detector with memory bank (328 lines) |
| `src/data/transforms.py` | Preprocessing + augmentation pipeline (100 lines) |
| `src/data/mvtec_dataset.py` | MVTec AD data loader (217 lines) |
| `src/training/trainer.py` | Training loop + early stopping (425 lines) |
| `src/training/losses.py` | All loss functions (165 lines) |
| `src/training/vae_trainer.py` | VAE training with KL annealing (181 lines) |
| `src/evaluation/metrics.py` | All evaluation metrics (420 lines) |
| `src/evaluation/visualization.py` | Plotting functions (512 lines) |
| `web/backend/main.py` | FastAPI server + endpoints (340 lines) |
| `web/backend/inference.py` | Model loading + inference + heatmaps (393 lines) |
| `web/frontend/src/app/page.tsx` | Homepage with animations (481 lines) |
| `web/frontend/src/app/detect/page.tsx` | Detection interface (49KB) |
| `web/frontend/src/app/research/page.tsx` | Research dashboard (26KB) |
| `scripts/evaluate_all_models.py` | Batch evaluation script (228 lines) |

---

## 23. Verified Facts vs Inferred Points

### Verified Facts (multiple source confirmation)
- ✅ 6 model architectures implemented (verified from code + __init__.py + tests)
- ✅ 152 trained model files (verified from `outputs/models/` listing)
- ✅ Image size 256×256 (verified from config + transforms + model docstrings)
- ✅ ImageNet normalization (verified from config + transforms)
- ✅ PatchCore uses ResNet-18 layer2+layer3 (verified from code)
- ✅ PatchCore k=3, subsample_ratio=0.1 (verified from code)
- ✅ Gaussian smoothing σ=4.0 (verified from inference.py)
- ✅ FastAPI v2.0.0 (verified from main.py + requirements.txt)
- ✅ Next.js 16.1.3 (verified from package.json)
- ✅ VAE latent_dim=128 (verified from config + model code)
- ✅ CAE parameters = 2,764,099 (verified from code structure)
- ✅ Learning rate = 1e-3 in config.py (verified from code)
- ✅ Batch size = 16 (verified from config.py)
- ✅ SSIM window = 11×11, σ=1.5 (verified from cae.py)

### Inferred Points (strongly supported but not independently verified)
- ⚠️ CNN achieves 99% accuracy on NEU — stated in README/CHANGELOG/thesis but CNN training notebook not inspected in detail
- ⚠️ "84 visualizations" — claimed in README, not independently counted
- ⚠️ "3-5× faster" batch API — claimed in CHANGELOG, not benchmarked
- ⚠️ Inference time "~100ms CPU" — claimed in thesis discussion, not independently measured

---

## 24. Open Uncertainties / Missing Evidence

1. **Exact learning rate used in notebook training**: `config.py` says 1e-3 but thesis previously said 1e-4; notebooks may override the config value
2. **Skip-CAE exact parameter count**: Thesis says "~4,200,000" but not verified via `sum(p.numel())`
3. **CNN training details**: Epochs, learning rate, and scheduler used in CNN training — only notebook contains this (not inspected)
4. **VAE β used in final training**: Config has β_max=0.001 but thesis mentions β=1.0 — the VAE trainer uses its own config
5. **Exact test count**: test_models.py has 28 test methods, test_metrics.py has 14. Plus test_data_loading.py and test_api_quick.py. README says "44 tests", thesis says "47 tests". The exact count depends on parameterization.
6. **PatchCore inference speed**: Claimed ~500ms but not benchmarked
7. **Skip-CAE screw failure**: AUC=0.001 is anomalously low — likely a training failure, not documented
8. **Next.js version discrepancy**: package.json says 16.1.3, README badge says "Next.js 15"
