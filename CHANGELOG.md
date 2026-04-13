# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [4.1.0] - 2026-04-14

### Added
- **Comprehensive Project Overview** (`docs/overview.md`)
  - 772-line technical reference verified against actual codebase, configs, and results
  - Covers all 6 models, training pipelines, evaluation, web architecture, and deployment
  - Includes verified-facts vs inferred-points transparency section
- **New Project Screenshots** (`web/frontend/public/`)
  - `aboutpage.png` — About page showcase
  - `homepage_scrolled.png` — Stats & workflow section below the fold
  - `researchpage.png` — Research dashboard with data visualizations
  - Updated `homepage.png` and `detectpage.png` with new 3D design

### Changed
- **Frontend UI Redesign** — Complete visual overhaul across all pages
  - Deep OLED-dark aesthetics with glassmorphism and glow borders
  - CSS perspective transforms and 3D card effects
  - Animated gradients and micro-hover animations
  - Responsive layout improvements (mobile to desktop)
- **Homepage** (`web/frontend/src/app/page.tsx`) — New hero section, interactive stats, workflow steps, feature cards, and model showcase with Framer Motion animations
- **Detection Page** (`web/frontend/src/app/detect/page.tsx`) — Redesigned upload interface and results layout with 3D-inspired styling
- **Research Page** (`web/frontend/src/app/research/page.tsx`) — Expanded dashboard with 5-model performance tables, per-category AUC data, and cross-dataset metrics
- **About Page** (`web/frontend/src/app/about/page.tsx`) — Updated project info, methodology, and architecture details
- **Navbar** (`web/frontend/src/components/Navbar.tsx`) — Refreshed navigation with theme toggle
- **Footer** (`web/frontend/src/components/Footer.tsx`) — Updated site-wide footer
- **ClientLayout** (`web/frontend/src/components/ClientLayout.tsx`) — Enhanced wrapper with improved theme support
- **Global CSS** (`web/frontend/src/app/globals.css`) — 300+ lines of new design tokens, glassmorphism utilities, and animation keyframes
- **Layout** (`web/frontend/src/app/layout.tsx`) — Updated metadata and font configuration
- **Thesis Paper** (`docs/thesis_paper.md`) — Audited and corrected technical claims against actual code and results
- **README** — Updated with new screenshots, 3D design descriptions, and feature highlights

## [4.0.0] - 2026-04-10

### Added
- **Kaggle GPU Training Pipeline** (`notebooks/11_train_skip_cae_kaggle.ipynb`, `notebooks/12_train_patchcore_kaggle.ipynb`)
  - Migrated intensive model training (Skip-CAE, PatchCore) to Kaggle Tesla T4/P100 instances
  - Unified dynamic path detection handling for cloud environment execution
  - Solved CPU Out-of-Memory (OOM) bottlenecks during extensive train loops
- **Model Integration** 
  - Over 150 successfully fitted `.pth` files synced from massive cloud executions to local `outputs/models` directory
  - FastAPI dynamic model resolver perfectly incorporates local cloud-trained model weights

## [3.0.0] - 2026-04-05

### Added
- **Skip-Connection CAE (U-Net Style)** (`src/models/skip_cae.py`)
  - Encoder-to-decoder skip connections preserve spatial detail
  - Produces sharper anomaly maps than vanilla CAE
  - New `SkipConvAutoencoder` class with bottleneck + skip architecture
- **PatchCore Feature-Based Detection** (`src/models/patchcore.py`)
  - SOTA anomaly detection using frozen pretrained ResNet-18 features
  - Memory bank of normal patch features with nearest-neighbor scoring
  - No training required — only feature extraction on normal data
  - Expected 0.85+ AUC on MVTec AD (vs 0.62 for CAE)
  - Pixel-precise anomaly maps via feature distance upsampling
- **SSIM-Based Anomaly Scoring** in CAE
  - `get_anomaly_map_ssim()` captures structural anomalies
  - `get_anomaly_map_combined()` fuses MSE + SSIM for robust detection
- **Batch Inference API** (`/predict/batch` endpoint)
  - Process one image with multiple models in a single request
  - 3-5x faster than sequential calls for comparison mode
- **44 Unit Tests** (`tests/test_models.py`, `tests/test_metrics.py`)
  - Full coverage for all 6 model architectures
  - Metric correctness tests with known inputs/outputs
- **Enhanced Data Augmentation** (`src/data/transforms.py`)
  - New `augment_level='strong'` option with ColorJitter, GaussianBlur, RandomAffine
  - Stronger regularization for small normal-only training sets
- **`/model-types` API endpoint** listing all supported model types

### Changed
- **Heatmap Quality** — Gaussian smoothing (sigma=4.0) + per-image normalization
  - Cleaner, more interpretable anomaly maps
  - Higher resolution output (512px vs 300px)
- **Auto Device Detection** — `DEVICE` auto-detects CUDA instead of hardcoded CPU
- **Frontend Model Selector** — 5 anomaly detection models + CNN classifier
- **Compare Mode** — Uses batch API, supports all 5 models simultaneously
- **Homepage** — Updated model cards to show all 6 models including PatchCore SOTA
- **README** — Updated to v3.0.0 with new model table and feature list

### Fixed
- **Security** — `SECRET_KEY` loaded from `IDD_SECRET_KEY` environment variable
- **PyTorch 2.4+ Deprecation** — `weights_only=True` in all `torch.load()` calls
- **Input Validation** — Images capped at 4 megapixels to prevent OOM crashes
- **API Response Type** — `model_type` field now includes `"anomaly_detector"` for non-AE models

## [2.2.0] - 2026-01-21

### Added
- **Thesis Document** (`docs/thesis_paper.md`)
  - Complete draft with 6 chapters: Introduction, Literature Review, Methodology, Experiments, Discussion, Conclusion.
  - 14 integrated figures (architecture diagrams, result charts, heatmaps).
  - Detailed hyperparameter tables and system architecture.
- **New Visualizations**
  - NEU Surface Defect category examples.
  - Training loss convergence curves.
  - Pixel-level performance metric charts.

### Documentation
- Updated **Methodology** chapter with specific CNN architecture (11M params) and Web App tech stack.
- Refined **Discussion** chapter with insights on VAE bottleneck (128-dim vector) vs CAE (65k-dim spatial) performance trade-offs.

## [2.1.0] - 2026-01-21

### Added
- **Comprehensive Evaluation Metrics Module** (`src/evaluation/metrics.py`)
  - Image-level: ROC-AUC, Average Precision, Precision, Recall, F1
  - Pixel-level: IoU, Dice, Pixel AUC, PRO score
  - `find_optimal_threshold()` with F1/Youden optimization
  - `evaluate_model()` for full dataset evaluation
  - `print_metrics()` for pretty output
- **Evaluation Notebook** (`notebooks/08_comprehensive_evaluation.ipynb`)
  - Bar charts, heatmaps, radar charts
  - Pixel-level metric visualizations
  - Model comparison summaries
- **Side-by-Side Model Comparison** in Detection Page
  - Compare CAE vs VAE vs DAE heatmaps simultaneously
  - Grouped by uploaded image with expandable details
- **Comprehensive Evaluation Section** in Research Page
  - Image-level and pixel-level metrics cards
  - Best/worst categories per model
- **10 Research Figures** including cross-dataset visualizations

### Changed
- Updated VAE results in Research Page with v2 training data
- Updated Cross-Dataset Evaluation with all VAE results
- Model Architecture Comparison cards updated with correct metrics

### Fixed
- **VAE Numerical Stability** - Added `logvar` clamping in `reparameterize()`
- **VAE Training** - Added gradient clipping to prevent exploding gradients
- Added `get_anomaly_score()` method to CAE and DAE models (was missing)
- Fixed `expandedResult` state type to support both number and string keys

## [2.0.0] - 2026-01-19

### Added
- **Full-stack Web Application** for interactive defect detection
  - Next.js 15 frontend with React and Tailwind CSS
  - FastAPI backend for model inference
  - Dark/Light mode theme support
- **CNN Classifier Integration** in web app (99% accuracy on NEU dataset)
  - Class probability bar chart visualization
  - 6-class classification: Crazing, Inclusion, Patches, Pitted, Rolled, Scratches
- **Detection Page Features**:
  - Autoencoder mode (CAE, VAE, DAE) with heatmap visualization
  - CNN mode for supervised classification
  - Score explanation panel with thresholds and model performance
  - Batch upload and processing
  - Compare All Models mode
  - Download individual/all results
- **Research Page** with interactive tables:
  - MVTec AD performance by category
  - Cross-dataset evaluation results
  - Model architecture comparison cards
- **History Page** with filtering and export
- **About Page** with project info, author/supervisor details
- **Homepage** with animated hero, feature cards, and workflow section
- Framer Motion animations throughout the UI
- Lucide React icons (replaced all emojis)
- Global ThemeContext for consistent theming

### Changed
- Updated main README with web application section
- Enhanced backend API to support both autoencoders and CNN classifier
- `/predict` endpoint now returns model-specific responses
- Added `/cnn/available` endpoint to check CNN model status

### Fixed
- CNN model path corrected to `cnn_classifier_final.pth`
- Model type detection in API response

## [1.1.0] - 2026-01-17

### Added
- Cross-dataset evaluation notebook (`06_cross_dataset_evaluation.ipynb`)
- Thesis figures generator notebook (`07_thesis_figures.ipynb`)
- KL annealing for VAE training stability
- 84 thesis-ready visualizations in `outputs/figures/`
- Results CSV files for all models
- Comprehensive README with embedded figures

### Changed
- Updated Kolektor dataset loader to parse JSON annotations (Supervisely format)
- Rewrote training notebooks with multi-category loops
- Enhanced visualization module with empty metric handling

### Fixed
- VAE KL divergence explosion (added beta annealing)
- Scheduler TypeError for ReduceLROnPlateau
- Model normalization mismatch (removed Sigmoid from decoder outputs)
- Denoising AE clamping issue for normalized inputs

### Results
- **CAE**: 0.62 mean ROC-AUC on MVTec AD (15 categories)
- **DAE**: 0.62 mean ROC-AUC on MVTec AD (15 categories)
- **VAE**: 0.53 mean ROC-AUC on MVTec AD (unstable on textures)
- **CNN**: 99% accuracy on NEU Surface Defect (6 classes)
- **Cross-dataset**: 0.69 ROC-AUC (MVTec Grid → Kolektor)

## [1.0.0] - 2026-01-16

### Added
- Initial project structure
- Data loading for MVTec AD, KolektorSDD2, and NEU Surface Defect datasets
- Convolutional Autoencoder (CAE) implementation
- Variational Autoencoder (VAE) implementation
- Denoising Autoencoder implementation
- Lightweight CNN classifier for supervised baseline
- Training utilities with early stopping and checkpointing
- Comprehensive visualization module
- Jupyter notebooks for training and analysis
- Complete research codebase for thesis
- Documentation and README
