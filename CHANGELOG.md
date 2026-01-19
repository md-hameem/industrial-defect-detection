# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
