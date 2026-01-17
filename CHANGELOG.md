# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
