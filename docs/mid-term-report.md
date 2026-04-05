# Mid-Term Check Report
# Research on Industrial Defect Detection Methods Based on Deep Learning

---

**Student Name:** Mohammad Hamim
**Student ID:** [Student ID]
**Major:** Computer Science and Technology
**Supervisor:** Lu Yang (卢洋)
**Department:** School of Computer Science and Artificial Intelligence
**University:** Zhengzhou University
**Report Date:** April 2026

---

## 1. Interim Results

### 1.1 Work Completed to Date

Since the thesis proposal was approved in early January 2026, the following work has been completed according to the planned schedule:

#### 1.1.1 Data Pipeline & Environment Setup

The full experimental environment has been configured and verified:

- **Python 3.12** + **PyTorch 2.0+** environment established on CPU-primary hardware (Intel Core i5)
- Three industrial datasets acquired and integrated:

  | Dataset | Purpose | Volume | Classes |
  |---------|---------|--------|---------|
  | MVTec Anomaly Detection (MVTec AD) | Primary benchmark | 5,354 images, 15 categories | Normal / Anomalous |
  | KolektorSDD2 | Cross-dataset generalization | 356 images | Surface defects |
  | NEU Surface Defect | Supervised classification baseline | 1,800 images | 6 defect types |

- Custom `DataLoader` modules implemented for all three datasets  
- Unified preprocessing pipeline: resize to **256×256**, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), conversion to PyTorch tensors

#### 1.1.2 Model Implementation (All 4 Models Complete)

All four target models have been fully implemented (`src/models/`):

**Convolutional Autoencoder (CAE)** — `cae.py`
- Encoder: 4 strided Conv2d blocks (channels: 3→32→64→128→256), each with BatchNorm + ReLU
- Spatial latent bottleneck: **16×16×256 = 65,536 dimensions**
- Decoder: 4 ConvTranspose2d blocks (256→128→64→32→3) with output_padding=1
- Total parameters: **2,764,099**

**Variational Autoencoder (VAE)** — `vae.py`
- Encoder with identical conv structure, but outputs two FC branches: `fc_mu` and `fc_logvar` (both → 128-dim)
- Dense latent space: **128 dimensions** (via reparameterization trick: z = μ + σ·ε)
- Numerical stability via logvar clamping: `clamp(logvar, -20, 2)`
- Loss: L = MSE + β·KL-divergence (KL annealed 0→1 over 10 epochs)
- Total parameters: **26,009,603**

**Denoising Autoencoder (DAE)** — `denoising_ae.py`
- Shares the CAE encoder-decoder backbone
- Training input corrupted with Gaussian noise: `x_noisy = x + N(0, σ²)`, where `σ = 0.3`
- Reconstructs clean image from noisy input; learns noise-invariant features
- At inference: clean pass (no noise added)
- Total parameters: **2,764,099**

**Lightweight CNN Classifier** — `cnn_classifier.py`
- 4 conv blocks (Conv2d + BatchNorm + ReLU + MaxPool2d), channels: [32, 64, 128, 256]
- Global Average Pooling → Dropout(0.5) → Linear(256→6)
- Total parameters: **11,177,030**

#### 1.1.3 Training Results

All models have been training across all 15 MVTec AD categories (Notebooks: `01_train_cae.ipynb`, `02_train_vae_v2.ipynb`, `03_train_denoising_ae.ipynb`, `04_train_cnn_classifier.ipynb`).

**Training configuration (common):**

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 1e-4 (Adam) |
| Max Epochs | 100 |
| Early Stopping | patience = 10 |
| Loss (CAE/DAE) | MSE |
| Loss (VAE) | MSE + β·KL |
| Device | CPU (Intel Core i5) |

#### 1.1.4 Evaluation Results (Quantitative)

**Image-Level Performance on MVTec AD (Means over 15 categories):**

| Model | Image AUC | Avg Precision | Precision | Recall | F1 Score |
|-------|-----------|---------------|-----------|--------|----------|
| CAE | 0.580 | 0.796 | 0.757 | 0.982 | 0.849 |
| VAE | 0.412 | 0.706 | 0.720 | 0.990 | 0.822 |
| **DAE** | **0.596** | **0.813** | **0.762** | **0.995** | **0.854** |

**Pixel-Level Localization Performance:**

| Model | Pixel AUC | Mean IoU | Mean Dice |
|-------|-----------|----------|-----------|
| **CAE** | **0.618** | 0.011 | 0.020 |
| VAE | 0.524 | 0.025 | 0.044 |
| DAE | 0.595 | 0.012 | 0.022 |

**Per-Category Image AUC Breakdown:**

| Category | CAE | VAE | DAE | Best |
|----------|-----|-----|-----|------|
| Bottle | 0.550 | 0.199 | 0.537 | CAE |
| Cable | 0.458 | 0.361 | 0.464 | DAE |
| Capsule | 0.477 | 0.482 | 0.466 | VAE |
| Carpet | 0.330 | **0.617** | 0.332 | **VAE** |
| Grid | 0.779 | 0.297 | **0.870** | **DAE** |
| Hazelnut | 0.877 | 0.255 | **0.888** | **DAE** |
| Leather | 0.447 | 0.303 | 0.389 | CAE |
| Metal Nut | 0.268 | 0.152 | 0.268 | CAE/DAE |
| Pill | 0.751 | 0.601 | **0.762** | **DAE** |
| Screw | 0.979 | 0.074* | **0.986** | **DAE** |
| Tile | **0.822** | 0.569 | 0.808 | CAE |
| Toothbrush | 0.656 | **0.686** | 0.650 | **VAE** |
| Transistor | 0.403 | 0.303 | **0.445** | **DAE** |
| Wood | 0.948 | **0.804** | **0.962** | **DAE** |
| Zipper | **0.506** | 0.480 | 0.487 | **CAE** |
| **Mean** | **0.580** | 0.412 | **0.596** | **DAE** |

*\*VAE on Screw (0.074 AUC) is a clear outlier caused by KL divergence collapse — addressed in Section 2.*

**Cross-Dataset Generalization (MVTec → KolektorSDD2):**

| Trained On (MVTec) | CAE AUC | VAE AUC | DAE AUC |
|--------------------|---------|---------|---------|
| Grid | **0.690** | 0.574 | 0.688 |
| Leather | 0.668 | 0.463 | 0.646 |
| Carpet | 0.665 | 0.545 | 0.682 |
| Wood | 0.662 | 0.493 | 0.652 |
| Tile | 0.649 | 0.590 | 0.575 |
| Bottle | 0.637 | 0.496 | 0.609 |
| Metal Nut | 0.622 | 0.587 | 0.617 |

**Supervised CNN Classifier (NEU Surface Defect Dataset):**

| Metric | Value |
|--------|-------|
| Accuracy | **99%** |
| Dataset | NEU Surface Defect (6 classes, 1800 images) |
| Training Epochs | 50 |
| Classes | Crazing, Inclusion, Patches, Pitted, Rolled-in scale, Scratches |

![CNN Confusion Matrix](../outputs/figures/cnn_confusion_matrix.png)
*Figure 1: Confusion matrix for CNN classifier — near-perfect classification across all 6 defect types.*

![Model Comparison](../outputs/figures/evaluation_bar_comparison.png)
*Figure 2: Image AUC and F1 Score comparison across all MVTec AD categories.*

![Radar Chart](../outputs/figures/evaluation_radar.png)
*Figure 3: Multi-metric radar chart comparing CAE, VAE, and DAE.*

![Cross-Dataset ROC](../outputs/figures/cross_dataset_roc.png)
*Figure 4: ROC curves for cross-dataset evaluation (MVTec → KolektorSDD2).*

![Reconstructions](../outputs/figures/thesis_fig4_reconstructions.png)
*Figure 5: Reconstruction examples — input image, model output, and pixel-level error map.*

#### 1.1.5 Web Application (Completed)

A full-stack deployment platform has been developed and is fully functional:

**Backend (FastAPI 0.109.0 + PyTorch):**
- RESTful API with endpoints: `/predict/autoencoder`, `/predict/cnn`, `/health`
- Handles image upload, model inference, and returns heatmap (base64) + anomaly score
- Supports all 4 models via runtime model-loading

**Frontend (Next.js 16.1.3 + React 19 + Tailwind CSS v4):**
- Image upload with drag-and-drop (`react-dropzone`)
- Real-time heatmap overlay visualization
- Side-by-side model comparison (CAE vs. VAE vs. DAE)
- Detection history with filtering and export
- Dark/Light mode, Framer Motion animations

![Web App Homepage](../web/frontend/public/homepage.png)
*Figure 6: Web application homepage.*

![Web App Detection](../web/frontend/public/detectpage.png)
*Figure 7: Detection interface with heatmap overlay output.*

---

## 2. Existing Problems

### 2.1 VAE Training Instability (Partially Resolved)

**Problem:** The Variational Autoencoder exhibits severe training instability on several MVTec categories. The most notable case is the **Screw** category, where VAE achieves only **0.074 AUC** — far below chance level — due to KL divergence collapse (the encoder pushes the variance to near-zero, causing the decoder to ignore the latent code entirely).

**Evidence:**
- VAE mean AUC (0.412) is ~30% lower than CAE (0.580) and DAE (0.596)
- VAE shows negative transfer on texture categories (Carpet: 0.617 vs. CAE: 0.330 — here VAE is unexpectedly *better*, suggesting the latent space occasionally encodes useful global texture statistics)

**Mitigation applied:**
- `logvar` clamping: `torch.clamp(logvar, min=-20, max=2)`
- KL annealing: β linearly increases from 0 → 1 over the first 10 epochs
- Gradient clipping during VAE training

**Remaining issue:** These fixes stabilize training but do not fully recover performance. The fundamental tension between KL regularization (forcing the posterior toward N(0,I)) and reconstruction fidelity remains unresolved at the current β=1.0 setting.

### 2.2 Low Pixel-Level Localization Accuracy

**Problem:** Despite reasonable image-level detection (AUC ~0.60), pixel-level segmentation quality is poor: mean IoU ≈ 0.011–0.025 and mean Dice ≈ 0.02–0.04 across all models.

**Root Cause:** Autoencoders trained with MSE pixel-wise loss tend to produce globally smoothed reconstructions. The error maps (E = (x − x̂)²) are noisy and diffuse, failing to produce sharp, mask-like boundaries around defect regions.

**Impact:** The anomaly heatmaps in the web application provide approximate defect localization, but do not achieve the precision needed for automated rejection systems in production.

### 2.3 Category-Specific Performance Degradation

**Problem:** Performance varies dramatically across categories. Some categories (Metal Nut: AUC ≈ 0.268; Transistor: AUC ≈ 0.403) are far below acceptable industrial thresholds (~0.75 AUC), even for the best model (DAE).

**Analysis:**
- **Complex viewpoint variation** (Transistor, Metal Nut): Objects appear in varied orientations, making "normal" reconstruction boundaries ill-defined.
- **Fine-grained texture defects** (Carpet, Leather): Defects (threads, tears) blend into the background texture, making reconstruction error small even at defect sites.
- **Uniform object categories** (Screw, Wood): These categories achieve high AUC (0.95+) because the "normal" appearance is highly constrained.

### 2.4 Computational Scalability

**Problem:** Training one model per MVTec category × 3 architectures = **45 separate training runs**. On CPU, each category requires approximately 7–12 minutes per model, totaling ~40 hours of training time. This limits rapid iteration.

**Impact:** Hyperparameter tuning (e.g., exploring different β values for VAE, noise levels for DAE) is time-consuming without GPU access.

### 2.5 Inferior Performance vs. State-of-the-Art

**Problem:** Current results significantly lag behind state-of-the-art methods on MVTec AD:

| Method | MVTec Image AUC | Architecture Type |
|--------|-----------------|--------------------|
| PatchCore (CVPR 2022) | 0.993 | Memory-based + pretrained ResNet |
| DRAEM (ICCV 2021) | 0.980 | Reconstructive discriminative |
| PaDiM (ICPR 2021) | 0.950 | Patch distribution (pretrained) |
| SimpleNet (CVPR 2023) | 0.990 | Pretrained feature + linear |
| **This work – DAE** | **0.596** | Autoencoder (from scratch, CPU) |
| **This work – CAE** | **0.580** | Autoencoder (from scratch, CPU) |

**Root cause:** The performance gap is attributable to:
1. No use of ImageNet-pretrained encoders (features learned from scratch)
2. No memory bank of normal features (PatchCore's key advantage)
3. Simple MSE loss without structural similarity components
4. Training without GPU acceleration limits dataset scale experimentation

---

## 3. Proposed Research Methods

The following improvements are planned to address the problems identified above, within the remaining thesis timeline (April–May 2026).

### 3.1 VAE Optimization: β-VAE Sweep and SSIM Loss

**Problem Targeted:** §2.1 — VAE instability and suboptimal performance.

**Proposed method:**

1. **β-VAE parameter sweep:** Train VAE with β ∈ {0.1, 0.5, 1.0, 2.0} on 3 representative categories (Bottle, Grid, Carpet) to find the optimal reconstruction-regularization balance. A lower β relaxes KL pressure, allowing the decoder to use more of the latent capacity for texture reconstruction.

2. **Hybrid MSE + SSIM Loss:** Replace pure MSE with a weighted combination:

   ```
   L_recon = α·MSE(x, x̂) + (1-α)·(1 - SSIM(x, x̂))
   ```

   SSIM captures structural and luminance similarity, which is more sensitive to perceptually meaningful defects than raw pixel differences. Proposed α = 0.85.

**Expected effect:** Improved VAE AUC on texture categories (Carpet, Leather) by 10–15%; reduced training instability.

### 3.2 Anomaly Map Post-Processing: Gaussian Smoothing + Otsu Thresholding

**Problem Targeted:** §2.2 — diffuse, low-quality pixel-level localization.

**Proposed method:**

1. Apply **Gaussian blur** (σ = 3–5 px) to raw error maps to suppress high-frequency noise while preserving defect regions.
2. Apply **Otsu's automatic thresholding** on the smoothed error map to produce binary anomaly masks without manual threshold selection.
3. Apply **morphological operations** (dilation, erosion) to close small gaps in detected defect contours.

```python
# Pseudocode
error_map = (image - reconstruction) ** 2
error_gray = error_map.mean(axis=0)
smoothed = gaussian_filter(error_gray, sigma=4)
threshold = threshold_otsu(smoothed)
binary_mask = smoothed > threshold
binary_mask = morphology.binary_closing(binary_mask, disk(3))
```

**Expected effect:** Improved pixel-level IoU from ~0.011 to ~0.04–0.08, making heatmaps sharper and more useful for operator visualization.

### 3.3 Per-Category Threshold Calibration

**Problem Targeted:** §2.3 — category-dependent performance degradation.

**Proposed method:**

Currently, a global threshold is applied. Instead, compute a **per-category optimal threshold** by:
1. Running the evaluation on the MVTec training split (normal images only) to establish the normal score distribution.
2. Setting the anomaly threshold at `μ_normal + k·σ_normal`, where k is tuned by maximizing F1 on a held-out validation set per category.
3. Store per-category thresholds in a calibration JSON file, loaded by the inference API.

**Expected effect:** Reduced false positive rate for high-variance categories (Carpet, Leather); improved precision from ~0.76 to ~0.82.

### 3.4 Multi-Scale Feature Fusion for Improved Localization

**Problem Targeted:** §2.2, §2.3 — blurry reconstruction and poor localization.

**Proposed method:**

Introduce **skip connections** in the decoder (U-Net style) to preserve fine-grained spatial detail across encoder resolutions:

```
Encoder: [256×256] → [128×128] → [64×64] → [32×32] → [16×16 latent]
                                                       ↓  skip connections
Decoder: [16×16] → [32×32]+e3 → [64×64]+e2 → [128×128]+e1 → [256×256]
```

This hybrid CAE-UNet architecture allows the decoder to access fine-grained features directly from intermediate encoder layers, preventing the "blurry averages" problem inherent in standard autoencoders.

**Expected effect:** Sharper error maps; improved pixel AUC from ~0.618 to ~0.68; better IoU scores.

### 3.5 Thesis Writing and Final Documentation

Concurrently with the above experiments:
- Complete all 6 chapters of `docs/thesis_paper.md` with final quantitative results
- Generate all remaining figures (`evaluation_heatmap.png`, cross-dataset bar charts)
- Conduct plagiarism check and submit draft (deadline: May 15, 2026)
- Finalize web application with improved inference pipeline

---

## 4. Feasibility Analysis

### 4.1 Technical Feasibility

| Task | Assessment | Justification |
|------|------------|---------------|
| β-VAE sweep (§3.1) | ✅ **High** | Only requires re-running existing training notebooks with modified β; estimated 2–3 days |
| SSIM loss integration (§3.1) | ✅ **High** | `pytorch-msssim` library available; straightforward loss function replacement |
| Gaussian + Otsu post-processing (§3.2) | ✅ **High** | `scipy.ndimage` and `skimage.filters` are standard; no retraining needed — inference-time change only |
| Per-category threshold calibration (§3.3) | ✅ **High** | Threshold computation is a post-training step; no retraining needed; computation takes minutes |
| Skip-connection CAE-UNet (§3.4) | ⚠️ **Medium** | Requires architecture change and full retraining across all categories; ~15 hours on CPU; risk of limited remaining time |

**Assessment:** Methods 3.1–3.3 are low-risk, high-impact improvements achievable within 1–2 weeks without retraining all models. Method 3.4 is optional/stretch goal.

### 4.2 Resource Feasibility

| Resource | Status | Notes |
|----------|--------|-------|
| Compute (CPU) | ✅ Available | Intel Core i5; sufficient for retraining VAE variants and evaluation runs |
| Datasets | ✅ Downloaded | MVTec AD, Kolektor, NEU all acquired and preprocessed |
| Codebase | ✅ Complete | All models, training loops, evaluation module, and web app are operational |
| Libraries | ✅ Installed | PyTorch 2.0+, scikit-learn, scikit-image, scipy, FastAPI, Next.js all configured |
| Time | ✅ Sufficient | 6 weeks remaining before draft deadline (May 15, 2026); proposed changes require ~2–3 weeks |

### 4.3 Schedule Feasibility

The current status is **on-track** with the original research schedule:

| Phase | Planned | Status |
|-------|---------|--------|
| Dataset acquisition & preprocessing | Jan–Feb 2026 | ✅ Complete |
| Model implementation (CAE, VAE, DAE, CNN) | Feb–Mar 2026 | ✅ Complete |
| Training across all categories | Mar–Apr 2026 | ✅ Complete |
| Evaluation & metrics | Mar–Apr 2026 | ✅ Complete |
| Cross-dataset generalization | Mar–Apr 2026 | ✅ Complete |
| Web application | Mar–Apr 2026 | ✅ Complete |
| Proposed improvements (§3.1–3.3) | Apr–May 2026 | 🔄 In Progress |
| Thesis draft | Apr–May 2026 | 🔄 In Progress |
| Final submission | May 29, 2026 | ⏳ On Track |

All **critical deliverables** (models trained, results collected, web app functional) have been completed ahead of the mid-term checkpoint. The proposed methods in Section 3 are refinements within the remaining time budget.

### 4.4 Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Skip-connection retraining takes too long | Medium | Medium | Mark as optional; proceed without it if time-constrained |
| SSIM loss worsens VAE results on some categories | Low | Low | Keep MSE as fallback; compare on 3 categories before full rollout |
| Per-category thresholds overfit to validation split | Low | Low | Use k-fold cross-validation on training set |
| Thesis plagiarism check failure | Very Low | High | All content is original research; auto-generated figures are unique |

### 4.5 Overall Feasibility Conclusion

The research is **fully feasible** within the remaining timeline. The core experimental work is complete. The proposed improvements (§3.1–3.3) are incremental optimizations requiring no significant infrastructure changes. The main remaining effort is:

1. Running β-VAE and SSIM loss experiments (~1 week)
2. Integrating post-processing pipeline into inference module (~3 days)
3. Completing and polishing the thesis document (~3 weeks)

The web application is deployment-ready and serves as a functional demonstration of the research-to-production workflow.

---

## Summary Table

| Dimension | Current State |
|-----------|--------------|
| Models trained | CAE, VAE, DAE (all 15 MVTec categories) + CNN (NEU) |
| Best anomaly detection AUC | **0.596** (DAE, mean over 15 categories) |
| Best supervised accuracy | **99%** (CNN on NEU) |
| Best generalization AUC | **0.690** (CAE trained on Grid → KolektorSDD2) |
| Web app status | ✅ Fully functional |
| Thesis document | 🔄 Draft in progress (~90% complete) |
| Planned improvements | β-VAE sweep, SSIM loss, post-processing, threshold calibration |
| Timeline status | ✅ On track for May 15 draft deadline |

---

*Prepared by: Mohammad Hamim*
*Submission: April 10, 2026 (Mid-Term Check Report Deadline)*
