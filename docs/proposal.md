# Thesis Proposal

**Title:** Research on Industrial Defect Detection Methods Based on Deep Learning  
**Author:** Mohammad Hamim  
**Supervisor:** Lu Yang  

---

## 1. Basis for Proposal (Current Status of Research at Home and Abroad)

### 1.1 Background and Industry Need
Industrial manufacturing is undergoing a profound transformation driven by Industry 4.0 paradigms. A critical component of this evolution is automated quality control. Traditional manual inspection methods suffer from subjectivity, fatigue, and an inability to scale to modern high-throughput production lines. Consequently, there is an urgent industry demand for fast, reliable, and automated surface defect detection systems to ensure product consistency and minimize economic losses.

### 1.2 The Class Imbalance Challenge
While supervised deep learning has achieved remarkable success in general computer vision tasks, its application in industrial defect detection faces a significant hurdle: the **class imbalance problem**. In real-world manufacturing, defective samples are extremely rare, their visual appearances are highly diverse (ranging from microscopic scratches to large morphological deformities), and they are unpredictable. Conversely, normal, defect-free samples are abundant. This makes the collection and annotation of large-scale defective datasets prohibitively expensive, rendering standard supervised classification approaches largely impractical for localized defect detection.

### 1.3 Current Research Landscape
To address this imbalance, the global academic and industrial research communities have pivoted heavily towards **unsupervised anomaly detection**. By training models exclusively on "normal" (defect-free) data, these systems learn the manifold of acceptable product variations. Any significant deviation during inference is flagged as an anomaly.

*   **Traditional Unsupervised Methods:** Early methods relied on hand-crafted features (SIFT, HOG), statistical modeling (Gaussian Mixture Models, One-class SVMs), and template matching techniques. While computationally light, these methods lack the representational power to handle complex, high-resolution industrial textures and dynamic factory lighting conditions.
*   **Deep Representation Learning:** Current state-of-the-art research leverages deep representation learning. Reconstruction-based methods, particularly Autoencoders (Convolutional Autoencoders, Variational Autoencoders) and Generative Adversarial Networks (GANs), are widely used. They attempt to compress and reconstruct the input; defective regions, unseen during training, reconstruct poorly, yielding high residual errors.
*   **Recent Advancements Abroad:** The most recent literature explores memory-augmented feature matching methods (e.g., PatchCore, PaDiM), which extract features using pre-trained ImageNet backbones and compare them against a "bank" of normal feature distributions. Normalizing flows (e.g., CFLOW-AD) are also actively researched for estimating the exact likelihood of visual features. While these models achieve near-perfect metrics (e.g., >0.98 AUC on the MVTec dataset), they often incur massive computational costs and memory footprints that hinder deployment on edge devices or standard factory hardware.

Therefore, a critical research gap remains: deeply understanding the trade-offs of lightweight, unsupervised autoencoder architectures, optimizing their latent space bottlenecks for texture vs. structural preservation, and establishing a complete, CPU-deployable pipeline from raw data to a user-facing interactive application.

## 2. Research Objectives and Significance

### 2.1 Research Objectives
This graduation thesis aims to systematically evaluate, optimize, and deploy autoencoder-based deep learning methods for industrial defect detection. The specific objectives are:

1.  **Architecture Implementation and Optimization:** Develop and meticulously compare three distinct unsupervised deep learning architectures: Convolutional Autoencoder (CAE), Variational Autoencoder (VAE), and Denoising Autoencoder (DAE). 
2.  **Benchmark Evaluation:** Rigorously evaluate these models on the MVTec Anomaly Detection (MVTec AD) dataset, which contains 15 diverse product categories (5 texture-based, 10 object-based), utilizing metrics such as Image-Level AUC, Pixel-Level AUC, F1 Score, and Average Precision.
3.  **Cross-Dataset Generalization Analysis:** Investigate the robustness and transferability of learned representations by evaluating models trained on MVTec directly on a real factory dataset (KolektorSDD2) without retraining.
4.  **System Deployment and Visualization:** Design and implement a full-stack, interactive web application (Next.js frontend, FastAPI/PyTorch backend) capable of processing image uploads and generating real-time algorithmic heatmaps to localize defects visually for operators.

### 2.2 Academic and Practical Significance
*   **Academic Value:** By systematically comparing a deterministic CAE against a probabilistic VAE, this research will provide empirical insights into the "information bottleneck" challenge. It will clarify why highly compressed representations (VAE) struggle with fine texture details compared to spatial bottlenecks (CAE), contributing to the theoretical understanding of autoencoder design for anomaly detection.
*   **Practical Value:** The reliance entirely on normal samples for training eliminates the costly data annotation bottleneck plaguing the industry. Furthermore, by focusing on CPU-optimized models and packaging them within a modern web application, this thesis provides a highly accessible, low-cost technological template that small-to-medium manufacturing enterprises can readily adopt for quality control.

## 3. Research Content and Expected Outcomes

### 3.1 Detailed Research Content

1.  **Data Processing Pipeline Construction:** 
    *   Acquire the MVTec AD, KolektorSDD2, and NEU Surface Defect datasets.
    *   Develop a robust preprocessing pipeline including dynamic resizing (256x256), ImageNet-standardized normalization, and data augmentation strategies (where appropriate for unsupervised learning). 
    *   Construct optimized PyTorch `DataLoader` modules for efficient batch processing.

2.  **Model Architecture Design and Implementation:**
    *   **CAE (Convolutional Autoencoder):** Design a baseline encoder-decoder with a structured spatial latent space (e.g., 16x16x256) designed to preserve high-resolution spatial relationships.
    *   **VAE (Variational Autoencoder):** Integrate a probabilistic latent distribution (e.g., 128-dimensional dense vector) optimized via the Reparameterization Trick and Kullback-Leibler (KL) divergence to understand the impact of extreme dimensional reduction on anomaly detection.
    *   **DAE (Denoising Autoencoder):** Introduce Gaussian noise injection techniques during training to force the network to learn invariant, robust features rather than trivial identity mappings.
    *   **CNN Baseline:** Implement a lightweight supervised CNN classifier on the NEU dataset to serve as an accuracy baseline for the unsupervised methods.

3.  **Training Regimen and Multi-Metric Evaluation:**
    *   Implement centralized training loops using continuous evaluation. Use Mean Squared Error (MSE) and structural similarity index (SSIM) losses.
    *   Develop an evaluation suite that generates pixel-wise error maps (E = (x - x̂)²) to produce intuitive anomaly heatmaps.
    *   Compute comprehensive metrics: Receiver Operating Characteristic Area Under the Curve (ROC-AUC), Precision-Recall curves, and optimal F1-score thresholding.

4.  **Full-Stack Application Development:**
    *   **Backend:** Build an asynchronous RESTful API using FastAPI (Python) to wrap the PyTorch inference engine, handling image decoding, forward passes, and heatmap base64 encoding.
    *   **Frontend:** Develop a responsive, modern web interface using Next.js 15, React 19, and Tailwind CSS. The UI will allow users to upload arbitrary images, select the inference model, and instantly view localized defect overlays.

### 3.2 Expected Outcomes
1.  **Quantitative Conclusions:** The research will identify the most effective architecture (hypothesized to be the DAE due to noise-induced robustness) for varying product types (textures vs. rigid objects). It will also quantify the zero-shot cross-dataset generalization capabilities.
2.  **Developed Software Asset:** A polished, fully functional, open-source web application repository combining a FastAPI ML inference server and a Next.js visualization dashboard, demonstrating a complete "research-to-production" lifecycle.
3.  **Comprehensive Thesis Document:** A detailed, academically rigorous graduation thesis (exceeding 50 pages) documenting the methodology, network parameters, and encompassing over 80 specific qualitative visualizations (heatmaps, reconstructions) and quantitative charts.

## 4. Research Schedule (Based on University Timeline)

| Timeframe | Task Description | Milestones & University Deadlines | Responsible Party |
| :--- | :--- | :--- | :--- |
| **Jan 2026** | Release of thesis topics, literature review, and environment setup. Acquire MVTec AD and configure PyTorch. | **Jan 16, 2026:** Release of thesis topics and submission of task sheets | Students, Supervisors |
| **Feb - Early Mar 2026** | Implementation of data pipelines and core Deep Learning architectures (CAE, VAE, DAE). | **Mar 6, 2026:** Upload of thesis proposals | Students, Supervisors |
| **Mar - Mid Apr 2026** | Extensive model training across all dataset categories. Hyperparameter tuning and generation of evaluation metrics. | **Apr 10, 2026:** Upload of mid-term check reports | Students, Supervisors |
| **Late Apr - Mid May 2026** | Development of Next.js frontend and FastAPI backend. Thesis drafting and figure generation. | **May 15, 2026:** Upload of thesis drafts and plagiarism check reports | Students, Supervisors |
| **Late May 2026** | Final UI polish, comprehensive thesis review, formatting, and presentation preparation. | **May 29, 2026:** Upload of final thesis versions and thesis grades | Students, Supervisors, Academic Admins |
| **Early Jun 2026** | Thesis evaluation panel, submission of training programs, and supervisor information forms. | **Jun 12, 2026:** Thesis evaluation and upload of all relevant thesis materials | Academic Administrators, Supervisors |
| **Mid Jun 2026** | Binding and final administrative clearance. | **Jun 18, 2026:** Archiving of graduation theses and related hard copies | Students, Academic Administrators |

## 5. Main References

1. Bergmann, P., et al. "MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." *CVPR* 2019.
2. Kingma, D. P., and Welling, M. "Auto-Encoding Variational Bayes." *ICLR* 2014.
3. Vincent, P., et al. "Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion." *JMLR* 2010.
4. Roth, K., et al. "Towards Total Recall in Industrial Anomaly Detection." *CVPR* 2022.
5. Zavrtanik, V., et al. "DRAEM—A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection." *ICCV* 2021.
6. Defard, T., et al. "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization." *ICPR* 2021.
7. Gudovskiy, D., et al. "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows." *WACV* 2022.
8. He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR* 2016.

