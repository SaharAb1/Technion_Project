<p align="center">
  <img src="assets/pipeline.png" alt="Pipeline" width="100%"/>
</p>

<h1 align="center">CNN vs. Vision Transformer for Medical Image Classification</h1>

<p align="center">
  <b>Deep Learning Final Project — Technion</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/timm-1.0+-blueviolet" alt="timm"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Task-Medical_Imaging-critical" alt="Task"/>
</p>

<p align="center">
  A rigorous comparative study of <b>ResNet-50</b> (CNN) and <b>ViT-B/16</b> (Vision Transformer) for binary classification of chest X-rays as <b>Normal</b> or <b>Pneumonia</b>, with <b>Grad-CAM</b> interpretability analysis.
</p>

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Results](#results)
  - [Training Curves](#training-curves)
  - [Test Performance](#test-performance)
  - [Confusion Matrices](#confusion-matrices)
  - [Model Complexity](#model-complexity)
- [Grad-CAM Interpretability](#grad-cam-interpretability)
- [Key Findings](#key-findings)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [References](#references)

---

## Motivation

Medical image classification demands models that are not only **accurate** but also **interpretable** — clinicians need to understand *why* a model makes a prediction to trust its output. This project explores two fundamentally different paradigms:

| | **CNNs** | **Vision Transformers** |
|---|---|---|
| **How they see** | Local receptive fields that grow hierarchically | Global self-attention over all image patches |
| **Inductive bias** | Strong (locality, translation equivariance) | Minimal (learns structure from data) |
| **Best for** | Smaller datasets, strong spatial priors | Large datasets, capturing long-range dependencies |

> **Research Question:** *Can a Vision Transformer match or outperform a CNN on a small-to-medium medical imaging dataset using transfer learning alone?*

---

## Architecture Overview

<p align="center">
  <img src="assets/architecture_comparison.png" alt="Architecture Comparison" width="100%"/>
</p>

### ResNet-50 (CNN)
- **Backbone:** 50-layer deep residual network with skip connections
- **Transfer learning:** ImageNet pre-trained, backbone **frozen**
- **Classification head:** `Global Avg Pool → Dropout(0.3) → Linear(2048, 2)`
- **Parameters:** ~23.5M total, ~4K trainable

### ViT-B/16 (Vision Transformer)
- **Backbone:** 12-layer transformer encoder, 16x16 patch embedding, 12 attention heads
- **Transfer learning:** ImageNet pre-trained via `timm`, backbone **frozen**
- **Classification head:** `[CLS] token → Linear(768, 2)`
- **Parameters:** ~86.6M total, ~1.5K trainable

---

## Dataset

**[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** — a widely used benchmark from Kaggle.

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| **Train** | 1,341 | 3,875 | **5,216** |
| **Validation** | 8 | 8 | **16** |
| **Test** | 234 | 390 | **624** |

### Preprocessing & Augmentation

| Stage | Transforms |
|-------|-----------|
| **Training** | Resize(224) → RandomHorizontalFlip → RandomRotation(10°) → ColorJitter → Normalize(ImageNet) |
| **Evaluation** | Resize(224) → Normalize(ImageNet) |

---

## Pipeline

<p align="center">
  <img src="assets/pipeline.png" alt="End-to-End Pipeline" width="100%"/>
</p>

**Training Configuration:**

```
Optimizer:      AdamW (lr=1e-4, weight_decay=1e-4)
Loss:           CrossEntropyLoss
Scheduler:      CosineAnnealingLR (T_max=10)
Epochs:         10
Batch Size:     32
Checkpointing:  Best model by validation accuracy
```

---

## Results

### Training Curves

<p align="center">
  <img src="assets/training_curves.png" alt="Training Curves" width="100%"/>
</p>

Both models converge within 10 epochs. ViT-B/16 trains faster and achieves lower final loss, while ResNet-50 shows a more gradual convergence. The gap between train and validation curves reflects the extremely small validation set (16 samples), which introduces noise in validation metrics.

### Test Performance

<p align="center">
  <img src="assets/metric_comparison.png" alt="Metric Comparison" width="85%"/>
</p>

| Model | Accuracy | F1-Score (weighted) | Loss |
|-------|----------|---------------------|------|
| **ResNet-50** | 83.17% | 82.81% | 0.394 |
| **ViT-B/16** | 82.85% | 81.83% | 0.386 |

Both architectures achieve comparable overall accuracy (~83%) using **only classification head fine-tuning** (~2-4K trainable parameters out of millions). The per-class breakdown reveals important differences:

| Model | | Precision | Recall | F1-Score | Support |
|-------|---------|-----------|--------|----------|---------|
| **ResNet-50** | NORMAL | 0.83 | 0.69 | 0.76 | 234 |
| | PNEUMONIA | 0.83 | 0.92 | 0.87 | 390 |
| **ViT-B/16** | NORMAL | 0.92 | 0.59 | 0.72 | 234 |
| | PNEUMONIA | 0.80 | 0.97 | 0.88 | 390 |

> **Key insight:** ViT-B/16 is more **conservative** — it achieves 92% precision on NORMAL (few false "healthy" labels) and 97% recall on PNEUMONIA (misses almost no disease cases). ResNet-50 is more **balanced** across classes. For clinical screening where missing pneumonia is dangerous, ViT's high pneumonia recall (97%) is particularly valuable.

### Confusion Matrices

<p align="center">
  <img src="assets/confusion_matrices.png" alt="Confusion Matrices" width="85%"/>
</p>

Key observations:
- **ViT-B/16 misses only ~3% of pneumonia cases** (12 false negatives out of 390) — critical for screening
- **ResNet-50 is more balanced**, with 69% recall on NORMAL vs ViT's 59%
- Both models show a bias toward predicting PNEUMONIA, reflecting the **3:1 class imbalance** in the training set (3,875 pneumonia vs 1,341 normal)

### Model Complexity

<p align="center">
  <img src="assets/model_complexity.png" alt="Model Complexity" width="85%"/>
</p>

| | ResNet-50 | ViT-B/16 |
|---|-----------|----------|
| **Total parameters** | 23.5M | 85.8M |
| **Trainable parameters** | 4,098 | 1,538 |
| **Trainable ratio** | 0.017% | 0.002% |

ResNet-50 is **3.7x more parameter-efficient** overall. Interestingly, ViT achieves comparable performance with even fewer trainable parameters (1.5K vs 4K), suggesting its pre-trained representations transfer more efficiently to this task.

---

## Grad-CAM Interpretability

<p align="center">
  <img src="assets/gradcam_example.png" alt="Grad-CAM Visualization" width="100%"/>
</p>

**Gradient-weighted Class Activation Mapping** reveals where the model "looks" when making predictions:

- **Pneumonia cases:** Heatmaps concentrate over areas of **opacity and consolidation** in the lung fields — consistent with clinical radiological findings
- **Normal cases:** Activations are **diffuse** across both lung fields, confirming the model evaluates the entire chest region
- **Clinical value:** Grad-CAM overlays could serve as a **"second opinion"** highlighting regions of concern for radiologists

### How Grad-CAM Works

```
1. Forward pass through ResNet → class score
2. Backpropagate gradients to layer4 feature maps
3. Global average pool the gradients → channel importance weights
4. Weighted sum of feature maps → ReLU → heatmap
5. Upsample heatmap to input resolution → overlay on X-ray
```

---

## Key Findings

### 1. Transfer Learning Is Remarkably Effective
With frozen ImageNet backbones and only classification heads fine-tuned (~2-4K parameters), both architectures achieve **~83% accuracy** on chest X-ray classification. This demonstrates that ImageNet features transfer well even to grayscale medical images — with only 0.002-0.017% of parameters trained.

### 2. Different Architectures, Different Clinical Profiles
While overall accuracy is similar, the models exhibit **distinct error profiles**:
- **ViT-B/16** strongly favors pneumonia detection (97% recall) at the cost of over-diagnosing normal cases — ideal for **screening** where missed disease is unacceptable
- **ResNet-50** is more balanced across classes — better suited for **diagnostic** contexts where both false positives and false negatives matter

### 3. CNN Wins on Efficiency
ResNet-50 achieves comparable accuracy with **3.7x fewer total parameters**, making it the practical choice for deployment in resource-constrained clinical environments (edge devices, hospital systems).

### 4. ViT Matches CNN Despite Small Data
Conventionally, transformers need large datasets to outperform CNNs. Here, ViT-B/16 **matches ResNet-50** on ~5K training images — entirely due to strong ImageNet pre-training bridging the data gap.

### 5. Interpretability is Non-Negotiable
Grad-CAM demonstrates that ResNet-50 attends to **clinically meaningful lung regions**. For real-world medical AI deployment, this interpretability is as important as raw accuracy metrics.

### 6. Dataset Limitations Shape Results
The training set has a **3:1 class imbalance** (pneumonia vs normal) and the validation set contains only **16 samples** — both of which influence model behavior. These are well-known limitations of this benchmark dataset and highlight the importance of careful dataset curation in medical AI.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/SaharAb1/Technion_Project.git
cd Technion_Project
pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm
```

### 2. Download Dataset

Download the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it:

```
chest_xray/
  train/    -> NORMAL/, PNEUMONIA/
  val/      -> NORMAL/, PNEUMONIA/
  test/     -> NORMAL/, PNEUMONIA/
```

### 3. Run the Notebook

```bash
jupyter notebook CNN_vs_ViT_Medical_Image_Classification.ipynb
```

Or upload directly to **Google Colab** — update `DATA_DIR` in cell 3 and run all cells.

### 4. (Optional) Regenerate README Assets

```bash
python assets/generate_readme_assets.py
```

---

## Project Structure

```
Technion_Project/
├── CNN_vs_ViT_Medical_Image_Classification.ipynb   # Main experiment notebook
├── README.md                                        # This file
├── assets/
│   ├── CNN_vs_ViT_Medical_Image_Classification (1).ipynb  # Executed notebook with outputs
│   ├── generate_readme_assets.py                    # Script to regenerate charts
│   ├── architecture_comparison.png                  # Architecture diagram
│   ├── training_curves.png                          # Loss & accuracy curves
│   ├── confusion_matrices.png                       # Confusion matrix heatmaps
│   ├── metric_comparison.png                        # Bar chart comparison
│   ├── model_complexity.png                         # Parameter analysis
│   ├── pipeline.png                                 # End-to-end pipeline
│   └── gradcam_example.png                          # Grad-CAM visualization
└── tests/
    └── test_notebook.py                             # Code validation tests
```

---

## References

1. **He et al.** (2016). *Deep Residual Learning for Image Recognition.* CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
2. **Dosovitskiy et al.** (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
3. **Selvaraju et al.** (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
4. **Kermany et al.** (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell.
5. **Wightman, R.** (2019). *PyTorch Image Models (timm).* [GitHub](https://github.com/huggingface/pytorch-image-models)

---

<p align="center">
  <sub>Built with PyTorch | Technion Deep Learning Course</sub>
</p>
