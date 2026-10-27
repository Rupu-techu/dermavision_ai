# DermaVision AI — Module 5 (Part A: Generative Adversarial Networks)

**Synthetic Skin Lesion Synthesis, Pigmentation Augmentation, and Conditional DCGAN Generation**

Module 5 introduces **Generative Artificial Intelligence** to DermaVision AI. Part A implements **Deep Convolutional Generative Adversarial Networks (DCGAN)** and **Conditional DCGAN (cDCGAN)** to generate realistic synthetic skin lesion images, directly addressing clinical dataset scarcity and severe class imbalance (such as rare Melanoma cases).

---

## 1. Executive Summary & Curriculum Alignment

| # | Task | Technique | Output / Artifact |
|:---:|:---|:---|:---|
| **1** | **Synthetic Skin Image Generation** | Deep Convolutional GAN (DCGAN) | `models/module5_gan_generator_best.pth`, `Module5_GAN_Synthetic_Samples.png` |
| **2** | **Adversarial Discriminator** | Convolutional Binary Classifier + LeakyReLU | `models/module5_gan_discriminator_best.pth`, `Module5_GAN_Loss_Curves.png` |
| **3** | **Targeted Condition Synthesis** | Conditional DCGAN (cDCGAN) with Class Embeddings | `models/module5_cgan_generator_best.pth` |
| **4** | **Curriculum Training Augmentation** | Automated Class Directory Augmentation Engine | `synthetic_skin_dataset/{ACK, BCC, MEL, NEV, SCC, SEK}/` |
| **5** | **Benchmark & Evaluation** | Minimax Loss Tracking & Summary Logging | `DermaVision_Module5_GAN_Results.csv` |

> **Medical Disclaimer:** DermaVision AI is an educational and research decision-support tool. Synthetic images are generated for machine learning training augmentation and research purposes, not for clinical diagnosis.

---

## 2. Theoretical Background

### A. The Challenge of Data Scarcity in Clinical Dermatology
Medical dermatology datasets (like PAD-UFES-20 and ISIC) suffer from extreme class imbalances. Benign conditions (such as Nevus or Seborrheic Keratosis) often outnumber malignant lesions (like Melanoma or Squamous Cell Carcinoma) by orders of magnitude. Training deep vision classifiers (ResNet, EfficientNet, YOLO) on imbalanced distributions leads to high false negative rates on life-threatening conditions.

### B. Generative Adversarial Networks (GANs) Formulation
A GAN frames generative modeling as a two-player zero-sum game between two neural networks:
1. **The Generator ($G$)**: Captures the true data distribution $p_{\text{data}}$ by mapping a prior noise vector $z \sim p_z(z)$ (where $p_z = \mathcal{N}(0, I)$) to realistic skin lesion images $G(z)$.
2. **The Discriminator ($D$)**: Evaluates whether an input image $x$ comes from real clinical data ($D(x) \to 1$) or is a synthetic forgery produced by $G$ ($D(G(z)) \to 0$).

The objective function is the minimax formulation:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]$$

### C. Deep Convolutional GAN (DCGAN) Architectural Guidelines
As established by Radford et al. and integrated with our Module 2 weight initialization principles:
- **Spatial Upsampling**: Replaces pooling layers with spatial fractional-strided convolutions (`ConvTranspose2d`) in the generator.
- **Spatial Downsampling**: Replaces max-pooling with strided convolutions (`Conv2d`) in the discriminator.
- **Batch Normalization**: Applied in both Generator and Discriminator to stabilize gradient flow and prevent mode collapse.
- **Activation Functions**:
  - Generator: `ReLU` in all hidden layers; `Tanh` on the output layer (mapping pixel values to $[-1, 1]$).
  - Discriminator: `LeakyReLU(0.2)` across all layers to prevent dead gradients; `Sigmoid` on the output unit.
- **Weight Initialization**: All convolutional and transposed convolutional weights initialized from $\mathcal{N}(0, 0.02)$, and batch normalization weights from $\mathcal{N}(1, 0.02)$.

### D. Conditional DCGAN (cDCGAN)
To allow targeted generation of specific dermatological diagnoses, class labels $y \in \{0, \dots, C-1\}$ are embedded into dense continuous representations:
- In the **Conditional Generator**, a class embedding $e(y) \in \mathbb{R}^{32}$ is concatenated directly to the latent noise vector $z \in \mathbb{R}^{100}$ before transposed convolutions.
- In the **Conditional Discriminator**, the class label is projected into a spatial feature map of dimension $1 \times 64 \times 64$ and concatenated as a 4th channel alongside the RGB image before downsampling convolutions.

---

## 3. End-to-End Pipeline & Integration with Modules 1–4

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│       MODULE 1 & MODULE 2       │      │            MODULE 3             │
│ • Diagnostic Classes:           │      │ • Extracted Lesion Patches      │
│   ACK, BCC, MEL, NEV, SCC, SEK  │      │ • U-Net Pigmentation Borders    │
│ • He / Xavier Weight Init       │      │ • Color & Contrast Statistics   │
└────────────────┬────────────────┘      └────────────────┬────────────────┘
                 │                                        │
                 └───────────────────┬────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │    Patch Extraction Engine    │
                     │  (64x64x3 Normalized RGB)     │
                     └───────────────┬───────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │   SkinLesionGenerator     │   │ SkinLesionDiscriminator   │
       │   z ~ N(0, I) in R^100    │   │ 64x64x3 Skin Patch        │
       │   Transposed Conv 4->64   │   │ Strided Conv 64->4        │
       └─────────────┬─────────────┘   └─────────────┬─────────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     │ Minimax BCE Loss
                                     ▼
                     ┌───────────────────────────────┐
                     │   Trained Generator Checkpoint │
                     │   models/module5_gan_best.pth │
                     └───────────────┬───────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │ Synthetic Lesion Gallery  │   │ Dataset Augmentation      │
       │ Module5_GAN_Samples.png   │   │ synthetic_skin_dataset/   │
       │ (High-Res 16-sample Grid) │   │ (Per-Class Balanced Data) │
       └───────────────────────────┘   └───────────────────────────┘
```

---

## 4. Hyperparameter Reference

| Component | Hyperparameter | Configured Value | Medical / Theoretical Rationale |
|:---|:---|:---:|:---|
| **Resolution** | Patch Size | $64 \times 64 \times 3$ | Standard dermatological patch scale for melanocytic lesions |
| **Latent Vector** | $z\_\text{dim}$ | $100$ | Standard Gaussian latent space for generative diversity |
| **Class Embedding** | $e\_\text{dim}$ | $32$ | Continuous semantic representation for 6 clinical classes |
| **Generator Filters** | Base Channels | $64$ (512 $\to$ 256 $\to$ 128 $\to$ 64 $\to$ 3) | Multi-scale progressive textural synthesis |
| **Discriminator Filters** | Base Channels | $64$ (3 $\to$ 64 $\to$ 128 $\to$ 256 $\to$ 512 $\to$ 1) | Hierarchical feature extraction for lesion boundaries |
| **Optimizers** | Algorithm & Hyperparams | Adam ($\text{lr}=0.0002, \beta_1=0.5, \beta_2=0.999$) | Radford et al. configuration preventing oscillation |
| **Label Smoothing** | Real Target Label | $0.9$ (One-sided smoothing) | Prevents discriminator overconfidence and gradient vanishing |

---

## 5. Artifacts & Deliverables Summary

| Artifact | Type | Description |
|:---|:---|:---|
| [`src/module5_gan.py`](file:///Users/pritimmondal/Projects/dermavision_ai/src/module5_gan.py) | Python Module | `SkinLesionGenerator`, `SkinLesionDiscriminator`, `ConditionalSkinGenerator`, and sampling engine |
| [`train_module5_gan.py`](file:///Users/pritimmondal/Projects/dermavision_ai/train_module5_gan.py) | Standalone Script | End-to-end training pipeline, patch extractor, and artifact exporter |
| [`tests/test_module5_gan.py`](file:///Users/pritimmondal/Projects/dermavision_ai/tests/test_module5_gan.py) | Test Suite | 8 automated unit tests covering shapes, ranges, conditioning, and disk export |
| `models/module5_gan_generator_best.pth` | PyTorch Weights | Best weights for standard DCGAN Generator |
| `models/module5_gan_discriminator_best.pth` | PyTorch Weights | Best weights for DCGAN Discriminator |
| `models/module5_cgan_generator_best.pth` | PyTorch Weights | Best weights for Conditional DCGAN Generator |
| `Module5_GAN_Synthetic_Samples.png` | Image Artifact | 16-sample high-resolution grid of synthesized skin lesions |
| `Module5_GAN_Loss_Curves.png` | Image Artifact | Discriminator vs. Generator adversarial loss curves across training |
| `synthetic_skin_dataset/` | Image Directory | Augmented clinical dataset partitioned into class folders for retraining |
| `DermaVision_Module5_GAN_Results.csv` | CSV Data | Quantitative benchmark summary table |
