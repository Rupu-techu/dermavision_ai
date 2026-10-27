# DermaVision AI — Module 5 (Part B: Variational Autoencoder & Clinical Similarity Search)

**Compact Latent Representation Learning, 2D Latent Manifold Visualization, and Case-Based Retrieval Engine**

Module 5 Part B implements a **Convolutional Variational Autoencoder (ConvVAE)** for DermaVision AI. It learns a continuous, low-dimensional representation ($z \in \mathbb{R}^{32}$) of skin lesions, enabling **Clinical Case Similarity Search**, **t-SNE / PCA Latent Clustering**, **Morphological Latent Traversal**, and **Anomaly Detection**.

---

## 1. Executive Summary & Curriculum Alignment

| # | Task | Technique | Output / Artifact |
|:---:|:---|:---|:---|
| **1** | **Representation Learning** | 4-Layer Convolutional VAE with Reparameterization | `models/module5_vae_best.pth` |
| **2** | **Clinical Case Similarity Search** | Cosine & Euclidean Nearest-Neighbor Retrieval | `models/module5_vae_case_gallery.pt` |
| **3** | **Morphological Reconstruction** | Convolutional Transposed Decoder + MSE Loss | `Module5_VAE_Reconstruction_Grid.png` |
| **4** | **Latent Space Visualization** | 2D t-SNE Projection colored by 6 Clinical Diagnoses | `Module5_VAE_Latent_Space_tSNE.png` |
| **5** | **Latent Manifold Traversal** | Linear Latent Interpolation ($z_1 \to z_2$) | `Module5_VAE_Latent_Interpolation.png` |
| **6** | **Quantitative Evaluation** | Loss & Gallery Indexing Summary | `DermaVision_Module5_VAE_Results.csv` |

> **Medical Disclaimer:** DermaVision AI is an educational and clinical decision-support research tool. Nearest-neighbor case retrieval provides historical case context for dermatologists, not automated diagnosis.

---

## 2. Theoretical Background

### A. The Need for Representation Learning in Clinical Dermatology
High-resolution skin lesion images contain redundant spatial information (lighting gradients, background healthy skin, surface glare). Diagnostically critical information—such as melanin distribution, pigment network asymmetry, border irregularity, and atypical color variegation—can be effectively captured in a compact, continuous latent space $z \in \mathbb{R}^{d}$.

### B. Variational Autoencoders (VAEs) vs. Standard Autoencoders
Standard autoencoders map inputs to deterministic point vectors, resulting in irregular latent spaces with gaps where decoded images appear nonsensical. In contrast, a Variational Autoencoder maps each image $x$ to a probability distribution $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ over the latent space, regularized by a standard Gaussian prior $p(z) = \mathcal{N}(0, I)$.

### C. Reparameterization Trick
To backpropagate through the stochastic bottleneck during training, the sampling step is reparameterized as:
$$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
Where $\sigma(x) = \exp(0.5 \times \log \sigma^2(x))$. This isolates stochasticity to $\epsilon$, allowing gradients $\nabla_\phi$ to flow directly through $\mu$ and $\log \sigma^2$.

### D. Objective Function: $\beta$-VAE Formulation
The loss function maximizes the Evidence Lower Bound (ELBO):
$$\mathcal{L}_{\text{VAE}}(x; \theta, \phi) = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \parallel p(z))$$
1. **Reconstruction Loss**: Measures fidelity between input skin lesion $x$ and reconstructed output $\hat{x} = \text{Dec}(z)$:
   $$\mathcal{L}_{\text{recon}} = \frac{1}{N} \|x - \hat{x}\|_2^2$$
2. **Kullback-Leibler (KL) Divergence**: Enforces a continuous Gaussian prior, preventing cluster collapse:
   $$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
3. **$\beta$ Hyperparameter**: Balances disentangled representation learning ($\beta=0.5$) with high-resolution reconstruction.

---

## 3. Clinical Case Similarity Search Engine

### Case-Based Reasoning (CBR)
Dermatologists frequently confirm difficult diagnoses by comparing a current patient's lesion against previously confirmed histological cases. The `SkinCaseSimilarityEngine` operates as follows:

```
                          ┌───────────────────────────┐
                          │   Query Patient Lesion    │
                          │        (64x64x3)          │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │      VAE Encoder          │
                          │   z* = mu(x_query)        │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
           ┌───────────────────────────┐ ┌───────────────────────────┐
           │   Reconstruction Error    │ │  Cosine Similarity Search │
           │   ||x - x_recon||^2       │ │  S(z*, z_gallery)         │
           ├───────────────────────────┤ ├───────────────────────────┤
           │ Atypical Anomaly Score    │ │ Top-3 Matched Cases       │
           │ (High for Malignancies)   │ │ • Histological Diagnosis  │
           │                           │ │ • Biopsy Confirmation     │
           │                           │ │ • Prior Treatment Outcome │
           └───────────────────────────┘ └───────────────────────────┘
```

1. **Latent Query Projection**: Given a query image, the encoder computes deterministic mean vector $\mu^* \in \mathbb{R}^{32}$.
2. **Cosine Similarity**:
   $$S_C(\mu^*, \mu_{\text{case}}) = \frac{\mu^* \cdot \mu_{\text{case}}}{\|\mu^*\|_2 \|\mu_{\text{case}}\|_2} \times 100\%$$
3. **Anomaly / Atypicality Scoring**: Reconstructs $\hat{x}^* = \text{Dec}(\mu^*)$. Rare, dysplastic, or malignant lesions exhibit significantly higher reconstruction error because the decoder has learned general dermatological morphological patterns.

---

## 4. Hyperparameter Reference

| Component | Parameter | Configured Value | Description / Rationale |
|:---|:---|:---:|:---|
| **Input Shape** | Resolution | $64 \times 64 \times 3$ | RGB normalized in $[0, 1]$ |
| **Latent Space** | $z\_\text{dim}$ | $32$ | Compact continuous feature space for fast retrieval |
| **Encoder Channels** | Conv2d Layers | $32 \to 64 \to 128 \to 256$ | Progressive spatial reduction ($64 \to 32 \to 16 \to 8 \to 4$) |
| **Decoder Channels** | ConvTranspose2d | $256 \to 128 \to 64 \to 32 \to 3$ | Progressive upsampling ending with Sigmoid activation |
| **Beta Weight** | $\beta$ | $0.5$ | Balances reconstruction sharpness with latent smoothness |
| **Optimizer** | Algorithm & LR | Adam ($\text{lr}=0.001, \text{weight\_decay}=10^{-5}$) | Fast, stable gradient convergence |
| **Gallery Size** | Indexed Cases | $150$ | Balanced historical clinical cases with confirmed outcomes |

---

## 5. Artifacts & Deliverables Summary

| Artifact | Type | Description |
|:---|:---|:---|
| [`src/module5_vae.py`](file:///Users/pritimmondal/Projects/dermavision_ai/src/module5_vae.py) | Python Module | `SkinLesionVAE`, `vae_loss_function`, `SkinCaseSimilarityEngine`, and visualization functions |
| [`train_module5_vae.py`](file:///Users/pritimmondal/Projects/dermavision_ai/train_module5_vae.py) | Standalone Script | End-to-end training pipeline, gallery indexer, and artifact generator |
| [`tests/test_module5_vae.py`](file:///Users/pritimmondal/Projects/dermavision_ai/tests/test_module5_vae.py) | Test Suite | 8 automated unit tests covering shapes, ranges, sampling, retrieval, and serialization |
| `models/module5_vae_best.pth` | PyTorch Weights | Best weights for trained Convolutional VAE |
| `models/module5_vae_case_gallery.pt` | Serialized Gallery | 150 indexed clinical cases with latent embeddings & metadata |
| `Module5_VAE_Reconstruction_Grid.png` | Image Artifact | Side-by-side original vs. VAE reconstructed skin lesions |
| `Module5_VAE_Latent_Space_tSNE.png` | Image Artifact | 2D t-SNE scatter plot of latent embeddings across 6 clinical classes |
| `Module5_VAE_Latent_Interpolation.png` | Image Artifact | 10-step continuous morphological traversal between Nevus and Melanoma |
| `DermaVision_Module5_VAE_Results.csv` | CSV Data | Quantitative benchmark summary table |
