# DermaVision AI — Module 4 (Mouli Mukherjee & Ankita)

**Intelligent Longitudinal Skin Pigmentation Tracking, Multi-Horizon Trend Forecasting (LSTM / GRU), and Clinical Feedback NLP (BiLSTM)**

Module 4 introduces **temporal intelligence and longitudinal analysis** to DermaVision AI, expanding upon the static diagnostic evaluations of Modules 1–3 by tracking how skin conditions and pigmentation evolve over time.

---

## 1. Executive Summary & Curriculum Alignment

| # | Task | Technique | Output |
|---|------|-----------|--------|
| **1** | **Longitudinal Progression Tracking** | Stacked 2-Layer LSTM with LayerNorm | Multi-horizon forecasts (1, 3, 6 months) for Skin Health Score, Pigmentation %, and Severity |
| **2** | **Recurrent Architectural Ablation** | Gated Recurrent Unit (GRU) | Comparative benchmark against LSTM (MAE, $R^2$, convergence speed) |
| **3** | **Trajectory Risk Classification** | Softmax Head on Recurrent Latent States | Categorization into `Improving`, `Stable`, or `Worsening` with clinical alerts |
| **4** | **Clinical Text & Skincare Journal NLP** | Bidirectional LSTM (BiLSTM) + Max Pooling | 3-Class treatment feedback classification (`Positive`, `Neutral`, `Adverse`) from doctor notes & patient diaries |

> **Medical Disclaimer:** DermaVision AI is an educational and research decision-support tool. It does not replace clinical diagnosis by a licensed dermatologist.

---

## 2. Theoretical Background

### A. The Challenge of Temporal Progression in Dermatology
Pigmentation disorders (melasma, post-inflammatory hyperpigmentation, lentigines) and neoplastic lesions (nevi, dysplastic lesions, melanoma) change dynamically in response to UV exposure, topical treatments (retinoids, azelaic acid, sunscreens), and systemic factors. A single snapshot examination cannot capture whether a lesion is responding to therapy or progressively deteriorating.

### B. Long Short-Term Memory (LSTM) Architecture
Standard Recurrent Neural Networks (RNNs) suffer from vanishing and exploding gradients when processing multi-visit history over months. LSTMs resolve this via regulated cell states and gating mechanisms:

$$\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &&\text{(Forget Gate: discards irrelevant past history)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &&\text{(Input Gate: selects new clinical updates)} \\
\tilde{C}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad &&\text{(Candidate Memory Cell)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &&\text{(Updated Cell State)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &&\text{(Output Gate)} \\
h_t &= o_t \odot \tanh(C_t) \quad &&\text{(Hidden State)}
\end{aligned}$$

### C. Gated Recurrent Unit (GRU)
The GRU merges cell state and hidden state, using an update gate $z_t$ and reset gate $r_t$, offering a faster, parameter-efficient recurrent alternative for ablation comparison.

### D. Bidirectional LSTM (BiLSTM)
Dermatological notes and patient journal entries contain medical context where forward and backward word dependencies are both critical (e.g., *"not experiencing erythema"* vs. *"erythema not resolving"*). The BiLSTM processes the input in both temporal directions:

$$h_t^{\text{BiLSTM}} = [ \overrightarrow{h}_t \,;\, \overleftarrow{h}_t ]$$

Followed by temporal global max-pooling to extract the most prominent clinical indicators.

---

## 3. End-to-End Pipeline & Integration with Modules 1, 2, and 3

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│     MODULE 1 & MODULE 2         │      │            MODULE 3             │
│ • Skin Health Score (0–100)     │      │ • Quantified Pigmentation %     │
│ • Clinical Severity Score       │      │   (U-Net segmentation 0–100%)   │
│ • Pigmentation Risk Score       │      │ • Lesion Diameters (d1, d2 mm)  │
└────────────────┬────────────────┘      └────────────────┬────────────────┘
                 │                                        │
                 └───────────────────┬────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │   Longitudinal Sequence Formulation Engine             │
        │   Sequence: (t_0=Baseline, t_1=Mo 1, t_2=Mo 2, t_3=Mo 3)│
        │   Features per visit (D=8): Health, Pig%, Sev, Risk,   │
        │   Diameter 1 & 2, Symptom count, Treatment Adherence   │
        └────────────────────────────┬───────────────────────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │                             │
                      ▼                             ▼
        ┌───────────────────────────┐ ┌───────────────────────────┐
        │   SkinProgressionLSTM     │ │  ClinicalJournalBiLSTM    │
        │   (Multi-Horizon Reg.)    │ │  (Doctor Notes & Diaries) │
        └─────────────┬─────────────┘ └─────────────┬─────────────┘
                      │                             │
                      ▼                             ▼
        ┌───────────────────────────┐ ┌───────────────────────────┐
        │ • 1-Month Projected Score │ │ • Treatment Response Cat. │
        │ • 3-Month Projected Score │ │ • Adverse Peeling/Allergy │
        │ • 6-Month Projected Score │ │ • Progress Sentiment %    │
        │ • Improving/Stable/Worse  │ │                           │
        └───────────────────────────┘ └───────────────────────────┘
```

---

## 4. Input Features & Sequence Specification

Each patient sequence consists of $T=4$ consecutive observation steps with 8 normalized continuous indicators:

| Index | Feature | Normalization | Description |
|:---:|:---|:---:|:---|
| 0 | `skin_health_score` | $[0.0, 1.0]$ | DermaVision Skin Health Score (higher = better) |
| 1 | `pigmentation_percentage` | $[0.0, 1.0]$ | Measured visual pigmentation ratio from U-Net |
| 2 | `skin_severity_score` | $[0.0, 1.0]$ | Clinical severity synthesized from diagnosis and symptoms |
| 3 | `pigmentation_risk_score` | $[0.0, 1.0]$ | Melanocytic pigmentation risk score |
| 4 | `diameter_1` | Normalized by $30\text{ mm}$ | Primary orthogonal lesion axis |
| 5 | `diameter_2` | Normalized by $30\text{ mm}$ | Secondary orthogonal lesion axis |
| 6 | `symptom_burden` | $[0.0, 1.0]$ | Proportion of active clinical symptoms |
| 7 | `treatment_adherence` | $[0.0, 1.0]$ | Patient compliance with topical/photoprotection regimen |

### Multi-Horizon Targets
Each model projects three critical clinical milestones:
1. **+1 Month Horizon:** Short-term response (early verification of tolerance and initial stabilization).
2. **+3 Months Horizon:** Medium-term milestone (typical dermatological evaluation interval).
3. **+6 Months Horizon:** Long-term outcome (clearance, chronic maintenance, or relapse).

---

## 5. Model Architecture & Hyperparameters

### SkinProgressionLSTM & SkinProgressionGRU
* **Input Dimension:** 8
* **Hidden Dimension:** 64
* **Recurrent Layers:** 2 stacked layers with LayerNorm and Dropout ($p=0.25$)
* **Shared Representation:** Dense 64 $\to$ ReLU $\to$ Dropout
* **Heads:** Three independent regression projection heads ($64 \to 32 \to 3$) for 1M, 3M, and 6M horizons, plus a 3-class linear classifier for trend trajectory (`Improving`, `Stable`, `Worsening`).
* **Loss Function:**
  $$\mathcal{L} = \frac{1}{3} \sum_{h \in \{1\text{m}, 3\text{m}, 6\text{m}\}} \text{MSE}(\hat{y}_h, y_h) + \text{CrossEntropy}(\hat{c}_{\text{trend}}, c_{\text{trend}})$$

### ClinicalJournalBiLSTM
* **Vocabulary Size:** Built from clinical dermatological terms and patient diary entries
* **Embedding Dimension:** 64
* **Hidden Dimension:** 64 (bidirectional $\implies 128$ concatenated latent units)
* **Pooling:** Temporal Global Max-Pooling over token sequence
* **Output:** Softmax probabilities over 3 response classes (`Positive`, `Neutral`, `Adverse`)

---

## 6. Artifacts & Deliverables Summary

All Module 4 deliverables are preserved in isolated, brand-new files:

| Artifact | Type | Description |
|:---|:---|:---|
| [`src/module4_models.py`](file:///c:/Users/MOULI%20MUKHERJEE/Desktop/dermavision_ai/src/module4_models.py) | Python Module | `SkinProgressionLSTM`, `SkinProgressionGRU`, and `ClinicalJournalBiLSTM` |
| [`src/module4_sequence_dataset.py`](file:///c:/Users/MOULI%20MUKHERJEE/Desktop/dermavision_ai/src/module4_sequence_dataset.py) | Python Module | Temporal sequence builder and clinical notes corpus |
| [`src/module4_forecast.py`](file:///c:/Users/MOULI%20MUKHERJEE/Desktop/dermavision_ai/src/module4_forecast.py) | Python Module | Multi-horizon inference and clinical recommendation engine |
| [`train_module4.py`](file:///c:/Users/MOULI%20MUKHERJEE/Desktop/dermavision_ai/train_module4.py) | Standalone Script | End-to-end training, cross-model evaluation, and checkpoint exporter |
| [`tests/test_module4.py`](file:///c:/Users/MOULI%20MUKHERJEE/Desktop/dermavision_ai/tests/test_module4.py) | Test Suite | 6 unit and integration tests covering shapes, unrolling, and predictions |
| `DermaVision_Module4_Experiment_Results.csv` | CSV Data | Comparative benchmark table across LSTM, GRU, and BiLSTM |
| `Module4_Progression_Loss_Curves.png` | Image | Training and validation multi-task loss curves |

---

## 7. Authors & Team Contributions

- **Mouli Mukherjee:** Longitudinal skin progression dataset formulation, `SkinProgressionLSTM` multi-horizon regression architecture, `SkinProgressionGRU` ablation model, and forecast inference engine (`src/module4_forecast.py`).
- **Ankita:** `ClinicalJournalBiLSTM` natural language processing architecture, clinical text tokenization & vocabulary engine, model training & cross-evaluation pipeline (`train_module4.py`), interactive Jupyter notebook (`Module4_Longitudinal_Skin_Progression.ipynb`), and integration testing (`tests/test_module4.py`).
