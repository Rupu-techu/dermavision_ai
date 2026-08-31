# DermaVision AI — Workflow Sequence Diagram (Module 1)

This document illustrates the step-by-step execution workflow from patient information entry to dashboard visualization.

## Workflow Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Patient / Clinician
    participant UI as DermaVision Web UI (index.html/app.js)
    participant API as Flask REST API (app.py)
    participant Pre as Preprocessing Engine (preprocessing.py)
    participant Model as PyTorch DeepANN Model (model.py)
    participant Score as Clinical Score Engine (score_engine.py)

    User->>UI: 1. Fills form fields or selects Quick Preset
    User->>UI: 2. Clicks "Execute DermaVision AI Prediction"
    UI->>API: 3. Sends POST request to /api/predict (JSON payload)
    
    activate API
    API->>Pre: 4. Pass raw input dict to preprocess_user_input()
    activate Pre
    Pre->>Pre: Standardize age, fitspatrick, diameter_1, diameter_2
    Pre->>Pre: Apply One-Hot Encoding across 18 categorical variables
    Pre->>Pre: Assemble 78-feature vector in exact model column order
    Pre-->>API: Return PyTorch FloatTensor of shape (1, 78)
    deactivate Pre

    API->>Model: 5. Pass (1, 78) Tensor to DeepSkinDiseaseANN
    activate Model
    Model->>Model: Execute 4-Layer Feedforward Pass (78 -> 256 -> 128 -> 64 -> 6)
    Model->>Model: Apply Softmax activation across 6 diagnostic classes
    Model-->>API: Return Softmax Probabilities Tensor
    deactivate Model

    API->>Score: 6. Pass Probabilities & Raw Input to calculate_clinical_scores()
    activate Score
    Score->>Score: Identify Primary Diagnosis (ACK, BCC, MEL, NEV, SCC, SEK)
    Score->>Score: Calculate Prediction Confidence %
    Score->>Score: Compute Skin Severity Score (Base + Symptoms + Biopsy)
    Score->>Score: Compute Skin Health Score (100 - Severity - Confidence Adj)
    Score->>Score: Compute Pigmentation Risk Score (Base + Growth + Color Change)
    Score->>Score: Determine Categories (Low, Moderate, High, Critical, Good, Poor)
    Score-->>API: Return Structured Clinical Results Dictionary
    deactivate Score

    API-->>UI: 7. HTTP 200 OK Response with Diagnostic & Risk Data
    deactivate API

    UI->>UI: 8. Render Primary Diagnostic Card & Confidence Badge
    UI->>UI: 9. Animate Skin Health & Pigmentation Risk Gauges
    UI->>UI: 10. Display 6-Class Probability Breakdown Chart & Guidance
```

## Step Summary Table

| Step | Action | Description |
| :--- | :--- | :--- |
| **Step 1** | Form Input | User inputs age, Fitzpatrick skin type, lesion diameter, body region, and clinical symptoms. |
| **Step 2** | One-Hot Encoding | Categorical values are converted into 74 binary indicator fields. |
| **Step 3** | Numerical Normalization | `age`, `fitspatrick`, `diameter_1`, and `diameter_2` are normalized via Z-score formula `(x - μ) / σ`. |
| **Step 4** | Tensor Assembly | Features are ordered into an exact 78-element PyTorch `FloatTensor`. |
| **Step 5** | Model Forward Pass | Tensor flows through `Linear` and `ReLU` layers of `DeepSkinDiseaseANN`. |
| **Step 6** | Score Calculation | Clinical risk and health scores are generated using diagnostic baselines and symptom adjustments. |
| **Step 7** | Dashboard Rendering | Web interface animates score bars, badges, and probability distributions. |
