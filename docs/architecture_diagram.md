# DermaVision AI — System Architecture Diagram (Module 1)

This document presents the high-level system architecture and data transformation pipeline for **DermaVision AI Module 1** (ANN/MLP-Based Skin Disease Classification & Risk Score Engine).

## High-Level Architecture Diagram

```mermaid
graph TD
    subgraph UI ["User Interface Layer (Frontend)"]
        A["Patient Intake Form<br>(Demographics, Lesion Metrics, Symptoms)"]
        H["DermaVision Dashboard<br>(Diagnosis, Health Score, Risk Gauges)"]
    end

    subgraph API ["Application Server (Flask Backend)"]
        B["Flask REST API Server<br>(app.py - /api/predict)"]
    end

    subgraph PRE ["Preprocessing & Feature Engine"]
        C["Raw Parameter Extractor"]
        D["Categorical One-Hot Encoder<br>(74 Dummy Dimensions)"]
        E["StandardScaler Z-Score Normalizer<br>(4 Numerical Features)"]
        F["78-Feature Vector Assembler<br>(Tensor Shape: 1 x 78)"]
    end

    subgraph ML ["PyTorch Neural Network Engine"]
        G["DeepANN Model<br>(PAD_UFES_20_deep_best_model.pth)"]
        I["Softmax Activation<br>(6-Class Probabilities)"]
    end

    subgraph SE ["DermaVision Clinical Score Engine"]
        J["Diagnosis & Confidence Calculator"]
        K["Skin Severity Score Engine<br>(Base + Symptoms + Biopsy)"]
        L["Skin Health Score Calculator<br>(100 - Severity - Confidence Adj)"]
        M["Pigmentation Risk Calculator<br>(Base + Growth + Color Change)"]
        N["Risk Categorization & Clinical Guidance"]
    end

    A -->|"JSON Payload"| B
    B --> C
    C --> D
    C --> E
    D --> F
    E --> F
    F -->|"PyTorch FloatTensor (1, 78)"| G
    G --> I
    I --> J
    J --> K
    K --> L
    K --> M
    L --> N
    M --> N
    J --> N
    N -->|"Structured JSON Analytics"| B
    B -->|"HTTP 200 OK Response"| H
```

## System Component Breakdown

| Component | Responsibility | Inputs / Outputs |
| :--- | :--- | :--- |
| **User Intake Form** | Collects patient demographics, lesion metrics, and clinical symptoms. | Raw user inputs → JSON Payload |
| **Flask API Server** | Handles HTTP requests, orchestrates pipeline, and returns JSON predictions. | POST `/api/predict` → Response JSON |
| **Preprocessing Engine** | Performs standardization and one-hot encoding into a 78-feature vector. | 18 Raw Parameters → `torch.Tensor` (1, 78) |
| **Deep ANN PyTorch Model** | Computes forward pass through trained neural network layers (`78 -> 256 -> 128 -> 64 -> 6`). | `torch.Tensor` (1, 78) → Logits & Probabilities |
| **Score Engine** | Derives **Skin Health Score**, **Pigmentation Risk Score**, and clinical categories. | Class Probabilities → Clinical Risk Metrics |
| **DermaVision Dashboard** | Renders interactive gauges, confidence metrics, probability bar charts, and clinical guidance. | Clinical Analytics → Rich Web Visuals |
