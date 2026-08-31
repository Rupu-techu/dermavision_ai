# DermaVision AI — DeepANN Skin Disease Classification & Risk Dashboard

**DermaVision AI** (Module 1) is an intelligent clinical decision-support platform powered by a PyTorch **Deep Artificial Neural Network (DeepANN)**. It transforms 78 patient demographic, clinical, and lesion parameters into multi-class skin cancer diagnoses, prediction confidence percentages, skin health scores, and pigmentation risk metrics.

---

## 🚀 Quick Start Guide (How to Run)

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. Required packages:
- `torch`
- `flask`
- `numpy`

### 2. Install Dependencies
In your terminal, navigate to the project directory and install requirements:
```bash
pip install torch flask numpy
```

### 3. Run the DermaVision AI Web Application
To start the Flask backend server and web dashboard:
```bash
python app.py
```
* **Server Output**:
  ```text
  Loading DermaVision PyTorch DeepANN Model from: .../models/PAD_UFES_20_deep_best_model.pth
  SUCCESS: PyTorch DeepANN Model loaded into memory!
  * Running on http://127.0.0.1:5000
  ```

### 4. Access the Web Dashboard
Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### 5. Run Automated Pipeline Tests
To execute the end-to-end integration tests (testing preprocessing, DeepANN model inference, and score generation):
```bash
python tests/test_pipeline.py
```

---

## 🧩 System Elements & Core Components

DermaVision AI consists of six key integrated elements:

```text
[User Form / Dashboard] ──> [78-Feature Preprocessor] ──> [DeepANN PyTorch Model] ──> [Clinical Score Engine] ──> [Interactive Gauges & Analytics]
```

### 1. User Input Intake Form (`static/index.html` & `static/app.js`)
An interactive web interface that collects raw patient inputs:
- **Demographics**: Age, Gender, Fitzpatrick Skin Type (I–VI), Paternal Ancestry, Maternal Ancestry.
- **Lesion Metrics**: Primary Diameter ($\text{d}_1$), Secondary Diameter ($\text{d}_2$), Anatomical Body Region (Face, Nose, Back, Arm, Chest, etc.).
- **Clinical Symptoms & Indicators**: Itch, Growth/Expansion, Pain/Tenderness, Color/Border Changes, Bleeding/Crusting, Raised Elevation, Biopsy History.
- **Exposure & History**: Personal/Family Cancer History, Tobacco Smoking, Pesticide Exposure.

### 2. 78-Feature Preprocessing Engine (`src/preprocessing.py`)
Converts raw user input into a standardized PyTorch `FloatTensor` of shape `(1, 78)`:
- **Numerical Standardization**: Applies Z-Score normalization ($\frac{x - \mu}{\sigma}$) using fitted `StandardScaler` parameters for `age`, `fitspatrick`, `diameter_1`, and `diameter_2`.
- **Categorical One-Hot Encoding**: Expands binary and categorical fields across 74 dummy feature columns in the exact feature order expected by the model.

### 3. PyTorch DeepANN Neural Network (`src/model.py` & `models/PAD_UFES_20_deep_best_model.pth`)
The core diagnostic engine ("the brain"):
- **Architecture**: 4-Layer Feedforward Neural Network (`78 -> 256 -> 128 -> 64 -> 6`).
- **Layers & Regularization**: Linear transformations with `ReLU` activations and `Dropout` (0.30, 0.25, 0.20) for generalization.
- **Output Diagnostic Classes (6-Class Softmax)**:
  - **ACK**: Actinic Keratosis (Pre-cancerous)
  - **BCC**: Basal Cell Carcinoma (Malignant)
  - **MEL**: Melanoma (High-Risk Malignant)
  - **NEV**: Nevus (Benign Mole)
  - **SCC**: Squamous Cell Carcinoma (Malignant)
  - **SEK**: Seborrheic Keratosis (Benign Growth)

### 4. Clinical Score & Risk Engine (`src/score_engine.py`)
Transforms raw class probability distributions into rich, clinically actionable metrics:
- **Primary Diagnosis & Confidence %**: High-confidence class prediction and full class probability breakdown.
- **Skin Severity Score (0–100)**: Evaluates baseline diagnostic severity weighted with positive symptom counts and biopsy history.
- **Skin Health Score (0–100)**: Measures remaining skin health integrity adjusted for prediction uncertainty ($100 - \text{Severity} - \text{Confidence Adj}$).
- **Pigmentation Risk Score (0–100)**: Calculates risk level based on lesion pigmentation baseline, recent expansion, and color changes.
- **Risk Categorization**: Categorizes scores into **Low**, **Moderate**, **High**, or **Critical** risk tiers with tailored dermatological guidance.

### 5. Flask REST API Backend (`app.py`)
Serves the web dashboard static assets and provides RESTful endpoints:
- `GET /`: Serves the DermaVision web dashboard interface.
- `GET /api/health`: Returns API online status, loaded model name (`DeepANN`), and feature dimensions.
- `POST /api/predict`: Accepts user form JSON payload, runs preprocessing, model inference, and score generation, returning structured diagnostic analytics.
- `GET /api/presets`: Provides instant clinical preset test cases (e.g., Melanoma Suspect, Basal Cell Carcinoma, Benign Mole, Actinic Keratosis).

### 6. Technical & Architectural Documentation (`docs/`)
Comprehensive documentation detailing the system design:
- [`docs/architecture_diagram.md`](file:///c:/Users/Purnendu/dermavision_ai/docs/architecture_diagram.md): Architecture diagram illustrating component data flows.
- [`docs/workflow_diagram.md`](file:///c:/Users/Purnendu/dermavision_ai/docs/workflow_diagram.md): Step-by-step sequence diagram from input submission to dashboard rendering.
- [`docs/module1_documentation.md`](file:///c:/Users/Purnendu/dermavision_ai/docs/module1_documentation.md): Module 1 technical specifications, feature definitions, mathematical formulas, and model evaluation summary.

---

## 📁 Repository Directory Structure

```text
dermavision_ai/
├── app.py                      # Flask REST API Server & Route Handlers
├── README.md                   # Project Overview & Execution Instructions
├── PAD_UFES_20_ANN.ipynb       # Jupyter Notebook for Model Training & Evaluation
├── models/
│   ├── PAD_UFES_20_deep_best_model.pth  # Active DeepANN PyTorch Trained Model
│   └── ...                     # Saved Checkpoint Models
├── src/
│   ├── __init__.py
│   ├── model.py                # DeepANN PyTorch Network Class & Model Loader
│   ├── preprocessing.py        # 78-Feature Normalizer & One-Hot Encoder
│   └── score_engine.py         # Skin Health & Pigmentation Risk Score Calculator
├── static/
│   ├── index.html              # Interactive DermaVision Dashboard HTML
│   ├── style.css               # Dashboard Styling & Visual System
│   └── app.js                  # Frontend UI Logic & API Communication
├── docs/
│   ├── architecture_diagram.md # System Architecture Diagram
│   ├── workflow_diagram.md     # Workflow Sequence Diagram
│   └── module1_documentation.md# Module 1 Comprehensive Technical Documentation
└── tests/
    └── test_pipeline.py        # Automated Integration Test Suite
```

---

## 🌐 Live Application Dashboard

When the app is running (`python app.py`), visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** to interact with the input form, click quick clinical presets, and view animated risk gauges and probability breakdowns.
