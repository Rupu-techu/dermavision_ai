# DermaVision AI — Module 1 Technical & Integration Documentation

## Executive Summary

DermaVision AI Module 1 provides **ANN/MLP-based skin disease classification and clinical risk analytics**. 

While the machine learning model ("the brain") was trained on 2,298 lesion samples from the PAD-UFES-20 dataset across 78 features, this documentation details the **user interface, feature preprocessing pipeline, model loading, clinical score engine, and interactive web dashboard integration**.

---

## 1. User Input Form Design

The user intake form collects required patient demographics, lesion metrics, and clinical symptoms necessary for neural inference:

### Input Parameters & Options

| Section | Parameter Name | Data Type | Options / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Demographics** | `age` | Numeric | 1 to 110 Years | Patient age in years |
| | `gender` | Categorical | `MALE`, `FEMALE` | Biological gender |
| | `fitspatrick` | Ordinal | Scale 1 to 6 (Types I–VI) | Fitzpatrick skin phototype scale |
| | `background_father` | Categorical | 13 European/South American Ethnicities | Paternal ancestry background |
| | `background_mother` | Categorical | 11 European/South American Ethnicities | Maternal ancestry background |
| **Lesion Metrics** | `diameter_1` | Numeric | 0.1 to 50.0 mm | Primary lesion diameter |
| | `diameter_2` | Numeric | 0.1 to 50.0 mm | Secondary orthogonal diameter |
| | `region` | Categorical | 14 Anatomical Body Regions | Body location (Face, Nose, Back, etc.) |
| **Clinical Symptoms** | `itch` | Boolean | `True` / `False` | Itching sensation present |
| | `grew` | Boolean | `True` / `False` | Recent lesion expansion or growth |
| | `hurt` | Boolean | `True` / `False` | Pain or tenderness |
| | `changed` | Boolean | `True` / `False` | Alteration in color, border, or shape |
| | `bleed` | Boolean | `True` / `False` | Bleeding, crusting, or oozing |
| | `elevation` | Boolean | `True` / `False` | Raised surface above skin level |
| | `biopsed` | Boolean | `True` / `False` | History of tissue biopsy |
| **History & Exposure**| `skin_cancer_history`| Boolean | `True` / `False` | Personal prior skin cancer |
| | `cancer_history` | Boolean | `True` / `False` | Family history of malignancy |
| | `smoke` | Boolean | `True` / `False` | Tobacco smoking history |

---

## 2. 78 Input Feature Mapping & Preprocessing

The trained PyTorch model (`PAD_UFES_20_deep_best_model.pth` - **DeepANN**) expects a vector of **exactly 78 features**:

### Feature Engineering Architecture

1. **Numerical Standardization (4 Features)**:
   Applied via StandardScaler Z-score formula:
   $$\text{Z} = \frac{x - \mu}{\sigma}$$
   - `age`: $\mu = 60.46, \sigma = 15.89$
   - `fitspatrick`: $\mu = 2.17, \sigma = 0.60$
   - `diameter_1`: $\mu = 11.23, \sigma = 7.02$
   - `diameter_2`: $\mu = 8.55, \sigma = 4.69$

2. **Categorical One-Hot Encoding (74 Dummy Features)**:
   - Binary/Tri-state flags: `smoke_False/True`, `drink_False/True`, `pesticide_False/True`, `itch_False/True/UNK`, `grew_False/True/UNK`, `hurt_False/True/UNK`, `changed_False/True/UNK`, `bleed_False/True/UNK`, `elevation_False/True/UNK`, `biopsed_False/True`.
   - Father Ancestry: 13 columns (`background_father_AUSTRIA` to `background_father_UNK`).
   - Mother Ancestry: 11 columns (`background_mother_BRAZIL` to `background_mother_UNK`).
   - Body Region: 14 columns (`region_ABDOMEN` to `region_THIGH`).
   - Gender: 2 columns (`gender_FEMALE`, `gender_MALE`).
   - Medical History & Infrastructure: `skin_cancer_history`, `cancer_history`, `has_piped_water`, `has_sewage_system`.

---

## 3. PyTorch Model Connection & Forward Inference

The `DeepANN` model architecture is defined in `src/model.py`:

```
Linear(78, 256) -> ReLU -> Dropout(0.3)
Linear(256, 128) -> ReLU -> Dropout(0.25)
Linear(128, 64)  -> ReLU -> Dropout(0.2)
Linear(64, 6)    -> Softmax
```

### Output Classes

- **0: ACK** — Actinic Keratosis (Pre-cancerous)
- **1: BCC** — Basal Cell Carcinoma (Malignant)
- **2: MEL** — Melanoma (High-Risk Malignant)
- **3: NEV** — Nevus (Benign Mole)
- **4: SCC** — Squamous Cell Carcinoma (Malignant)
- **5: SEK** — Seborrheic Keratosis (Benign Growth)

---

## 4. DermaVision Score Engine Formulas

Instead of displaying raw class names, DermaVision computes comprehensive clinical health metrics:

### 1. Skin Severity Score (0 – 100)

$$\text{Severity Score} = \text{Base Severity} + 5 \times N_{\text{symptoms}} + 10 \times \text{Biopsied}$$

- **Base Severity**: `NEV`: 10, `SEK`: 15, `ACK`: 35, `BCC`: 50, `SCC`: 60, `MEL`: 85.
- **Symptoms ($N_{\text{symptoms}}$)**: Number of positive flags among `itch`, `grew`, `hurt`, `changed`, `bleed`, `elevation`.

### 2. Skin Health Score (0 – 100)

$$\text{Health Score} = 100 - \text{Severity Score} - (100 - \text{Confidence \%}) \times 0.10$$

Higher scores represent superior skin integrity and lower risk.

### 3. Pigmentation Risk Score (0 – 100)

$$\text{Pigmentation Risk} = \text{Base Pigmentation} + 10 \times \text{Changed} + 5 \times \text{Grew}$$

- **Base Risk**: `NEV`: 35, `SEK`: 40, `ACK`: 55, `BCC`: 60, `SCC`: 65, `MEL`: 90.

### 4. Clinical Categories

- **Severity Category**: Low ($\le 20$), Moderate ($\le 50$), High ($\le 75$), Critical ($> 75$)
- **Skin Health Category**: Good ($\ge 80$), Moderate ($\ge 50$), Poor ($< 50$)
- **Pigmentation Risk Category**: Low ($< 40$), Moderate ($< 70$), High ($\ge 70$)

---

## 5. Dashboard Integration & API Endpoint

The system includes a Flask application (`app.py`) serving both the REST API and the interactive web interface (`static/index.html`).

### API Specifications

- **Endpoint**: `POST /api/predict`
- **Request Format**: JSON object with raw patient fields.
- **Response Format**:
  ```json
  {
    "success": true,
    "data": {
      "diagnosis_code": "MEL",
      "diagnosis_name": "Melanoma",
      "prediction_confidence": 98.45,
      "class_probabilities": {
        "ACK": 0.12,
        "BCC": 1.10,
        "MEL": 98.45,
        "NEV": 0.05,
        "SCC": 0.20,
        "SEK": 0.08
      },
      "skin_health_score": 12.50,
      "skin_health_category": "Poor",
      "pigmentation_risk_score": 100.00,
      "pigmentation_risk_category": "High",
      "skin_severity_score": 87.50,
      "severity_category": "Critical",
      "recommendation": "Urgent clinical consultation with a board-certified dermatologist is recommended."
    }
  }
  ```

---

## 6. Model Performance & Evaluation Summary

Evaluated on 2,298 patient samples from the PAD-UFES-20 clinical dataset:

- **Overall Accuracy**: **93.95%**
- **ACK Precision/Recall**: 92.74% / 96.30%
- **BCC Precision/Recall**: 97.64% / 97.87%
- **MEL Precision/Recall**: 94.12% / 92.31%
- **NEV Precision/Recall**: 89.53% / 94.67%
- **SCC Precision/Recall**: 94.65% / 92.19%
- **SEK Precision/Recall**: 87.82% / 73.62%
