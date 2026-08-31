import torch

DIAGNOSIS_LABELS = {
    0: "ACK",
    1: "BCC",
    2: "MEL",
    3: "NEV",
    4: "SCC",
    5: "SEK"
}

DIAGNOSIS_FULL_NAMES = {
    "ACK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "MEL": "Melanoma",
    "NEV": "Nevus (Benign Mole)",
    "SCC": "Squamous Cell Carcinoma",
    "SEK": "Seborrheic Keratosis"
}

DIAGNOSIS_DESCRIPTIONS = {
    "ACK": "Pre-cancerous scaly growth caused by sun damage.",
    "BCC": "Common, slow-growing malignant skin cancer.",
    "MEL": "Aggressive, high-risk malignant skin cancer requiring urgent evaluation.",
    "NEV": "Common benign mole formed by melanocytes.",
    "SCC": "Malignant skin cancer arising from squamous cells.",
    "SEK": "Common, harmless non-cancerous skin growth."
}

SEVERITY_BASE = {
    "NEV": 10,
    "SEK": 15,
    "ACK": 35,
    "BCC": 50,
    "SCC": 60,
    "MEL": 85
}

PIGMENTATION_BASE = {
    "NEV": 35,
    "SEK": 40,
    "ACK": 55,
    "BCC": 60,
    "SCC": 65,
    "MEL": 90
}

def calculate_clinical_scores(probabilities_tensor: torch.Tensor, raw_input: dict) -> dict:
    """
    Computes diagnostic predictions, confidence percentages, clinical scores, and categories.
    
    Returns a structured dictionary formatted for the DermaVision Dashboard.
    """
    probs = probabilities_tensor.squeeze().detach().cpu().numpy()
    
    # 1. Primary Diagnosis & Confidence
    predicted_idx = int(probs.argmax())
    predicted_code = DIAGNOSIS_LABELS.get(predicted_idx, "UNKNOWN")
    confidence_pct = round(float(probs[predicted_idx]) * 100, 2)
    
    # Class Probabilities Dictionary (%)
    class_probabilities = {
        DIAGNOSIS_LABELS[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(probs))
    }
    
    # 2. Base Severity & Skin Severity Score
    base_severity = SEVERITY_BASE.get(predicted_code, 30)
    severity_score = float(base_severity)
    
    symptoms = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation']
    for symptom in symptoms:
        val = str(raw_input.get(symptom, False)).upper()
        if val in ['TRUE', '1', 'YES']:
            severity_score += 5.0
            
    if str(raw_input.get('biopsed', False)).upper() in ['TRUE', '1', 'YES']:
        severity_score += 10.0
        
    severity_score = max(0.0, min(100.0, severity_score))
    
    # 3. Skin Health Score
    health_score = 100.0 - severity_score
    confidence_adj = (100.0 - confidence_pct) * 0.10
    health_score = max(0.0, min(100.0, health_score - confidence_adj))
    
    # 4. Pigmentation Risk Score
    base_pigmentation = PIGMENTATION_BASE.get(predicted_code, 40)
    pigmentation_risk = float(base_pigmentation)
    
    if str(raw_input.get('changed', False)).upper() in ['TRUE', '1', 'YES']:
        pigmentation_risk += 10.0
        
    if str(raw_input.get('grew', False)).upper() in ['TRUE', '1', 'YES']:
        pigmentation_risk += 5.0
        
    pigmentation_risk = max(0.0, min(100.0, pigmentation_risk))
    
    # 5. Risk Categorizations
    def get_severity_category(score):
        if score <= 20:
            return "Low"
        elif score <= 50:
            return "Moderate"
        elif score <= 75:
            return "High"
        else:
            return "Critical"
            
    def get_health_category(score):
        if score >= 80:
            return "Good"
        elif score >= 50:
            return "Moderate"
        else:
            return "Poor"
            
    def get_pigmentation_category(score):
        if score < 40:
            return "Low"
        elif score < 70:
            return "Moderate"
        else:
            return "High"
            
    severity_cat = get_severity_category(severity_score)
    health_cat = get_health_category(health_score)
    pigmentation_cat = get_pigmentation_category(pigmentation_risk)
    
    # Clinical Recommendations
    recommendation = ""
    if predicted_code in ["MEL", "SCC", "BCC"] or severity_cat in ["High", "Critical"]:
        recommendation = "Urgent clinical consultation with a board-certified dermatologist is recommended. Biopsy or dermoscopic evaluation advised."
    elif predicted_code in ["ACK"] or severity_cat == "Moderate":
        recommendation = "Schedule a routine dermatological assessment for monitoring and preventative skin health management."
    else:
        recommendation = "Low clinical risk detected. Continue regular skin self-examinations and routine annual skin health checkups."
        
    return {
        "diagnosis_code": predicted_code,
        "diagnosis_name": DIAGNOSIS_FULL_NAMES.get(predicted_code, predicted_code),
        "diagnosis_description": DIAGNOSIS_DESCRIPTIONS.get(predicted_code, ""),
        "prediction_confidence": confidence_pct,
        "class_probabilities": class_probabilities,
        "skin_health_score": round(health_score, 2),
        "skin_health_category": health_cat,
        "pigmentation_risk_score": round(pigmentation_risk, 2),
        "pigmentation_risk_category": pigmentation_cat,
        "skin_severity_score": round(severity_score, 2),
        "severity_category": severity_cat,
        "recommendation": recommendation
    }
