"""
DermaVision AI — Module 4: Future Skin Progression Forecasting & NLP Inference Engine
====================================================================================
Provides inference pipelines to project skin condition trajectories (1, 3, 6 months)
and analyze clinical notes/skincare journals.

Author: Mouli Mukherjee (Module 4)
"""

import os
import torch
import numpy as np

from src.module4_models import SkinProgressionLSTM, SkinProgressionGRU, ClinicalJournalBiLSTM
from src.module4_sequence_dataset import tokenize_text

TREND_LABELS = {
    0: "Improving (Favorable Response)",
    1: "Stable (Maintenance)",
    2: "Worsening (Progression Alert)"
}

JOURNAL_FEEDBACK_LABELS = {
    0: "Positive (Favorable Skincare / Treatment Response)",
    1: "Neutral (Condition Stable / Ongoing Observation)",
    2: "Adverse (Clinical Irritation / Suspected Progression)"
}

def load_progression_model(model_path, model_type='lstm', input_dim=8, hidden_dim=64, device='cpu'):
    """
    Loads trained weights into SkinProgressionLSTM or SkinProgressionGRU.
    """
    if model_type.lower() == 'lstm':
        model = SkinProgressionLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2)
    else:
        model = SkinProgressionGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2)
        
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_clinical_notes_model(model_path, vocab_size=5000, embed_dim=64, hidden_dim=64, device='cpu'):
    """
    Loads trained ClinicalJournalBiLSTM model.
    """
    model = ClinicalJournalBiLSTM(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=3)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_skin_progression(model, past_visits: list, device='cpu') -> dict:
    """
    Computes future projections for a patient based on historical visit sequence.
    past_visits: list of visit dicts or 2D array of shape (seq_len, 8):
      [health_score, pig_pct, severity_score, pig_risk, d1, d2, symptom_count, adherence]
    
    Returns structured forecast dictionary:
    - 1 Month projection
    - 3 Months projection
    - 6 Months projection
    - Trajectory category & confidence
    - Timeline progression data points for charting
    """
    if isinstance(past_visits, list):
        seq_array = np.array(past_visits, dtype=np.float32)
    else:
        seq_array = np.asarray(past_visits, dtype=np.float32)
        
    if seq_array.ndim == 2:
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32).to(device)
        
    model.eval()
    with torch.no_grad():
        outputs = model(seq_tensor)
        
    pred_1m = outputs['forecast_1m'].squeeze().cpu().numpy()
    pred_3m = outputs['forecast_3m'].squeeze().cpu().numpy()
    pred_6m = outputs['forecast_6m'].squeeze().cpu().numpy()
    trend_probs = torch.softmax(outputs['trend_logits'], dim=1).squeeze().cpu().numpy()
    
    trend_idx = int(trend_probs.argmax())
    trend_name = TREND_LABELS.get(trend_idx, "Unknown")
    trend_conf = round(float(trend_probs[trend_idx]) * 100, 2)
    
    # Clip continuous forecasts to valid medical range [0, 100]
    def format_forecast(arr):
        return {
            'skin_health_score': round(float(np.clip(arr[0], 0.0, 100.0)), 2),
            'pigmentation_percentage': round(float(np.clip(arr[1], 0.0, 100.0)), 2),
            'clinical_severity_score': round(float(np.clip(arr[2], 0.0, 100.0)), 2)
        }
        
    f_1m = format_forecast(pred_1m)
    f_3m = format_forecast(pred_3m)
    f_6m = format_forecast(pred_6m)
    
    # Clinical recommendations based on trajectory
    if trend_idx == 0:
        rec = "Current skincare and clinical management plan is showing positive therapeutic efficacy. Maintain current photoprotection and follow-up in 3 months."
    elif trend_idx == 1:
        rec = "Skin lesion and pigmentation parameters remain stable. Continue daily sunscreen adherence and regular quarterly self-monitoring."
    else:
        rec = "Warning: Projected trajectory indicates progressive pigmentation expansion or worsening clinical risk. Clinical review with a dermatologist is strongly recommended."
        
    return {
        'trajectory_category': trend_name,
        'trajectory_code': trend_idx,
        'trajectory_confidence': trend_conf,
        'forecast_1_month': f_1m,
        'forecast_3_months': f_3m,
        'forecast_6_months': f_6m,
        'clinical_recommendation': rec,
        'timeline_chart_points': [
            {'horizon': 'Current Baseline', 'health_score': round(float(seq_array[-1, 0] * 100.0), 2), 'pigmentation_pct': round(float(seq_array[-1, 1] * 100.0), 2)},
            {'horizon': '+1 Month', 'health_score': f_1m['skin_health_score'], 'pigmentation_pct': f_1m['pigmentation_percentage']},
            {'horizon': '+3 Months', 'health_score': f_3m['skin_health_score'], 'pigmentation_pct': f_3m['pigmentation_percentage']},
            {'horizon': '+6 Months', 'health_score': f_6m['skin_health_score'], 'pigmentation_pct': f_6m['pigmentation_percentage']}
        ]
    }

def analyze_clinical_notes(model, vocab: dict, note_text: str, max_len=30, device='cpu') -> dict:
    """
    Parses a clinical note or patient journal entry through BiLSTM.
    """
    tokens = tokenize_text(note_text)
    indexed = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens][:max_len]
    if len(indexed) < max_len:
        indexed += [0] * (max_len - len(indexed))
        
    input_tensor = torch.tensor([indexed], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    pred_idx = int(probs.argmax())
    feedback_name = JOURNAL_FEEDBACK_LABELS.get(pred_idx, "Unknown")
    confidence = round(float(probs[pred_idx]) * 100, 2)
    
    return {
        'input_text': note_text,
        'feedback_category': feedback_name,
        'feedback_code': pred_idx,
        'confidence': confidence,
        'probabilities': {
            'Positive': round(float(probs[0]) * 100, 2),
            'Neutral': round(float(probs[1]) * 100, 2),
            'Adverse': round(float(probs[2]) * 100, 2)
        }
    }
