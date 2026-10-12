"""
DermaVision AI — Module 4: Sequence Dataset & Preprocessing Engine
===================================================================
Constructs longitudinal time-series progression datasets from PAD-UFES-20
and U-Net pigmentation reports, as well as clinical text logs for BiLSTM NLP.

Author: Mouli Mukherjee (Module 4)
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class LongitudinalSkinDataset(Dataset):
    """
    PyTorch Dataset for multi-visit longitudinal skin progression tracking.
    Each sample consists of:
    - sequence: (seq_len, num_features) past visit features
    - target_1m: (3,) [health_score, pigmentation_pct, severity_score] at +1 Month
    - target_3m: (3,) [health_score, pigmentation_pct, severity_score] at +3 Months
    - target_6m: (3,) [health_score, pigmentation_pct, severity_score] at +6 Months
    - trend_label: int (0: Improving, 1: Stable, 2: Worsening)
    """
    def __init__(self, sequences, targets_1m, targets_3m, targets_6m, trend_labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets_1m = torch.tensor(targets_1m, dtype=torch.float32)
        self.targets_3m = torch.tensor(targets_3m, dtype=torch.float32)
        self.targets_6m = torch.tensor(targets_6m, dtype=torch.float32)
        self.trend_labels = torch.tensor(trend_labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'target_1m': self.targets_1m[idx],
            'target_3m': self.targets_3m[idx],
            'target_6m': self.targets_6m[idx],
            'trend_label': self.trend_labels[idx]
        }


def build_longitudinal_dataset(base_dir=None, seq_len=4, seed=42):
    """
    Loads PAD-UFES-20 metadata and Module 3 pigmentation reports to construct
    longitudinal sequences of patient visits.
    """
    np.random.seed(seed)
    
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    metadata_path = os.path.join(base_dir, 'metadata.csv')
    pigmentation_path = os.path.join(base_dir, 'pigmentation_report.csv')
    
    # Read PAD-UFES-20 metadata
    if os.path.exists(metadata_path):
        meta_df = pd.read_csv(metadata_path)
    else:
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
        
    # Read pigmentation report if available
    pigment_dict = {}
    if os.path.exists(pigmentation_path):
        try:
            p_df = pd.read_csv(pigmentation_path)
            for _, row in p_df.iterrows():
                fname = str(row['filename'])
                pct = float(row['percentage'])
                pigment_dict[fname] = pct
        except Exception:
            pass
            
    # Diagnosis baseline severity mapping
    diag_base_severity = {
        'NEV': 10.0, 'SEK': 15.0, 'ACK': 35.0, 'BCC': 50.0, 'SCC': 60.0, 'MEL': 85.0
    }
    diag_base_pigment = {
        'NEV': 23.5, 'SEK': 14.3, 'ACK': 25.8, 'BCC': 18.2, 'SCC': 20.5, 'MEL': 30.9
    }
    
    sequences = []
    targets_1m = []
    targets_3m = []
    targets_6m = []
    trend_labels = []
    
    # Iterate through patients in the dataset
    for idx, row in meta_df.iterrows():
        diag = str(row.get('diagnostic', 'NEV')).upper()
        base_sev = diag_base_severity.get(diag, 25.0)
        
        # Determine base pigmentation % (grounded in Module 3 U-Net report where possible)
        img_id = str(row.get('img_id', ''))
        if img_id in pigment_dict:
            base_pig = pigment_dict[img_id]
        else:
            base_pig = diag_base_pigment.get(diag, 22.0)
            
        d1 = float(row.get('diameter_1', 10.0)) if pd.notna(row.get('diameter_1')) else 10.0
        d2 = float(row.get('diameter_2', 8.0)) if pd.notna(row.get('diameter_2')) else 8.0
        
        # Clinical symptoms count
        symptoms = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation']
        symptom_count = sum(1 for s in symptoms if str(row.get(s, False)).strip().upper() in ['TRUE', '1'])
        
        # Calculate DermaVision clinical score engine values
        biopsed = 10.0 if str(row.get('biopsed', False)).strip().upper() in ['TRUE', '1'] else 0.0
        calculated_severity = min(100.0, base_sev + (5.0 * symptom_count) + biopsed)
        calculated_health = max(0.0, 100.0 - calculated_severity)
        calculated_pig_risk = min(100.0, base_sev * 0.9 + (10.0 if str(row.get('changed', False)).strip().upper() in ['TRUE', '1'] else 0.0))
        
        # Determine natural clinical progression course based on pathology
        # Benign (NEV, SEK) with treatment usually improve or stay stable
        # Malignant / pre-cancerous untreated (MEL, BCC, SCC, ACK) tend to worsen without intervention
        if diag in ['NEV', 'SEK']:
            traj_type = np.random.choice([0, 1], p=[0.7, 0.3]) # Improving or Stable
        elif diag in ['MEL', 'SCC']:
            traj_type = np.random.choice([1, 2], p=[0.2, 0.8]) # Stable or Worsening
        else: # ACK, BCC
            traj_type = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
            
        # Synthesize realistic longitudinal visit series (T=4 past observations)
        seq_steps = []
        cur_health = calculated_health
        cur_pig = base_pig
        cur_sev = calculated_severity
        cur_d1 = d1
        cur_d2 = d2
        cur_symp = symptom_count
        cur_adherence = float(np.random.uniform(0.75, 1.0) if traj_type == 0 else np.random.uniform(0.2, 0.6))
        cur_uv = float(np.random.uniform(1.0, 3.0) if traj_type == 0 else np.random.uniform(4.0, 8.0))
        
        for t in range(seq_len):
            step_features = [
                cur_health / 100.0,          # Normalized Health Score [0, 1]
                cur_pig / 100.0,             # Normalized Pigmentation % [0, 1]
                cur_sev / 100.0,             # Normalized Clinical Severity [0, 1]
                calculated_pig_risk / 100.0, # Pigmentation Risk Score [0, 1]
                cur_d1 / 30.0,               # Normalized Primary Diameter
                cur_d2 / 30.0,               # Normalized Secondary Diameter
                cur_symp / 6.0,              # Symptom Burden [0, 1]
                cur_adherence                # Adherence / Treatment Factor [0, 1]
            ]
            seq_steps.append(step_features)
            
            # Step perturbation towards trajectory
            noise = np.random.normal(0, 0.02)
            if traj_type == 0:  # Improving
                cur_health = min(100.0, cur_health + np.random.uniform(1.0, 2.5) + noise)
                cur_pig = max(2.0, cur_pig - np.random.uniform(0.8, 2.0) - noise)
                cur_sev = max(5.0, cur_sev - np.random.uniform(1.0, 2.5) - noise)
            elif traj_type == 1: # Stable
                cur_health = min(100.0, max(0.0, cur_health + noise))
                cur_pig = max(2.0, cur_pig + noise)
                cur_sev = max(5.0, cur_sev + noise)
            else: # Worsening
                cur_health = max(0.0, cur_health - np.random.uniform(1.5, 3.5) - noise)
                cur_pig = min(95.0, cur_pig + np.random.uniform(1.0, 3.0) + noise)
                cur_sev = min(95.0, cur_sev + np.random.uniform(1.5, 3.5) + noise)
                
        # Multi-Horizon Targets: 1 Month, 3 Months, 6 Months ahead
        # Target representation: [Health Score, Pigmentation %, Severity Score] (0-100 scale)
        def project_horizon(months, t_type, h, p, s):
            mult = months / 3.0
            if t_type == 0: # Improving
                t_h = min(100.0, h + mult * 4.0 + np.random.normal(0, 1.0))
                t_p = max(2.0, p - mult * 3.5 + np.random.normal(0, 0.8))
                t_s = max(5.0, s - mult * 4.0 + np.random.normal(0, 1.0))
            elif t_type == 1: # Stable
                t_h = min(100.0, max(0.0, h + np.random.normal(0, 1.5)))
                t_p = max(2.0, p + np.random.normal(0, 1.0))
                t_s = max(5.0, s + np.random.normal(0, 1.5))
            else: # Worsening
                t_h = max(0.0, h - mult * 5.0 + np.random.normal(0, 1.5))
                t_p = min(95.0, p + mult * 4.5 + np.random.normal(0, 1.2))
                t_s = min(95.0, s + mult * 5.0 + np.random.normal(0, 1.5))
            return [round(t_h, 2), round(t_p, 2), round(t_s, 2)]
            
        tar_1m = project_horizon(1.0, traj_type, cur_health, cur_pig, cur_sev)
        tar_3m = project_horizon(3.0, traj_type, cur_health, cur_pig, cur_sev)
        tar_6m = project_horizon(6.0, traj_type, cur_health, cur_pig, cur_sev)
        
        sequences.append(seq_steps)
        targets_1m.append(tar_1m)
        targets_3m.append(tar_3m)
        targets_6m.append(tar_6m)
        trend_labels.append(traj_type)
        
    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets_1m, dtype=np.float32),
        np.array(targets_3m, dtype=np.float32),
        np.array(targets_6m, dtype=np.float32),
        np.array(trend_labels, dtype=np.int64)
    )


# ---------------------------------------------------------------------------
# BiLSTM Clinical & Patient Journal Text Dataset Engine
# ---------------------------------------------------------------------------

CLINICAL_JOURNAL_CORPUS = [
    # Positive / Improving (Class 0)
    ("Patient reports hyperpigmentation significantly faded after 6 weeks of topical niacinamide and azelaic acid.", 0),
    ("Follow-up exam shows melasma patches on bilateral cheeks reduced by approximately 40%. No erythema observed.", 0),
    ("Lesion border has contracted. Dark pigmentation intensity is noticeably reduced compared to baseline visit.", 0),
    ("Patient adhered strictly to broad-spectrum SPF50. Skin texture improved, dark spots visibly lighter.", 0),
    ("Dermoscopic examination indicates regression of pigmented network. Patient tolerates retinoid treatment well.", 0),
    ("Freckles and sun spots have reduced in size and contrast after 3 months of consistent skincare.", 0),
    ("Acne marks are almost completely resolved. Minimal residual post-inflammatory hyperpigmentation.", 0),
    ("Excellent treatment adherence. Patient reports no itching or tenderness; pigmentation area shrinking.", 0),
    ("Significant clearance of actinic keratosis lesion following 4-week topical field therapy.", 0),
    ("Follow-up photograph confirms positive response. Melanin distribution appears more uniform across forehead.", 0),
    
    # Neutral / Monitoring (Class 1)
    ("Lesion appears stable in size, border, and color. Continued routine dermatological monitoring advised.", 1),
    ("Patient reports no noticeable change in pigmentation intensity over the past 4 weeks.", 1),
    ("Pigmented macule unchanged from previous checkup. Diameter remains 5.2 mm with uniform coloration.", 1),
    ("Routine skin checkup. Nevus shows typical symmetric pigment pattern without signs of evolution.", 1),
    ("Skin quality stable. Patient continues maintenance regimen with moisturization and sunscreen.", 1),
    ("Follow-up at 3 months reveals stable pigmentation area. Recommended to maintain current regimen.", 1),
    ("Dermoscopic evaluation confirms benign symmetric pattern. Re-evaluate in 6 months for progression.", 1),
    ("Patient notes slight seasonal pigmentation variation without structural growth or textural change.", 1),
    ("Pigmented spot shows stationary dimensions. No symptoms of itching, elevation, or bleeding reported.", 1),
    ("Baseline evaluation documented. Ongoing quarterly tracking scheduled to assess long-term trend.", 1),
    
    # Adverse / Irritation / Worsening (Class 2)
    ("Patient developed moderate contact dermatitis and severe peeling with redness after tretinoin application.", 2),
    ("Lesion has expanded rapidly in diameter with new irregular borders and mottled hyperpigmentation.", 2),
    ("Patient reports persistent pruritus, burning sensation, and oozing around the pigmented plaque.", 2),
    ("Follow-up reveals new dark asymmetry and elevated nodule. Urgent excision biopsy recommended.", 2),
    ("Severe irritation and post-inflammatory hyperpigmentation rebound after sun exposure without protection.", 2),
    ("Lesion has changed color from light brown to dark black with focal ulceration.", 2),
    ("Patient discontinued serum due to intense burning, erythema, and marked localized swelling.", 2),
    ("Significant lesion growth observed from 8 mm to 14 mm over 3 months with recurrent bleeding.", 2),
    ("Increased hyperpigmentation contrast and active inflammatory flare-up around jawline.", 2),
    ("Lesion exhibits high dermatological suspicion: spontaneous bleeding, itch, and color variegation.", 2)
]

class ClinicalJournalDataset(Dataset):
    """
    PyTorch Dataset for Clinical Notes & Skincare Journal BiLSTM classification.
    """
    def __init__(self, texts, labels, vocab, max_len=30):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = tokenize_text(text)
        indexed = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens][:self.max_len]
        
        # Padding
        if len(indexed) < self.max_len:
            indexed += [0] * (self.max_len - len(indexed))
            
        return {
            'input_ids': torch.tensor(indexed, dtype=torch.long),
            'label': self.labels[idx],
            'length': min(len(tokens), self.max_len)
        }

def tokenize_text(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_vocab(corpus, min_freq=1):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word_counts = {}
    for text, _ in corpus:
        tokens = tokenize_text(text)
        for t in tokens:
            word_counts[t] = word_counts.get(t, 0) + 1
            
    for word, count in sorted(word_counts.items()):
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def generate_augmented_clinical_corpus(multiplier=20, seed=42):
    """
    Expands the clinical corpus for robust neural training using paraphrasing and medical variants.
    """
    np.random.seed(seed)
    augmented = []
    modifiers_pos = ["Noticeable", "Clear", "Documented", "Patient shows", "Recent checkup confirms"]
    modifiers_neu = ["Routine check shows", "Clinical exam indicates", "Observational follow-up reveals"]
    modifiers_adv = ["Urgent alert:", "Severe reaction:", "Significant concern:", "Patient complains of"]
    
    for base_text, label in CLINICAL_JOURNAL_CORPUS:
        augmented.append((base_text, label))
        for _ in range(multiplier - 1):
            if label == 0:
                prefix = np.random.choice(modifiers_pos)
            elif label == 1:
                prefix = np.random.choice(modifiers_neu)
            else:
                prefix = np.random.choice(modifiers_adv)
            aug_text = f"{prefix} {base_text.lower()}"
            augmented.append((aug_text, label))
            
    np.random.shuffle(augmented)
    return augmented
