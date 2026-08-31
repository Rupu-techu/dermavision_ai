import numpy as np
import torch

# Exact 78 feature column order expected by PAD_UFES_20_FINAL_ANN.pth PyTorch model
FEATURE_NAMES = [
    "age",
    "fitspatrick",
    "diameter_1",
    "diameter_2",
    "smoke_False",
    "smoke_True",
    "drink_False",
    "drink_True",
    "background_father_AUSTRIA",
    "background_father_BRASIL",
    "background_father_BRAZIL",
    "background_father_CZECH",
    "background_father_GERMANY",
    "background_father_ISRAEL",
    "background_father_ITALY",
    "background_father_NETHERLANDS",
    "background_father_POLAND",
    "background_father_POMERANIA",
    "background_father_PORTUGAL",
    "background_father_SPAIN",
    "background_father_UNK",
    "background_mother_BRAZIL",
    "background_mother_FRANCE",
    "background_mother_GERMANY",
    "background_mother_ITALY",
    "background_mother_NETHERLANDS",
    "background_mother_NORWAY",
    "background_mother_POLAND",
    "background_mother_POMERANIA",
    "background_mother_PORTUGAL",
    "background_mother_SPAIN",
    "background_mother_UNK",
    "pesticide_False",
    "pesticide_True",
    "gender_FEMALE",
    "gender_MALE",
    "skin_cancer_history_False",
    "skin_cancer_history_True",
    "cancer_history_False",
    "cancer_history_True",
    "has_piped_water_False",
    "has_piped_water_True",
    "has_sewage_system_False",
    "has_sewage_system_True",
    "region_ABDOMEN",
    "region_ARM",
    "region_BACK",
    "region_CHEST",
    "region_EAR",
    "region_FACE",
    "region_FOOT",
    "region_FOREARM",
    "region_HAND",
    "region_LIP",
    "region_NECK",
    "region_NOSE",
    "region_SCALP",
    "region_THIGH",
    "itch_False",
    "itch_True",
    "itch_UNK",
    "grew_False",
    "grew_True",
    "grew_UNK",
    "hurt_False",
    "hurt_True",
    "hurt_UNK",
    "changed_False",
    "changed_True",
    "changed_UNK",
    "bleed_False",
    "bleed_True",
    "bleed_UNK",
    "elevation_False",
    "elevation_True",
    "elevation_UNK",
    "biopsed_False",
    "biopsed_True"
]

# Scaler mean and std for numerical feature standardization (StandardScaler fitted during training)
SCALER_MEANS = {
    'age': 60.4647,
    'fitspatrick': 2.1728,
    'diameter_1': 11.2333,
    'diameter_2': 8.5540
}

SCALER_STDS = {
    'age': 15.8914,
    'fitspatrick': 0.6011,
    'diameter_1': 7.0183,
    'diameter_2': 4.6903
}

def preprocess_user_input(raw_input: dict) -> torch.Tensor:
    """
    Transforms raw user input dictionary into standard 78-feature PyTorch tensor.
    
    Expected raw_input keys:
    - age: int/float (e.g. 55)
    - fitspatrick: int (1 to 6)
    - diameter_1: float (mm)
    - diameter_2: float (mm)
    - smoke: bool / 'True' / 'False'
    - drink: bool / 'True' / 'False'
    - background_father: str (e.g. 'GERMANY', 'BRAZIL', etc.)
    - background_mother: str (e.g. 'ITALY', 'POMERANIA', etc.)
    - pesticide: bool
    - gender: str ('MALE' / 'FEMALE')
    - skin_cancer_history: bool
    - cancer_history: bool
    - has_piped_water: bool / 'UNK'
    - has_sewage_system: bool / 'UNK'
    - region: str (e.g. 'FACE', 'ARM', 'CHEST', etc.)
    - itch: bool / 'UNK'
    - grew: bool / 'UNK'
    - hurt: bool / 'UNK'
    - changed: bool / 'UNK'
    - bleed: bool / 'UNK'
    - elevation: bool / 'UNK'
    - biopsed: bool
    """
    feature_dict = {col: 0.0 for col in FEATURE_NAMES}
    
    # 1. Numerical Features (StandardScaler Z-score normalization)
    age = float(raw_input.get('age', 50))
    fitspatrick = float(raw_input.get('fitspatrick', 2))
    diameter_1 = float(raw_input.get('diameter_1', 10.0))
    diameter_2 = float(raw_input.get('diameter_2', 8.0))
    
    feature_dict['age'] = (age - SCALER_MEANS['age']) / SCALER_STDS['age']
    feature_dict['fitspatrick'] = (fitspatrick - SCALER_MEANS['fitspatrick']) / SCALER_STDS['fitspatrick']
    feature_dict['diameter_1'] = (diameter_1 - SCALER_MEANS['diameter_1']) / SCALER_STDS['diameter_1']
    feature_dict['diameter_2'] = (diameter_2 - SCALER_MEANS['diameter_2']) / SCALER_STDS['diameter_2']
    
    # Helper to set boolean / UNK columns
    def set_bool_unk(prefix, val):
        str_val = str(val).strip().upper()
        if str_val in ['TRUE', '1', 'YES']:
            col = f"{prefix}_True"
        elif str_val in ['FALSE', '0', 'NO']:
            col = f"{prefix}_False"
        else:
            col = f"{prefix}_UNK"
            
        if col in feature_dict:
            feature_dict[col] = 1.0
        elif f"{prefix}_False" in feature_dict:
            feature_dict[f"{prefix}_False"] = 1.0
            
    # Binary / Tri-state columns
    set_bool_unk('smoke', raw_input.get('smoke', False))
    set_bool_unk('drink', raw_input.get('drink', False))
    set_bool_unk('pesticide', raw_input.get('pesticide', False))
    set_bool_unk('skin_cancer_history', raw_input.get('skin_cancer_history', False))
    set_bool_unk('cancer_history', raw_input.get('cancer_history', False))
    set_bool_unk('has_piped_water', raw_input.get('has_piped_water', True))
    set_bool_unk('has_sewage_system', raw_input.get('has_sewage_system', True))
    
    set_bool_unk('itch', raw_input.get('itch', False))
    set_bool_unk('grew', raw_input.get('grew', False))
    set_bool_unk('hurt', raw_input.get('hurt', False))
    set_bool_unk('changed', raw_input.get('changed', False))
    set_bool_unk('bleed', raw_input.get('bleed', False))
    set_bool_unk('elevation', raw_input.get('elevation', False))
    set_bool_unk('biopsed', raw_input.get('biopsed', False))
    
    # Gender
    gender_str = str(raw_input.get('gender', 'MALE')).upper()
    if gender_str == 'FEMALE':
        feature_dict['gender_FEMALE'] = 1.0
    else:
        feature_dict['gender_MALE'] = 1.0
        
    # Father Background
    father_str = str(raw_input.get('background_father', 'UNK')).upper()
    father_col = f"background_father_{father_str}"
    if father_col in feature_dict:
        feature_dict[father_col] = 1.0
    else:
        feature_dict['background_father_UNK'] = 1.0
        
    # Mother Background
    mother_str = str(raw_input.get('background_mother', 'UNK')).upper()
    mother_col = f"background_mother_{mother_str}"
    if mother_col in feature_dict:
        feature_dict[mother_col] = 1.0
    else:
        feature_dict['background_mother_UNK'] = 1.0
        
    # Body Region
    region_str = str(raw_input.get('region', 'FACE')).upper()
    region_col = f"region_{region_str}"
    if region_col in feature_dict:
        feature_dict[region_col] = 1.0
    else:
        feature_dict['region_FACE'] = 1.0
        
    # Build vector in exact 78 feature order
    feature_vector = np.array([feature_dict[col] for col in FEATURE_NAMES], dtype=np.float32)
    tensor_input = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0) # Shape (1, 78)
    
    return tensor_input
