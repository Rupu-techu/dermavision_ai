import os
import sys
import torch

# Add root project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import load_trained_model, DeepSkinDiseaseANN
from src.preprocessing import preprocess_user_input, FEATURE_NAMES
from src.score_engine import calculate_clinical_scores

def test_full_pipeline():
    print("=" * 70)
    print("DERMAVISION MODULE 1 PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # 1. Test Feature Names Count
    print(f"\n1. Feature Names Count: {len(FEATURE_NAMES)}")
    assert len(FEATURE_NAMES) == 78, f"Expected 78 features, got {len(FEATURE_NAMES)}"
    print("   [PASSED] Feature vector size verified (78 features).")

    # 2. Test Raw Input Preprocessing
    sample_input = {
        'age': 65,
        'fitspatrick': 3,
        'diameter_1': 12.5,
        'diameter_2': 10.0,
        'gender': 'MALE',
        'smoke': True,
        'drink': False,
        'background_father': 'GERMANY',
        'background_mother': 'ITALY',
        'pesticide': False,
        'skin_cancer_history': True,
        'cancer_history': True,
        'has_piped_water': True,
        'has_sewage_system': True,
        'region': 'FACE',
        'itch': True,
        'grew': True,
        'hurt': False,
        'changed': True,
        'bleed': False,
        'elevation': True,
        'biopsed': False
    }

    tensor_input = preprocess_user_input(sample_input)
    print(f"\n2. Preprocessed Tensor Shape: {tensor_input.shape}")
    assert tensor_input.shape == (1, 78), f"Expected shape (1, 78), got {tensor_input.shape}"
    print("   [PASSED] Preprocessing output shape is (1, 78).")

    # 3. Model Loading & Inference
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "PAD_UFES_20_deep_best_model.pth"))
    print(f"\n3. Loading PyTorch DeepANN Model from: {model_path}")
    assert os.path.exists(model_path), f"Model path not found: {model_path}"

    model = load_trained_model(model_path)
    print("   [PASSED] PyTorch DeepANN Model loaded successfully.")

    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1)

    print(f"\n4. Model Softmax Probabilities:\n   {probs.numpy()}")
    assert probs.shape == (1, 6), f"Expected shape (1, 6), got {probs.shape}"
    assert abs(float(probs.sum()) - 1.0) < 1e-4, "Probabilities do not sum to 1"
    print("   [PASSED] Softmax probabilities calculated correctly.")

    # 4. Clinical Score Engine Test
    scores = calculate_clinical_scores(probs, sample_input)
    print("\n5. DermaVision Score Engine Output:")
    for key, value in scores.items():
        print(f"   - {key}: {value}")

    assert 'diagnosis_code' in scores
    assert 'prediction_confidence' in scores
    assert 'skin_health_score' in scores
    assert 'pigmentation_risk_score' in scores
    assert 'severity_category' in scores
    print("\n   [PASSED] All score engine metrics calculated successfully!")
    print("=" * 70)
    print("PIPELINE TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_full_pipeline()
