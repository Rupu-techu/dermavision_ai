import os
import sys
import torch

# Ensure UTF-8 output encoding for stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, send_from_directory

# Ensure src is in Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.model import load_trained_model
from src.preprocessing import preprocess_user_input
from src.score_engine import calculate_clinical_scores

app = Flask(__name__, static_folder='static', static_url_path='')

# Model Initialization
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'PAD_UFES_20_deep_best_model.pth')
print(f"Loading DermaVision PyTorch DeepANN Model from: {MODEL_PATH}")

try:
    model = load_trained_model(MODEL_PATH)
    print("SUCCESS: PyTorch DeepANN Model loaded into memory!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "model_name": "DeepANN",
        "model_loaded": model is not None,
        "input_features": 78,
        "classes": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "PyTorch ANN Model is not loaded"}), 500
        
    try:
        data = request.get_json() or {}
        
        # 1. Preprocess raw input to 78-feature tensor (1, 78)
        tensor_input = preprocess_user_input(data)
        
        # 2. PyTorch Forward Pass
        with torch.no_grad():
            logits = model(tensor_input)
            probabilities = torch.softmax(logits, dim=1)
            
        # 3. Calculate Clinical Scores & Risk Metrics
        result = calculate_clinical_scores(probabilities, data)
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/api/presets', methods=['GET'])
def get_presets():
    presets = {
        "melanoma_risk": {
            "title": "High-Risk Melanoma Suspect",
            "age": 68,
            "fitspatrick": 1,
            "diameter_1": 14.0,
            "diameter_2": 11.5,
            "gender": "MALE",
            "smoke": True,
            "drink": True,
            "background_father": "GERMANY",
            "background_mother": "NORWAY",
            "pesticide": False,
            "skin_cancer_history": True,
            "cancer_history": True,
            "has_piped_water": True,
            "has_sewage_system": True,
            "region": "BACK",
            "itch": True,
            "grew": True,
            "hurt": False,
            "changed": True,
            "bleed": True,
            "elevation": True,
            "biopsed": True
        },
        "bcc_malignant": {
            "title": "Basal Cell Carcinoma Case",
            "age": 62,
            "fitspatrick": 2,
            "diameter_1": 9.5,
            "diameter_2": 7.0,
            "gender": "FEMALE",
            "smoke": False,
            "drink": False,
            "background_father": "ITALY",
            "background_mother": "SPAIN",
            "pesticide": False,
            "skin_cancer_history": False,
            "cancer_history": True,
            "has_piped_water": True,
            "has_sewage_system": True,
            "region": "NOSE",
            "itch": False,
            "grew": True,
            "hurt": False,
            "changed": True,
            "bleed": True,
            "elevation": True,
            "biopsed": False
        },
        "benign_nevus": {
            "title": "Benign Mole (Nevus)",
            "age": 28,
            "fitspatrick": 3,
            "diameter_1": 4.5,
            "diameter_2": 4.0,
            "gender": "FEMALE",
            "smoke": False,
            "drink": False,
            "background_father": "BRAZIL",
            "background_mother": "PORTUGAL",
            "pesticide": False,
            "skin_cancer_history": False,
            "cancer_history": False,
            "has_piped_water": True,
            "has_sewage_system": True,
            "region": "ARM",
            "itch": False,
            "grew": False,
            "hurt": False,
            "changed": False,
            "bleed": False,
            "elevation": False,
            "biopsed": False
        },
        "actinic_keratosis": {
            "title": "Actinic Keratosis (Pre-cancerous)",
            "age": 74,
            "fitspatrick": 2,
            "diameter_1": 8.0,
            "diameter_2": 6.5,
            "gender": "MALE",
            "smoke": True,
            "drink": True,
            "background_father": "POMERANIA",
            "background_mother": "GERMANY",
            "pesticide": True,
            "skin_cancer_history": True,
            "cancer_history": False,
            "has_piped_water": True,
            "has_sewage_system": True,
            "region": "FACE",
            "itch": True,
            "grew": True,
            "hurt": True,
            "changed": True,
            "bleed": False,
            "elevation": False,
            "biopsed": False
        }
    }
    return jsonify(presets)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
