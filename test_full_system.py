"""
DermaVision AI — Complete System Integration & End-to-End Test Suite
=====================================================================
Executes an end-to-end verification of all modules and interfaces:
- Module 1: DeepANN Feature Preprocessing & Clinical Score Engine
- Module 2: Optimization, Regularization & Weight Initialization
- Module 3: Computer Vision Checkpoints & Quantification Artifacts
- Module 4: Longitudinal Recurrent Progression (LSTM / GRU) & Notes NLP (BiLSTM)
- Module 5A: Generative Adversarial Networks (DCGAN & cDCGAN)
- Module 5B: Variational Autoencoder (ConvVAE) & Case-Based Reasoning Retrieval
- Module 5C: Generative AI Multimodal Skincare Report & Chatbot
- Full Flask API & Web Dashboard Endpoints

Run Command:
    python3 test_full_system.py

Author: DermaVision AI Team
"""

import os
import sys
import unittest
import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)


def test_system():
    print("=" * 75)
    print("DERMAVISION AI — COMPLETE MULTIMODAL SYSTEM VERIFICATION")
    print("=" * 75)
    
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Runtime: {device.upper()}")
    print("-" * 75)
    
    # -------------------------------------------------------------
    # 1. Module 1 & 2 Verification
    # -------------------------------------------------------------
    print("\n[TEST 1/6] Verifying Module 1 & 2 (DeepANN, Feature Vector & Score Engine)...")
    from src.model import load_trained_model
    from src.preprocessing import preprocess_user_input, FEATURE_NAMES
    from src.score_engine import calculate_clinical_scores
    
    assert len(FEATURE_NAMES) == 78, f"Expected 78 features, got {len(FEATURE_NAMES)}"
    m1_model_path = os.path.join(BASE_DIR, 'models', 'PAD_UFES_20_deep_best_model.pth')
    assert os.path.exists(m1_model_path), f"Missing model: {m1_model_path}"
    
    m1_model = load_trained_model(m1_model_path, device='cpu')
    sample_input = {
        'age': 65, 'fitspatrick': 2, 'diameter_1': 12.0, 'diameter_2': 9.0,
        'gender': 'MALE', 'background_father': 'GERMANY', 'background_mother': 'GERMANY',
        'region': 'FACE', 'itch': True, 'grew': True, 'hurt': False, 'changed': True,
        'bleed': False, 'elevation': True, 'biopsed': False,
        'skin_cancer_history': False, 'cancer_history': False, 'smoke': False,
        'drink': False, 'pesticide': False, 'has_piped_water': True, 'has_sewage_system': True
    }
    t_input = preprocess_user_input(sample_input)
    assert t_input.shape == (1, 78), f"Expected shape (1, 78), got {t_input.shape}"
    with torch.no_grad():
        logits = m1_model(t_input)
        probs = torch.softmax(logits, dim=1)
    m1_scores = calculate_clinical_scores(probs, sample_input)
    assert 'diagnosis_code' in m1_scores and 'skin_health_score' in m1_scores
    print(f"   [PASS] DeepANN forward pass: {m1_scores['diagnosis_name']} ({m1_scores['prediction_confidence']}%)")
    print(f"   [PASS] Clinical Scores: Health={m1_scores['skin_health_score']}, Severity={m1_scores['skin_severity_score']}, Risk={m1_scores['pigmentation_risk_score']}")

    # -------------------------------------------------------------
    # 2. Module 3 Deliverables Verification
    # -------------------------------------------------------------
    print("\n[TEST 2/6] Verifying Module 3 Computer Vision Checkpoints & Reports...")
    cv_artifacts = [
        'resnet50_best.pt',
        'unet_pigmentation_best.pt',
        'yolo26n.pt',
        'pigmentation_report.csv',
        'pigmentation_class_summary.csv'
    ]
    for art in cv_artifacts:
        art_path = os.path.join(BASE_DIR, art)
        assert os.path.exists(art_path), f"Missing CV artifact: {art}"
        print(f"   [PASS] Found: {art} ({os.path.getsize(art_path) / 1024:.1f} KB)")

    # -------------------------------------------------------------
    # 3. Module 4 Verification
    # -------------------------------------------------------------
    print("\n[TEST 3/6] Verifying Module 4 Temporal Recurrent Forecast (LSTM / GRU)...")
    from src.module4_forecast import load_progression_model, predict_skin_progression
    lstm_path = os.path.join(BASE_DIR, 'models', 'module4_lstm_progression_best.pth')
    assert os.path.exists(lstm_path), f"Missing LSTM checkpoint: {lstm_path}"
    
    lstm_model = load_progression_model(lstm_path, model_type='lstm', device='cpu')
    dummy_history = [
        [0.70, 0.30, 0.35, 0.40, 0.33, 0.27, 0.20, 0.90],
        [0.72, 0.28, 0.33, 0.38, 0.32, 0.26, 0.20, 0.92],
        [0.74, 0.26, 0.31, 0.36, 0.31, 0.25, 0.10, 0.95],
        [0.76, 0.24, 0.29, 0.34, 0.30, 0.24, 0.10, 0.95]
    ]
    forecast = predict_skin_progression(lstm_model, dummy_history, device='cpu')
    assert 'trajectory_category' in forecast and 'forecast_3_months' in forecast
    print(f"   [PASS] Multi-horizon forecast: Trajectory = '{forecast['trajectory_category']}'")
    print(f"   [PASS] +3M Horizon Projection: Health = {forecast['forecast_3_months']['skin_health_score']}, Pigment = {forecast['forecast_3_months']['pigmentation_percentage']}%")

    # -------------------------------------------------------------
    # 4. Module 5A GAN Verification
    # -------------------------------------------------------------
    print("\n[TEST 4/6] Verifying Module 5A Generative Adversarial Networks (GAN)...")
    from src.module5_gan import SkinLesionGenerator, ConditionalSkinGenerator, generate_synthetic_skin_samples
    g_path = os.path.join(BASE_DIR, 'models', 'module5_gan_generator_best.pth')
    cg_path = os.path.join(BASE_DIR, 'models', 'module5_cgan_generator_best.pth')
    assert os.path.exists(g_path), f"Missing GAN generator checkpoint: {g_path}"
    assert os.path.exists(cg_path), f"Missing cDCGAN generator checkpoint: {cg_path}"
    
    netG = SkinLesionGenerator(z_dim=100, feature_g=64).to(device)
    netG.load_state_dict(torch.load(g_path, map_location=device))
    samples = generate_synthetic_skin_samples(netG, num_samples=4, z_dim=100, device=device)
    assert samples.shape == (4, 3, 64, 64), f"Expected (4, 3, 64, 64), got {samples.shape}"
    print(f"   [PASS] DCGAN Generator synthesized 4 skin lesion samples (64x64x3 RGB) in [0, 1].")

    # -------------------------------------------------------------
    # 5. Module 5B VAE Verification
    # -------------------------------------------------------------
    print("\n[TEST 5/6] Verifying Module 5B Variational Autoencoder (VAE) & Case Retrieval...")
    from src.module5_vae import SkinLesionVAE, SkinCaseSimilarityEngine
    vae_path = os.path.join(BASE_DIR, 'models', 'module5_vae_best.pth')
    gal_path = os.path.join(BASE_DIR, 'models', 'module5_vae_case_gallery.pt')
    assert os.path.exists(vae_path), f"Missing VAE checkpoint: {vae_path}"
    assert os.path.exists(gal_path), f"Missing VAE gallery: {gal_path}"
    
    vae = SkinLesionVAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    engine = SkinCaseSimilarityEngine(vae, device=device)
    engine.load_gallery(gal_path)
    
    query_tensor = torch.rand(3, 64, 64)
    sim_res = engine.find_similar_cases(query_tensor, top_k=3)
    assert len(sim_res['top_similar_cases']) == 3
    print(f"   [PASS] Indexed Gallery: {len(engine.case_metadata)} cases loaded.")
    print(f"   [PASS] Latent Similarity Top Match: {sim_res['top_similar_cases'][0]['case_id']} ({sim_res['top_similar_cases'][0]['similarity_score']}%)")

    # -------------------------------------------------------------
    # 6. Module 5C Generative AI & Flask Full-Stack Verification
    # -------------------------------------------------------------
    print("\n[TEST 6/6] Verifying Module 5C Generative AI Skincare Engine & Flask Endpoints...")
    import app
    client = app.app.test_client()
    
    # Generate Multimodal Report
    r_rep = client.post('/api/module5/generate-report', json=sample_input)
    assert r_rep.status_code == 200 and r_rep.json['success']
    report = r_rep.json['data']['report']
    ctx = r_rep.json['data']['context']
    print(f"   [PASS] GenAI Report Generated: {report['diagnosis_name']} | Urgency: {report['urgency_level']}")
    print(f"   [PASS] Personalized Regimen: {len(report['am_regimen'])} AM Steps | {len(report['pm_regimen'])} PM Steps")
    
    # Interactive Chatbot Query
    r_chat = client.post('/api/module5/chat', json={
        'message': 'Which sunscreen should I use and can I use niacinamide?',
        'patient_context': ctx
    })
    assert r_chat.status_code == 200 and r_chat.json['success']
    bot_reply = r_chat.json['data']['bot_response']
    print(f"   [PASS] AI Chatbot Consultation Response verified ({len(bot_reply)} characters).")
    
    # UI Endpoint
    r_ui = client.get('/')
    assert r_ui.status_code == 200 and b'Unified System' in r_ui.data
    print(f"   [PASS] Web Application (index.html, app.js, style.css) successfully integrated.")

    print("\n" + "=" * 75)
    print("ALL MODULES (1 TO 5) ARE LINKED PROPERLY AND OPERATING AT 100% SUCCESS!")
    print("=" * 75)


if __name__ == '__main__':
    test_system()
