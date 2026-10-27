"""
DermaVision AI — Module 4 Integration & Unit Test Suite
=========================================================
Tests:
1. Longitudinal Sequence Dataset Generation & Feature Shapes
2. SkinProgressionLSTM Forward Pass & Multi-Horizon Heads (1m, 3m, 6m)
3. SkinProgressionGRU Architecture Forward Pass
4. ClinicalJournalBiLSTM NLP Classifier
5. End-to-end Multi-Horizon Forecasting Function
6. Clinical Notes Inference Pipeline

Author: Mouli Mukherjee (Module 4)
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import torch
from src.module4_models import SkinProgressionLSTM, SkinProgressionGRU, ClinicalJournalBiLSTM
from src.module4_sequence_dataset import (
    LongitudinalSkinDataset,
    build_longitudinal_dataset,
    build_vocab,
    CLINICAL_JOURNAL_CORPUS
)
from src.module4_forecast import predict_skin_progression, analyze_clinical_notes


class TestModule4Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[Test Setup] Running Module 4 test suite on: {cls.device.upper()}")

    def test_01_dataset_generation(self):
        """Verify longitudinal sequence formulation from PAD-UFES-20 metadata."""
        seqs, t1, t3, t6, trends = build_longitudinal_dataset(base_dir=BASE_DIR, seq_len=4, seed=42)
        
        self.assertGreater(len(seqs), 1000, "Dataset should have over 1,000 patient sequences")
        self.assertEqual(seqs.shape[1:], (4, 8), f"Expected shape (N, 4, 8), got {seqs.shape}")
        self.assertEqual(t1.shape[1], 3, f"Expected 3 target metrics for 1M, got {t1.shape[1]}")
        self.assertEqual(t3.shape[1], 3, f"Expected 3 target metrics for 3M, got {t3.shape[1]}")
        self.assertEqual(t6.shape[1], 3, f"Expected 3 target metrics for 6M, got {t6.shape[1]}")
        self.assertTrue(set(np.unique(trends)).issubset({0, 1, 2}), "Trend labels should be in {0, 1, 2}")
        print("   [PASS] Longitudinal dataset generation & feature shapes verified.")

    def test_02_lstm_forward_pass(self):
        """Verify SkinProgressionLSTM multi-horizon output shapes."""
        model = SkinProgressionLSTM(input_dim=8, hidden_dim=64, num_layers=2).to(self.device)
        model.eval()
        
        dummy_input = torch.randn(4, 4, 8, dtype=torch.float32).to(self.device) # (batch=4, seq=4, feat=8)
        with torch.no_grad():
            outputs = model(dummy_input)
            
        self.assertIn('forecast_1m', outputs)
        self.assertIn('forecast_3m', outputs)
        self.assertIn('forecast_6m', outputs)
        self.assertIn('trend_logits', outputs)
        
        self.assertEqual(outputs['forecast_1m'].shape, (4, 3))
        self.assertEqual(outputs['forecast_3m'].shape, (4, 3))
        self.assertEqual(outputs['forecast_6m'].shape, (4, 3))
        self.assertEqual(outputs['trend_logits'].shape, (4, 3))
        print("   [PASS] SkinProgressionLSTM multi-horizon forward pass verified.")

    def test_03_gru_forward_pass(self):
        """Verify SkinProgressionGRU forward pass."""
        model = SkinProgressionGRU(input_dim=8, hidden_dim=64, num_layers=2).to(self.device)
        model.eval()
        
        dummy_input = torch.randn(2, 4, 8, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = model(dummy_input)
            
        self.assertEqual(outputs['forecast_1m'].shape, (2, 3))
        self.assertEqual(outputs['forecast_3m'].shape, (2, 3))
        self.assertEqual(outputs['forecast_6m'].shape, (2, 3))
        self.assertEqual(outputs['trend_logits'].shape, (2, 3))
        print("   [PASS] SkinProgressionGRU forward pass verified.")

    def test_04_bilstm_notes_forward_pass(self):
        """Verify ClinicalJournalBiLSTM NLP token index handling and output logits."""
        vocab = build_vocab(CLINICAL_JOURNAL_CORPUS)
        model = ClinicalJournalBiLSTM(vocab_size=len(vocab), embed_dim=64, hidden_dim=64, num_classes=3).to(self.device)
        model.eval()
        
        dummy_tokens = torch.randint(0, len(vocab), (3, 20), dtype=torch.long).to(self.device) # batch 3, len 20
        with torch.no_grad():
            logits = model(dummy_tokens)
            
        self.assertEqual(logits.shape, (3, 3), f"Expected logits shape (3, 3), got {logits.shape}")
        probs = torch.softmax(logits, dim=1)
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(3, device=self.device), atol=1e-4))
        print("   [PASS] ClinicalJournalBiLSTM text logits and softmax verified.")

    def test_05_predict_skin_progression_forecast(self):
        """Verify high-level prediction formatting for dashboard integration."""
        model = SkinProgressionLSTM(input_dim=8, hidden_dim=64, num_layers=2).to(self.device)
        model.eval()
        
        # 4 consecutive visits
        sample_visit_history = [
            [0.85, 0.25, 0.15, 0.35, 0.33, 0.27, 0.17, 0.90], # Visit 1
            [0.87, 0.23, 0.13, 0.33, 0.32, 0.26, 0.17, 0.92], # Visit 2
            [0.89, 0.21, 0.11, 0.30, 0.30, 0.25, 0.00, 0.95], # Visit 3
            [0.91, 0.19, 0.09, 0.28, 0.29, 0.24, 0.00, 0.95]  # Visit 4 (Current)
        ]
        
        forecast = predict_skin_progression(model, sample_visit_history, device=self.device)
        
        self.assertIn('trajectory_category', forecast)
        self.assertIn('forecast_1_month', forecast)
        self.assertIn('forecast_3_months', forecast)
        self.assertIn('forecast_6_months', forecast)
        self.assertIn('timeline_chart_points', forecast)
        
        self.assertEqual(len(forecast['timeline_chart_points']), 4)
        for h in ['forecast_1_month', 'forecast_3_months', 'forecast_6_months']:
            self.assertIn('skin_health_score', forecast[h])
            self.assertIn('pigmentation_percentage', forecast[h])
            self.assertIn('clinical_severity_score', forecast[h])
            self.assertTrue(0.0 <= forecast[h]['skin_health_score'] <= 100.0)
            self.assertTrue(0.0 <= forecast[h]['pigmentation_percentage'] <= 100.0)
            
        print("   [PASS] End-to-end multi-horizon forecast data verified.")

    def test_06_analyze_clinical_notes_nlp(self):
        """Verify high-level clinical notes analysis function."""
        vocab = build_vocab(CLINICAL_JOURNAL_CORPUS)
        model = ClinicalJournalBiLSTM(vocab_size=len(vocab), embed_dim=64, hidden_dim=64, num_classes=3).to(self.device)
        model.eval()
        
        sample_note = "Patient reports dark spots on cheeks have visibly faded after 6 weeks of sunscreen use."
        res = analyze_clinical_notes(model, vocab, sample_note, device=self.device)
        
        self.assertIn('feedback_category', res)
        self.assertIn('confidence', res)
        self.assertIn('probabilities', res)
        self.assertEqual(len(res['probabilities']), 3)
        print("   [PASS] Clinical notes NLP inference function verified.")


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING DERMAVISION MODULE 4 UNIT & INTEGRATION TESTS")
    print("=" * 70)
    unittest.main()
