"""
DermaVision AI — Module 5: Generative AI Unit & Integration Test Suite
======================================================================
Tests:
1. Multimodal Patient Context Synthesizer (M1 + M3 + M4 + M5 Fusion)
2. Clinical & Skincare Report Generation
3. AM/PM Regimen Structure and Active Ingredients
4. Plain Language Diagnostic Translation
5. Interactive Chatbot Sunscreen Guidance
6. Interactive Chatbot Ingredient & Contraindication Advice
7. Interactive Chatbot Trajectory & Forecast Explanation
8. Stateless Chat API Functional Helper

Author: DermaVision AI Team (Module 5)
"""

import os
import sys
import unittest

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.module5_genai import (
    synthesize_patient_context,
    generate_clinical_and_skincare_report,
    DermaVisionConsultationBot,
    chat_with_dermavision_bot,
    CLINICAL_KNOWLEDGE_BASE
)


class TestModule5GenAI(unittest.TestCase):

    def setUp(self):
        # Simulated M1 score engine output
        self.sample_m1_scores = {
            'diagnosis_code': 'BCC',
            'diagnosis_name': 'Basal Cell Carcinoma',
            'prediction_confidence': 94.8,
            'skin_health_score': 55.0,
            'skin_health_category': 'Moderate',
            'skin_severity_score': 65.0,
            'severity_category': 'High',
            'pigmentation_risk_score': 70.0,
            'pigmentation_risk_category': 'High'
        }
        self.sample_patient_info = {
            'age': 62,
            'gender': 'FEMALE',
            'fitspatrick': 2,
            'region': 'NOSE',
            'grew': True,
            'bleed': True
        }
        self.sample_m3_cv = {
            'pigmentation_percentage': 28.5,
            'lesion_detected_boxes': 1
        }
        self.sample_m4_forecast = {
            'trajectory_category': 'Stable (Maintenance)',
            'forecast_1_month': {'skin_health_score': 56.0, 'pigmentation_percentage': 28.0},
            'forecast_3_months': {'skin_health_score': 58.0, 'pigmentation_percentage': 26.5},
            'forecast_6_months': {'skin_health_score': 60.0, 'pigmentation_percentage': 25.0}
        }
        self.sample_m5_vae = {
            'query_anomaly_score': 1.25,
            'anomaly_interpretation': 'Typical Lesion Morphology',
            'top_similar_cases': [
                {'case_id': 'DV-REF-0042', 'diagnosis_code': 'BCC', 'similarity_score': 92.4, 'clinical_outcome': 'Excised with clear margins'}
            ]
        }

    def test_01_multimodal_context_synthesis(self):
        """Verify seamless fusion of M1, M3, M4, and M5 outputs."""
        context = synthesize_patient_context(
            module1_scores=self.sample_m1_scores,
            patient_info=self.sample_patient_info,
            module3_cv=self.sample_m3_cv,
            module4_trajectory=self.sample_m4_forecast,
            module5_vae=self.sample_m5_vae
        )
        self.assertEqual(context['diagnosis_code'], 'BCC')
        self.assertEqual(context['skin_health_score'], 55.0)
        self.assertEqual(context['computer_vision_module3']['pigmentation_percentage'], 28.5)
        self.assertEqual(context['temporal_forecast_module4']['trajectory_category'], 'Stable (Maintenance)')
        self.assertEqual(len(context['case_similarity_module5']['top_similar_cases']), 1)
        print("   [PASS] Multimodal patient context synthesis verified.")

    def test_02_report_generation_structure(self):
        """Verify generated report contains all required clinical sections."""
        context = synthesize_patient_context(self.sample_m1_scores, self.sample_patient_info)
        report = generate_clinical_and_skincare_report(context)
        
        required_keys = [
            'diagnosis_code', 'diagnosis_name', 'urgency_level',
            'executive_summary', 'plain_language_explanation',
            'trajectory_narrative', 'am_regimen', 'pm_regimen',
            'target_actives', 'contraindicated', 'markdown_report'
        ]
        for k in required_keys:
            self.assertIn(k, report)
            
        self.assertGreater(len(report['markdown_report']), 200)
        print("   [PASS] Clinical & skincare report structure verified.")

    def test_03_am_pm_regimen_structure(self):
        """Verify AM and PM regimens include numbered steps, products, actives, and instructions."""
        context = synthesize_patient_context(self.sample_m1_scores, self.sample_patient_info)
        report = generate_clinical_and_skincare_report(context)
        
        self.assertGreaterEqual(len(report['am_regimen']), 3)
        self.assertGreaterEqual(len(report['pm_regimen']), 2)
        
        for step in report['am_regimen']:
            self.assertIn('step', step)
            self.assertIn('product', step)
            self.assertIn('active', step)
            self.assertIn('instruction', step)
            
        print("   [PASS] AM/PM structured regimens verified.")

    def test_04_plain_language_translation(self):
        """Verify empathetic, jargon-free diagnostic explanation."""
        for code, kb in CLINICAL_KNOWLEDGE_BASE.items():
            self.assertIn('plain_explanation', kb)
            self.assertGreater(len(kb['plain_explanation']), 50)
            self.assertIn('target_actives', kb)
            self.assertIn('contraindicated', kb)
            
        print("   [PASS] Plain language explanations & knowledge base verified.")

    def test_05_chatbot_sunscreen_inquiry(self):
        """Verify chatbot answers sunscreen queries with tailored SPF guidance."""
        context = synthesize_patient_context(self.sample_m1_scores, self.sample_patient_info)
        bot = DermaVisionConsultationBot(context)
        
        reply = bot.respond("Which sunscreen should I use and how often should I apply it?")
        self.assertIn("sunscreen", reply.lower())
        self.assertIn("spf 50+", reply.lower())
        print("   [PASS] Chatbot sunscreen consultation response verified.")

    def test_06_chatbot_ingredient_inquiry(self):
        """Verify chatbot gives accurate active ingredient and contraindication advice."""
        context = synthesize_patient_context(self.sample_m1_scores, self.sample_patient_info)
        bot = DermaVisionConsultationBot(context)
        
        reply = bot.respond("Can I use high-strength chemical peel acids on this spot?")
        self.assertIn("recommended actives", reply.lower())
        self.assertIn("avoid", reply.lower())
        print("   [PASS] Chatbot ingredient guidance & safety warnings verified.")

    def test_07_chatbot_trajectory_inquiry(self):
        """Verify chatbot explains 3-month forecast in user-friendly terms."""
        context = synthesize_patient_context(
            self.sample_m1_scores, self.sample_patient_info,
            module4_trajectory=self.sample_m4_forecast
        )
        bot = DermaVisionConsultationBot(context)
        
        reply = bot.respond("What is my trajectory and how will my skin look in 3 months?")
        self.assertIn("trajectory", reply.lower())
        self.assertIn("3-month", reply.lower())
        print("   [PASS] Chatbot longitudinal trajectory response verified.")

    def test_08_stateless_chat_api(self):
        """Verify functional chat endpoint helper returns structured response."""
        context = synthesize_patient_context(self.sample_m1_scores, self.sample_patient_info)
        res = chat_with_dermavision_bot(context, "Do I need to see a dermatologist?")
        
        self.assertIn('user_query', res)
        self.assertIn('bot_response', res)
        self.assertIn('urgency_level', res)
        self.assertIn('dermatologist', res['bot_response'].lower())
        print("   [PASS] Stateless chat API helper verified.")


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING DERMAVISION MODULE 5 GENERATIVE AI UNIT & INTEGRATION TESTS")
    print("=" * 70)
    unittest.main()
