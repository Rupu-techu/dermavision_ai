"""
DermaVision AI — Module 5: VAE & Similarity Search Unit & Integration Test Suite
================================================================================
Tests:
1. SkinLesionVAE Encoder Output Shapes (mu, logvar)
2. SkinLesionVAE Decoder Output Shape & Range [0, 1]
3. Reparameterization Trick (Training vs. Eval Mode)
4. Beta-VAE Loss Computation (Reconstruction MSE + KL Divergence)
5. Case Similarity Gallery Indexing & Latent Embedding Extraction
6. Nearest-Neighbor Clinical Case Retrieval (Cosine Similarity & Distance)
7. Morphological Anomaly Scoring (Reconstruction Error)
8. Gallery Save and Load Serialization

Author: DermaVision AI Team (Module 5)
"""

import os
import sys
import unittest
import tempfile
import torch

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.module5_vae import (
    SkinLesionVAE,
    vae_loss_function,
    SkinCaseSimilarityEngine
)


class TestModule5VAE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[Module 5 VAE Test Setup] Running VAE test suite on: {cls.device.upper()}")

    def test_01_encoder_forward(self):
        """Verify VAE Encoder produces mu and logvar of shape (B, latent_dim)."""
        latent_dim = 32
        model = SkinLesionVAE(latent_dim=latent_dim).to(self.device)
        model.eval()
        
        batch_size = 4
        dummy_x = torch.rand(batch_size, 3, 64, 64, device=self.device)
        with torch.no_grad():
            mu, logvar = model.encode(dummy_x)
            
        self.assertEqual(mu.shape, (batch_size, latent_dim))
        self.assertEqual(logvar.shape, (batch_size, latent_dim))
        print("   [PASS] VAE Encoder output shapes (mu, logvar) verified.")

    def test_02_decoder_forward(self):
        """Verify VAE Decoder maps latent vector z to 64x64 RGB in [0, 1]."""
        latent_dim = 32
        model = SkinLesionVAE(latent_dim=latent_dim).to(self.device)
        model.eval()
        
        batch_size = 4
        dummy_z = torch.randn(batch_size, latent_dim, device=self.device)
        with torch.no_grad():
            recon = model.decode(dummy_z)
            
        self.assertEqual(recon.shape, (batch_size, 3, 64, 64))
        self.assertTrue(torch.all(recon >= 0.0) and torch.all(recon <= 1.0))
        print("   [PASS] VAE Decoder reconstruction shape and [0, 1] range verified.")

    def test_03_reparameterization_trick(self):
        """Verify sampling behavior in training vs evaluation mode."""
        model = SkinLesionVAE(latent_dim=16).to(self.device)
        mu = torch.zeros(2, 16, device=self.device)
        logvar = torch.zeros(2, 16, device=self.device) # std = 1
        
        # In eval mode: returns deterministic mu
        model.eval()
        z_eval = model.reparameterize(mu, logvar)
        self.assertTrue(torch.equal(z_eval, mu))
        
        # In train mode: adds stochastic noise
        model.train()
        z_train = model.reparameterize(mu, logvar)
        self.assertFalse(torch.equal(z_train, mu))
        print("   [PASS] Reparameterization trick verified (deterministic eval, stochastic train).")

    def test_04_loss_function(self):
        """Verify Beta-VAE loss computation (MSE + KL divergence)."""
        b_size = 4
        dummy_orig = torch.rand(b_size, 3, 64, 64, device=self.device)
        dummy_recon = torch.rand(b_size, 3, 64, 64, device=self.device)
        mu = torch.randn(b_size, 32, device=self.device)
        logvar = torch.randn(b_size, 32, device=self.device)
        
        total_loss, recon_loss, kl_div = vae_loss_function(dummy_recon, dummy_orig, mu, logvar, beta=1.0)
        
        self.assertGreater(float(total_loss.item()), 0.0)
        self.assertGreater(float(recon_loss.item()), 0.0)
        self.assertGreaterEqual(float(kl_div.item()), 0.0)
        print("   [PASS] Beta-VAE loss function verified.")

    def test_05_similarity_gallery_building(self):
        """Verify indexing clinical cases into latent embedding gallery."""
        model = SkinLesionVAE(latent_dim=32).to(self.device)
        engine = SkinCaseSimilarityEngine(model, device=self.device)
        
        num_cases = 10
        dummy_gallery_imgs = torch.rand(num_cases, 3, 64, 64)
        dummy_metadata = [{'case_id': f'C_{i}', 'diagnosis': 'MEL' if i % 2 == 0 else 'NEV'} for i in range(num_cases)]
        
        engine.build_gallery(dummy_gallery_imgs, dummy_metadata)
        
        self.assertIsNotNone(engine.case_embeddings)
        self.assertEqual(engine.case_embeddings.shape, (num_cases, 32))
        self.assertEqual(len(engine.case_metadata), num_cases)
        print("   [PASS] Clinical case gallery indexed successfully in latent space.")

    def test_06_similarity_search_retrieval(self):
        """Verify querying similar cases via Cosine Similarity in latent space."""
        model = SkinLesionVAE(latent_dim=32).to(self.device)
        engine = SkinCaseSimilarityEngine(model, device=self.device)
        
        num_cases = 8
        dummy_gallery_imgs = torch.rand(num_cases, 3, 64, 64)
        dummy_metadata = [
            {'case_id': f'CASE_{i}', 'diagnosis': 'ACK', 'diagnosis_name': 'Actinic Keratosis'}
            for i in range(num_cases)
        ]
        engine.build_gallery(dummy_gallery_imgs, dummy_metadata)
        
        # Test query image
        query_img = torch.rand(3, 64, 64)
        res = engine.find_similar_cases(query_img, top_k=3)
        
        self.assertIn('top_similar_cases', res)
        self.assertIn('query_anomaly_score', res)
        self.assertEqual(len(res['top_similar_cases']), 3)
        
        for case in res['top_similar_cases']:
            self.assertIn('similarity_score', case)
            self.assertIn('latent_distance', case)
            self.assertTrue(0.0 <= case['similarity_score'] <= 100.0)
            
        print("   [PASS] Case similarity search and top-K ranking verified.")

    def test_07_anomaly_scoring(self):
        """Verify reconstruction error anomaly score calculation."""
        model = SkinLesionVAE(latent_dim=32).to(self.device)
        engine = SkinCaseSimilarityEngine(model, device=self.device)
        
        dummy_imgs = torch.rand(4, 3, 64, 64)
        dummy_meta = [{'case_id': f'C_{i}', 'diagnosis': 'BCC'} for i in range(4)]
        engine.build_gallery(dummy_imgs, dummy_meta)
        
        query_img = torch.rand(3, 64, 64)
        res = engine.find_similar_cases(query_img, top_k=2)
        
        self.assertIn('query_anomaly_score', res)
        self.assertIn('anomaly_interpretation', res)
        self.assertGreaterEqual(res['query_anomaly_score'], 0.0)
        print("   [PASS] Lesion anomaly score & reconstruction error verified.")

    def test_08_gallery_serialization(self):
        """Verify saving and reloading reference case gallery."""
        model = SkinLesionVAE(latent_dim=32).to(self.device)
        engine = SkinCaseSimilarityEngine(model, device=self.device)
        
        dummy_imgs = torch.rand(5, 3, 64, 64)
        dummy_meta = [{'case_id': f'C_{i}', 'diagnosis': 'SEK'} for i in range(5)]
        engine.build_gallery(dummy_imgs, dummy_meta)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            gal_path = os.path.join(tmp_dir, "test_gallery.pt")
            engine.save_gallery(gal_path)
            self.assertTrue(os.path.exists(gal_path))
            
            engine2 = SkinCaseSimilarityEngine(model, device=self.device)
            engine2.load_gallery(gal_path)
            self.assertEqual(len(engine2.case_metadata), 5)
            self.assertEqual(engine2.case_embeddings.shape, (5, 32))
            
        print("   [PASS] Gallery serialization (save & load) verified.")


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING DERMAVISION MODULE 5 VAE UNIT & INTEGRATION TESTS")
    print("=" * 70)
    unittest.main()
