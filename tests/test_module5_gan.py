"""
DermaVision AI — Module 5: GAN Unit & Integration Test Suite
============================================================
Tests:
1. SkinLesionGenerator Forward Pass & Tensor Shapes
2. SkinLesionDiscriminator Forward Pass & Probability Scores
3. ConditionalSkinGenerator Forward Pass & Class Conditioning
4. ConditionalSkinDiscriminator Forward Pass
5. DCGAN Weight Initialization Function
6. Synthetic Image Sampling & Value Range [0, 1]
7. Grid Image Export Utility
8. GAN Dataset Augmentation Engine (Class Directory Structure)

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

from src.module5_gan import (
    SkinLesionGenerator,
    SkinLesionDiscriminator,
    ConditionalSkinGenerator,
    ConditionalSkinDiscriminator,
    weights_init_dcgan,
    generate_synthetic_skin_samples,
    save_synthetic_image_grid,
    augment_dataset_with_gan
)


class TestModule5GAN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[Module 5 Test Setup] Running GAN test suite on: {cls.device.upper()}")

    def test_01_generator_forward(self):
        """Verify DCGAN Generator outputs (B, 3, 64, 64) with Tanh bounded in [-1, 1]."""
        netG = SkinLesionGenerator(z_dim=100, feature_g=32).to(self.device)
        netG.eval()
        
        batch_size = 4
        z = torch.randn(batch_size, 100, device=self.device)
        with torch.no_grad():
            fake_imgs = netG(z)
            
        self.assertEqual(fake_imgs.shape, (batch_size, 3, 64, 64))
        self.assertTrue(torch.all(fake_imgs >= -1.0) and torch.all(fake_imgs <= 1.0))
        print("   [PASS] SkinLesionGenerator output shape and [-1, 1] bounds verified.")

    def test_02_discriminator_forward(self):
        """Verify DCGAN Discriminator outputs (B, 1) with Sigmoid probabilities in [0, 1]."""
        netD = SkinLesionDiscriminator(in_channels=3, feature_d=32).to(self.device)
        netD.eval()
        
        batch_size = 4
        dummy_imgs = torch.randn(batch_size, 3, 64, 64, device=self.device)
        with torch.no_grad():
            scores = netD(dummy_imgs)
            
        self.assertEqual(scores.shape, (batch_size, 1))
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))
        print("   [PASS] SkinLesionDiscriminator output shape and [0, 1] probabilities verified.")

    def test_03_conditional_generator(self):
        """Verify Conditional cDCGAN Generator conditioned on diagnostic classes."""
        netG = ConditionalSkinGenerator(num_classes=6, z_dim=100, embed_dim=16, feature_g=32).to(self.device)
        netG.eval()
        
        batch_size = 6
        z = torch.randn(batch_size, 100, device=self.device)
        labels = torch.arange(0, 6, device=self.device) # ACK, BCC, MEL, NEV, SCC, SEK
        
        with torch.no_grad():
            fake_imgs = netG(z, labels)
            
        self.assertEqual(fake_imgs.shape, (batch_size, 3, 64, 64))
        self.assertTrue(torch.all(fake_imgs >= -1.0) and torch.all(fake_imgs <= 1.0))
        print("   [PASS] ConditionalSkinGenerator class conditioning verified.")

    def test_04_conditional_discriminator(self):
        """Verify Conditional cDCGAN Discriminator forward pass."""
        netD = ConditionalSkinDiscriminator(num_classes=6, in_channels=3, feature_d=32).to(self.device)
        netD.eval()
        
        batch_size = 6
        dummy_imgs = torch.randn(batch_size, 3, 64, 64, device=self.device)
        labels = torch.arange(0, 6, device=self.device)
        
        with torch.no_grad():
            scores = netD(dummy_imgs, labels)
            
        self.assertEqual(scores.shape, (batch_size, 1))
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))
        print("   [PASS] ConditionalSkinDiscriminator forward pass verified.")

    def test_05_weight_initialization(self):
        """Verify DCGAN custom normal weight initialization."""
        netG = SkinLesionGenerator(z_dim=100, feature_g=32)
        netG.apply(weights_init_dcgan)
        
        # Check first conv layer weight std is roughly 0.02
        first_layer = netG.network[0]
        std = first_layer.weight.data.std().item()
        self.assertAlmostEqual(std, 0.02, delta=0.01)
        print("   [PASS] DCGAN custom weight initialization verified.")

    def test_06_synthetic_sampling(self):
        """Verify generate_synthetic_skin_samples returns [0, 1] image tensor."""
        netG = SkinLesionGenerator(z_dim=100, feature_g=32).to(self.device)
        samples = generate_synthetic_skin_samples(netG, num_samples=8, z_dim=100, device=self.device)
        
        self.assertEqual(samples.shape, (8, 3, 64, 64))
        self.assertTrue(torch.all(samples >= 0.0) and torch.all(samples <= 1.0))
        print("   [PASS] Synthetic skin sampling and [0, 1] normalization verified.")

    def test_07_grid_export(self):
        """Verify saving synthetic skin grid image file."""
        netG = SkinLesionGenerator(z_dim=100, feature_g=32).to(self.device)
        samples = generate_synthetic_skin_samples(netG, num_samples=4, z_dim=100, device=self.device)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_grid.png")
            save_synthetic_image_grid(samples, out_file, nrow=2)
            self.assertTrue(os.path.exists(out_file))
            self.assertGreater(os.path.getsize(out_file), 100)
            
        print("   [PASS] Synthetic image grid export verified.")

    def test_08_dataset_augmentation_pipeline(self):
        """Verify automated dataset augmentation for class imbalance."""
        netG = ConditionalSkinGenerator(num_classes=3, z_dim=100, embed_dim=16, feature_g=32).to(self.device)
        classes = ["MEL", "NEV", "BCC"]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = augment_dataset_with_gan(
                netG, output_dir=tmp_dir, num_samples_per_class=3, classes=classes, device=self.device
            )
            for c in classes:
                self.assertIn(c, manifest)
                self.assertEqual(len(manifest[c]), 3)
                for fpath in manifest[c]:
                    self.assertTrue(os.path.exists(fpath))
                    
        print("   [PASS] GAN Dataset Augmentation Engine successfully verified.")


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING DERMAVISION MODULE 5 GAN UNIT & INTEGRATION TESTS")
    print("=" * 70)
    unittest.main()
