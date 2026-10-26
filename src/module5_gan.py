"""
DermaVision AI — Module 5: Generative Adversarial Network (GAN) Architecture
=============================================================================
Implementation of Deep Convolutional GAN (DCGAN) and Conditional DCGAN (cDCGAN)
for synthetic skin lesion and pigmentation image generation.

Curriculum Requirement:
"GAN generates synthetic skin images to increase the training dataset."

Key Components:
1. SkinLesionGenerator: DCGAN Generator (Latent z -> 64x64x3 RGB synthetic lesion)
2. SkinLesionDiscriminator: DCGAN Discriminator (64x64x3 RGB -> Real/Fake probability)
3. ConditionalSkinGenerator: Generates class-specific skin lesions (e.g. MEL, NEV, BCC)
4. ConditionalSkinDiscriminator: Evaluates image authenticity conditioned on diagnostic class
5. weights_init_dcgan: Custom weight initialization (N(0, 0.02))
6. High-level generation & dataset augmentation utilities
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.utils as vutils


def weights_init_dcgan(m):
    """
    Custom weight initialization for DCGAN as per Radford et al.
    Initializes Conv2d, ConvTranspose2d to N(0, 0.02) and BatchNorm to N(1, 0.02).
    Directly aligns with Module 2 weight initialization theory.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class SkinLesionGenerator(nn.Module):
    """
    DCGAN Generator for DermaVision AI.
    Maps latent noise vector z in R^(z_dim) to a 64x64x3 synthetic skin lesion patch.
    Uses progressive ConvTranspose2d layers with BatchNorm and ReLU, ending with Tanh.
    """
    def __init__(self, z_dim=100, feature_g=64, out_channels=3):
        super(SkinLesionGenerator, self).__init__()
        self.z_dim = z_dim
        self.network = nn.Sequential(
            # Input: latent vector z of shape (B, z_dim, 1, 1) -> (B, feature_g*8, 4, 4)
            nn.ConvTranspose2d(z_dim, feature_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            
            # State: (B, feature_g*8, 4, 4) -> (B, feature_g*4, 8, 8)
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            
            # State: (B, feature_g*4, 8, 8) -> (B, feature_g*2, 16, 16)
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            
            # State: (B, feature_g*2, 16, 16) -> (B, feature_g, 32, 32)
            nn.ConvTranspose2d(feature_g * 2, feature_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            
            # State: (B, feature_g, 32, 32) -> (B, out_channels, 64, 64)
            nn.ConvTranspose2d(feature_g, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        if z.ndim == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.network(z)


class SkinLesionDiscriminator(nn.Module):
    """
    DCGAN Discriminator for DermaVision AI.
    Classifies an input 64x64x3 skin image patch as real (1) or synthetic (0).
    Uses progressive Conv2d layers with LeakyReLU(0.2), BatchNorm, and Dropout.
    """
    def __init__(self, in_channels=3, feature_d=64):
        super(SkinLesionDiscriminator, self).__init__()
        self.network = nn.Sequential(
            # Input: (B, in_channels, 64, 64) -> (B, feature_d, 32, 32)
            nn.Conv2d(in_channels, feature_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (B, feature_d, 32, 32) -> (B, feature_d*2, 16, 16)
            nn.Conv2d(feature_d, feature_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # State: (B, feature_d*2, 16, 16) -> (B, feature_d*4, 8, 8)
            nn.Conv2d(feature_d * 2, feature_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # State: (B, feature_d*4, 8, 8) -> (B, feature_d*8, 4, 4)
            nn.Conv2d(feature_d * 4, feature_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (B, feature_d*8, 4, 4) -> (B, 1, 1, 1)
            nn.Conv2d(feature_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.network(x)
        return out.view(-1, 1)


class ConditionalSkinGenerator(nn.Module):
    """
    Conditional DCGAN Generator (cDCGAN).
    Generates synthetic skin lesion images conditioned on clinical disease class
    (e.g., 0: ACK, 1: BCC, 2: MEL, 3: NEV, 4: SCC, 5: SEK).
    Enables targeted synthetic data generation for rare or critical conditions (like Melanoma).
    """
    def __init__(self, num_classes=6, z_dim=100, embed_dim=32, feature_g=64, out_channels=3):
        super(ConditionalSkinGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        
        total_dim = z_dim + embed_dim
        self.network = nn.Sequential(
            # Input: (B, total_dim, 1, 1) -> (B, feature_g*8, 4, 4)
            nn.ConvTranspose2d(total_dim, feature_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            
            # State: (B, feature_g*8, 4, 4) -> (B, feature_g*4, 8, 8)
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            
            # State: (B, feature_g*4, 8, 8) -> (B, feature_g*2, 16, 16)
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            
            # State: (B, feature_g*2, 16, 16) -> (B, feature_g, 32, 32)
            nn.ConvTranspose2d(feature_g * 2, feature_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            
            # State: (B, feature_g, 32, 32) -> (B, out_channels, 64, 64)
            nn.ConvTranspose2d(feature_g, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        if z.ndim == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        c = self.class_embed(labels).unsqueeze(-1).unsqueeze(-1)
        combined = torch.cat([z, c], dim=1)
        return self.network(combined)


class ConditionalSkinDiscriminator(nn.Module):
    """
    Conditional DCGAN Discriminator (cDCGAN).
    Evaluates authenticity of 64x64x3 skin image conditioned on class label.
    Embeds class label into a 64x64 spatial feature plane concatenated with the image.
    """
    def __init__(self, num_classes=6, in_channels=3, embed_dim=1, feature_d=64):
        super(ConditionalSkinDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, 64 * 64)
        
        total_in = in_channels + embed_dim
        self.network = nn.Sequential(
            # Input: (B, 4, 64, 64) -> (B, feature_d, 32, 32)
            nn.Conv2d(total_in, feature_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, feature_d, 32, 32) -> (B, feature_d*2, 16, 16)
            nn.Conv2d(feature_d, feature_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # (B, feature_d*2, 16, 16) -> (B, feature_d*4, 8, 8)
            nn.Conv2d(feature_d * 2, feature_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # (B, feature_d*4, 8, 8) -> (B, feature_d*8, 4, 4)
            nn.Conv2d(feature_d * 4, feature_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, feature_d*8, 4, 4) -> (B, 1, 1, 1)
            nn.Conv2d(feature_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        batch_size = x.size(0)
        c = self.class_embed(labels).view(batch_size, 1, 64, 64)
        combined = torch.cat([x, c], dim=1)
        out = self.network(combined)
        return out.view(-1, 1)


# =====================================================================
# High-Level Inference, Sampling, and Dataset Augmentation Utilities
# =====================================================================

def generate_synthetic_skin_samples(generator, num_samples=16, z_dim=100, class_idx=None, device='cpu'):
    """
    Generates synthetic skin lesion images using a trained generator.
    
    Args:
        generator: Trained SkinLesionGenerator or ConditionalSkinGenerator
        num_samples: Number of images to synthesize
        z_dim: Latent dimension size
        class_idx: Optional class ID (int or tensor) for conditional generation
        device: 'cpu', 'cuda', or 'mps'
        
    Returns:
        torch.Tensor of shape (num_samples, 3, 64, 64) in normalized [0, 1] range.
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, 1, 1, device=device)
        if isinstance(generator, ConditionalSkinGenerator):
            if class_idx is None:
                labels = torch.randint(0, generator.num_classes, (num_samples,), device=device)
            elif isinstance(class_idx, int):
                labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
            else:
                labels = class_idx.to(device)
            fake_imgs = generator(z, labels)
        else:
            fake_imgs = generator(z)
            
        # Denormalize from [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs + 1.0) / 2.0
        fake_imgs = torch.clamp(fake_imgs, 0.0, 1.0)
        
    return fake_imgs


def save_synthetic_image_grid(images_tensor, output_path, nrow=4):
    """
    Exports a PyTorch image tensor (B, 3, 64, 64) as a high-resolution grid image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    grid = vutils.make_grid(images_tensor.cpu(), nrow=nrow, padding=2, normalize=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(output_path)
    return output_path


def augment_dataset_with_gan(generator, output_dir, num_samples_per_class=50, classes=None, device='cpu'):
    """
    Fulfills the core curriculum mapping:
    "GAN generates synthetic skin images to increase the training dataset."
    
    Generates synthetic skin images and organizes them into structured class directories
    for seamless integration into PyTorch ImageFolder or computer vision pipelines.
    
    Returns:
        dict: Mapping of class_name -> list of generated file paths
    """
    if classes is None:
        classes = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
        
    os.makedirs(output_dir, exist_ok=True)
    manifest = {}
    
    for c_idx, c_name in enumerate(classes):
        class_folder = os.path.join(output_dir, c_name)
        os.makedirs(class_folder, exist_ok=True)
        manifest[c_name] = []
        
        # Generate batch for this class
        if isinstance(generator, ConditionalSkinGenerator):
            samples = generate_synthetic_skin_samples(generator, num_samples=num_samples_per_class, class_idx=c_idx, device=device)
        else:
            samples = generate_synthetic_skin_samples(generator, num_samples=num_samples_per_class, device=device)
            
        for i in range(num_samples_per_class):
            img_tensor = samples[i]
            img_np = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            img_pil = Image.fromarray(img_np)
            
            img_filename = f"gan_synthetic_{c_name}_{i+1:04d}.png"
            img_path = os.path.join(class_folder, img_filename)
            img_pil.save(img_path)
            manifest[c_name].append(img_path)
            
    return manifest
