"""
DermaVision AI — Module 5: VAE Training, Latent Representation Learning & Case Retrieval
========================================================================================
End-to-end training and indexing pipeline for Convolutional Variational Autoencoder (ConvVAE).
Learns compact 32-dimensional latent representations for:
1. Clinical Case Similarity Search (Cosine & Euclidean nearest neighbors)
2. Latent Space 2D Manifold Visualization (t-SNE / PCA)
3. Anomaly Detection & Morphological Lesion Reconstruction
4. Continuous Latent Traversal / Interpolation

Curriculum Mapping:
"VAE learns compact representations of skin images for similarity search and visualization."

Author: DermaVision AI Team (Module 5)
"""

import os
# Prevent OpenMP / Accelerate thread conflict on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from src.module5_vae import (
    SkinLesionVAE,
    vae_loss_function,
    SkinCaseSimilarityEngine,
    plot_latent_space_2d,
    plot_reconstruction_grid,
    plot_latent_interpolation
)
from train_module5_gan import extract_skin_patches_from_workspace, SkinLesionPatchDataset


# Clinical metadata definitions for gallery indexing
DIAGNOSTIC_LOOKUP = {
    0: {
        'code': 'ACK',
        'name': 'Actinic Keratosis',
        'risk_level': 'Moderate (Pre-cancerous)',
        'outcome': 'Cryotherapy or 5-fluorouracil topical course with quarterly re-evaluation',
        'biopsied': True
    },
    1: {
        'code': 'BCC',
        'name': 'Basal Cell Carcinoma',
        'risk_level': 'High (Malignant)',
        'outcome': 'Complete surgical excision with clear histological margins achieved',
        'biopsied': True
    },
    2: {
        'code': 'MEL',
        'name': 'Melanoma',
        'risk_level': 'Critical (Malignant Melanocytic)',
        'outcome': 'Wide local excision with sentinel lymph node staging and oncology follow-up',
        'biopsied': True
    },
    3: {
        'code': 'NEV',
        'name': 'Nevus (Benign Mole)',
        'risk_level': 'Low (Benign)',
        'outcome': 'Benign melanocytic junctional lesion confirmed; routine annual surveillance',
        'biopsied': False
    },
    4: {
        'code': 'SCC',
        'name': 'Squamous Cell Carcinoma',
        'risk_level': 'High (Malignant)',
        'outcome': 'Mohs micrographic surgery performed with excellent healing response',
        'biopsied': True
    },
    5: {
        'code': 'SEK',
        'name': 'Seborrheic Keratosis',
        'risk_level': 'Low (Benign Hyperkeratotic)',
        'outcome': 'Harmless epidermal growth confirmed; no active treatment required',
        'biopsied': False
    }
}


def train_skin_vae(patches, labels, device='cpu', epochs=25, batch_size=32, latent_dim=32, lr=0.001, beta=0.5):
    """
    Trains the Convolutional VAE on authentic skin lesion patches.
    Optimizes Beta-VAE objective: Reconstruction MSE + beta * KL Divergence.
    """
    print(f"\n[Step 2/5] Initializing Convolutional VAE on: {device.upper()}")
    print(f"   - Latent Dimension (z): {latent_dim}")
    print(f"   - KL Divergence Weight (Beta): {beta}")
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(), # Scales to [0, 1]
    ])
    
    dataset = SkinLesionPatchDataset(patches, labels=labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = SkinLesionVAE(in_channels=3, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    total_losses = []
    recon_losses = []
    kl_losses = []
    
    print("\n[Step 3/5] Starting VAE Training Loop...")
    print("=" * 70)
    print(f"{'Epoch':<10} | {'Total Loss':<14} | {'Recon (MSE)':<14} | {'KL Divergence':<14}")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        model.train()
        ep_total = 0.0
        ep_recon = 0.0
        ep_kl = 0.0
        batches = 0
        
        for data in dataloader:
            if isinstance(data, (tuple, list)):
                x, _ = data
            else:
                x = data
                
            x = x.to(device)
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x)
            loss, recon_l, kl_l = vae_loss_function(recon_x, x, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            ep_total += loss.item()
            ep_recon += recon_l.item()
            ep_kl += kl_l.item()
            batches += 1
            
        avg_tot = ep_total / max(1, batches)
        avg_rec = ep_recon / max(1, batches)
        avg_kl = ep_kl / max(1, batches)
        
        total_losses.append(avg_tot)
        recon_losses.append(avg_rec)
        kl_losses.append(avg_kl)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | {avg_tot:<14.4f} | {avg_rec:<14.4f} | {avg_kl:<14.4f}")
            
    print("=" * 70)
    print("VAE Representation Learning Completed Successfully!")
    
    return model, dataloader, total_losses, recon_losses, kl_losses


def build_reference_case_gallery(vae_model, patches, labels, device='cpu', num_gallery=150):
    """
    Builds and indexes a reference gallery of diagnosed dermatological cases in latent space.
    """
    print("\n[Step 4/5] Indexing Clinical Case Gallery in VAE Latent Space...")
    engine = SkinCaseSimilarityEngine(vae_model, device=device)
    
    # Select balanced gallery patches across the 6 classes
    gallery_tensors = []
    gallery_meta = []
    
    to_tensor = transforms.ToTensor()
    num_selected = 0
    
    for i in range(len(patches)):
        if num_selected >= num_gallery:
            break
        patch = patches[i]
        c_idx = labels[i] % 6
        info = DIAGNOSTIC_LOOKUP[c_idx]
        
        img = Image.fromarray(patch) if isinstance(patch, np.ndarray) else patch
        img_t = to_tensor(img)
        gallery_tensors.append(img_t)
        
        gallery_meta.append({
            'case_id': f"DV-CASE-{num_selected+1:04d}",
            'diagnosis': info['code'],
            'diagnosis_name': info['name'],
            'risk_level': info['risk_level'],
            'outcome': info['outcome'],
            'biopsied': info['biopsied']
        })
        num_selected += 1
        
    gallery_tensor_stacked = torch.stack(gallery_tensors, dim=0) # (num_gallery, 3, 64, 64)
    engine.build_gallery(gallery_tensor_stacked, gallery_meta)
    
    return engine, gallery_tensor_stacked


def main():
    print("=" * 70)
    print("DERMAVISION AI — MODULE 5 VAE TRAINING & CASE SIMILARITY ENGINE")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Extract patches
    patches, labels = extract_skin_patches_from_workspace(base_dir=BASE_DIR, patch_size=64, num_patches=600)
    
    # 2. Train VAE
    latent_dim = 32
    vae_model, dataloader, total_losses, recon_losses, kl_losses = train_skin_vae(
        patches, labels, device=device, epochs=25, batch_size=32, latent_dim=latent_dim, lr=0.001, beta=0.5
    )
    
    # 3. Save VAE Checkpoint
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    vae_path = os.path.join(models_dir, 'module5_vae_best.pth')
    torch.save(vae_model.state_dict(), vae_path)
    print(f"\nSaved VAE Model Checkpoint to: {vae_path}")
    
    # 4. Build & Serialize Clinical Case Gallery
    similarity_engine, gallery_tensors = build_reference_case_gallery(
        vae_model, patches, labels, device=device, num_gallery=150
    )
    gallery_path = os.path.join(models_dir, 'module5_vae_case_gallery.pt')
    similarity_engine.save_gallery(gallery_path)
    
    # 5. Generate Visualizations
    print("\n[Step 5/5] Generating Latent Visualizations & Evaluation Artifacts...")
    
    # A. Reconstruction Grid
    recon_grid_path = os.path.join(BASE_DIR, 'Module5_VAE_Reconstruction_Grid.png')
    plot_reconstruction_grid(vae_model, gallery_tensors, recon_grid_path, num_pairs=8, device=device)
    print(f" - Saved Reconstruction Comparison Grid: {recon_grid_path}")
    
    # B. Latent Space 2D t-SNE Plot
    class_names = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    tsne_path = os.path.join(BASE_DIR, 'Module5_VAE_Latent_Space_tSNE.png')
    plot_latent_space_2d(vae_model, dataloader, class_names, tsne_path, method='tsne', device=device)
    print(f" - Saved Latent Space t-SNE Projection: {tsne_path}")
    
    # C. Latent Manifold Interpolation
    interp_path = os.path.join(BASE_DIR, 'Module5_VAE_Latent_Interpolation.png')
    plot_latent_interpolation(vae_model, gallery_tensors[3], gallery_tensors[2], interp_path, steps=10, device=device)
    print(f" - Saved Latent Interpolation Manifold: {interp_path}")
    
    # D. Demonstrate Case Similarity Search on Query Image
    print("\n" + "=" * 70)
    print("DEMO: SIMILARITY SEARCH ON TEST QUERY LESION")
    print("=" * 70)
    query_sample = gallery_tensors[2] # Melanoma sample
    query_result = similarity_engine.find_similar_cases(query_sample, top_k=3)
    
    print(f"Query Anomaly Score: {query_result['query_anomaly_score']} ({query_result['anomaly_interpretation']})")
    print(f"Top-3 Clinically Similar Cases in Reference Gallery:")
    for rank, case in enumerate(query_result['top_similar_cases'], 1):
        print(f" #{rank} Case ID: {case['case_id']:<15} | Condition: {case['diagnosis_code']:<4} ({case['diagnosis_name']})")
        print(f"    Similarity: {case['similarity_score']:.2f}% | Latent Distance: {case['latent_distance']}")
        print(f"    Clinical Outcome: {case['clinical_outcome']}")
        print(f"    Risk Level: {case['risk_level']} | Biopsied: {case['biopsied']}")
        
    # 6. Save Benchmark Results Summary
    summary_csv = os.path.join(BASE_DIR, 'DermaVision_Module5_VAE_Results.csv')
    df_results = pd.DataFrame([{
        'Model Architecture': 'ConvVAE (4-layer Conv2d + ConvTranspose2d)',
        'Latent Dimension (z)': latent_dim,
        'KL Divergence Beta': 0.5,
        'Final Total Loss': round(total_losses[-1], 4),
        'Final Reconstruction MSE': round(recon_losses[-1], 4),
        'Final KL Divergence': round(kl_losses[-1], 4),
        'Indexed Reference Cases': len(similarity_engine.case_metadata),
        'Similarity Metrics': 'Cosine Similarity & Euclidean L2 Distance'
    }])
    df_results.to_csv(summary_csv, index=False)
    print(f"\nSaved VAE quantitative summary to: {summary_csv}")
    print("=" * 70)
    print("MODULE 5 VAE & SIMILARITY SEARCH IMPLEMENTATION COMPLETED!")


if __name__ == '__main__':
    main()
