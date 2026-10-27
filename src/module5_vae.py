"""
DermaVision AI — Module 5: Variational Autoencoder (VAE) & Case Similarity Search
================================================================================
Implementation of Convolutional Variational Autoencoder (ConvVAE) for:
1. Learning compact continuous latent representations (z in R^latent_dim) of skin lesions.
2. Clinical Case Similarity Search: Finding top-K visually and structurally similar diagnosed cases.
3. Latent space 2D manifold projection & visualization (t-SNE / PCA).
4. Morphological latent traversal and reconstruction error (anomaly detection).

Curriculum Requirement:
"VAE learns compact representations of skin images for similarity search and visualization."

Author: DermaVision AI Team (Module 5)
"""

import os
# Prevent OpenMP / Accelerate thread conflict on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class SkinLesionVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for Skin Lesion Representation Learning.
    
    Architecture:
      Encoder: 64x64x3 -> (Conv2d + BatchNorm + LeakyReLU) x 4 -> Dense -> (mu, logvar)
      Reparameterization: z = mu + eps * exp(0.5 * logvar)
      Decoder: z -> Dense -> (ConvTranspose2d + BatchNorm + ReLU) x 3 -> ConvTranspose2d -> Sigmoid
      Output: 64x64x3 reconstructed lesion image in [0, 1]
    """
    def __init__(self, in_channels=3, latent_dim=32, hidden_dims=None):
        super(SkinLesionVAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
            
        # -----------------------------
        # 1. Convolutional Encoder
        # -----------------------------
        modules = []
        c_in = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(c_in, h_dim, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            c_in = h_dim
            
        self.encoder_conv = nn.Sequential(*modules)
        
        # 64x64 halved 4 times = 4x4 spatial dimension with 256 channels -> 4096
        self.flatten_dim = hidden_dims[-1] * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # -----------------------------
        # 2. Convolutional Decoder
        # -----------------------------
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        hidden_dims.reverse() # [256, 128, 64, 32]
        dec_modules = []
        for i in range(len(hidden_dims) - 1):
            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )
            
        self.decoder_conv = nn.Sequential(*dec_modules)
        
        # Final layer: 32 -> 3 (64x64) with Sigmoid for [0, 1] RGB output
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Maps input skin image to Gaussian distribution parameters (mu, logvar)."""
        h = self.encoder_conv(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z from N(mu, var) with differentiable backprop."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        """Maps latent representation z back to reconstructed 64x64 RGB skin lesion image."""
        h = self.decoder_input(z)
        h = h.view(-1, 256, 4, 4)
        h = self.decoder_conv(h)
        out = self.final_layer(h)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes Beta-VAE Loss = Reconstruction Loss (BCE / MSE) + beta * KL Divergence.
    
    Args:
        recon_x: Reconstructed images (B, 3, 64, 64)
        x: Ground truth images (B, 3, 64, 64) in [0, 1]
        mu: Latent mean (B, latent_dim)
        logvar: Latent log-variance (B, latent_dim)
        beta: Weighting factor for KL divergence (Beta-VAE)
    """
    # Reconstruction loss (MSE scaled by image size)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div


# =====================================================================
# 2. Clinical Case Similarity Search Engine
# =====================================================================

class SkinCaseSimilarityEngine:
    """
    Case-Based Reasoning (CBR) and Similarity Retrieval Engine.
    Indexes clinical skin lesion images into compact VAE latent space (z in R^32).
    Given a query patient image, computes Cosine Similarity and Euclidean distance
    to retrieve top-K most similar diagnosed reference cases.
    """
    def __init__(self, vae_model, device='cpu'):
        self.vae = vae_model
        self.device = device
        self.vae.eval()
        
        self.case_embeddings = None # (N, latent_dim)
        self.case_metadata = []     # List of metadata dicts

    def build_gallery(self, image_tensors, metadata_list):
        """
        Populates reference case gallery by computing latent embeddings.
        
        Args:
            image_tensors: Tensor of shape (N, 3, 64, 64) in [0, 1]
            metadata_list: List of N dicts (e.g. {'id': ..., 'diagnosis': ..., 'severity': ...})
        """
        self.vae.eval()
        embeddings = []
        batch_size = 32
        num_cases = len(image_tensors)
        
        with torch.no_grad():
            for i in range(0, num_cases, batch_size):
                batch = image_tensors[i:i + batch_size].to(self.device)
                mu, _ = self.vae.encode(batch)
                embeddings.append(mu.cpu())
                
        self.case_embeddings = torch.cat(embeddings, dim=0) # (N, latent_dim)
        self.case_metadata = metadata_list
        print(f"[Similarity Engine] Indexed {len(self.case_metadata)} clinical cases in latent space.")

    def find_similar_cases(self, query_img_tensor, top_k=3):
        """
        Given a query skin lesion image tensor (1, 3, 64, 64), returns top-K closest historical cases.
        
        Returns:
            list of dicts containing similarity score, diagnosis, and clinical recommendations.
        """
        if self.case_embeddings is None or len(self.case_metadata) == 0:
            raise ValueError("Gallery is empty. Call build_gallery() first.")
            
        self.vae.eval()
        with torch.no_grad():
            if query_img_tensor.ndim == 3:
                query_img_tensor = query_img_tensor.unsqueeze(0)
            query_img_tensor = query_img_tensor.to(self.device)
            
            # Encode query to latent vector mu (deterministic representation)
            query_mu, query_logvar = self.vae.encode(query_img_tensor)
            recon_query = self.vae.decode(query_mu)
            
            # Reconstruction error as anomaly score
            anomaly_error = F.mse_loss(recon_query, query_img_tensor).item()
            
            query_vec = query_mu.cpu() # (1, latent_dim)
            
            # Compute Cosine Similarities across indexed gallery
            norm_query = F.normalize(query_vec, p=2, dim=1)
            norm_gallery = F.normalize(self.case_embeddings, p=2, dim=1)
            cos_sims = torch.mm(norm_query, norm_gallery.t()).squeeze(0) # (N,)
            
            # Compute Euclidean distances
            euc_dists = torch.norm(self.case_embeddings - query_vec, p=2, dim=1) # (N,)
            
        # Get top-K matches based on Cosine Similarity
        top_indices = torch.topk(cos_sims, k=min(top_k, len(self.case_metadata))).indices.tolist()
        
        results = []
        for idx in top_indices:
            meta = self.case_metadata[idx].copy()
            sim_score = round(float(cos_sims[idx].item()) * 100, 2)
            dist_val = round(float(euc_dists[idx].item()), 4)
            
            results.append({
                'case_id': meta.get('case_id', f'CASE_{idx+1:04d}'),
                'diagnosis_code': meta.get('diagnosis', 'UNK'),
                'diagnosis_name': meta.get('diagnosis_name', 'Skin Condition'),
                'similarity_score': max(0.0, sim_score),
                'latent_distance': dist_val,
                'clinical_outcome': meta.get('outcome', 'Standard dermatological treatment response'),
                'biopsied': meta.get('biopsied', False),
                'risk_level': meta.get('risk_level', 'Moderate')
            })
            
        return {
            'top_similar_cases': results,
            'query_anomaly_score': round(anomaly_error * 100, 3),
            'anomaly_interpretation': 'High Morphological Atypicality' if anomaly_error > 0.05 else 'Typical Lesion Morphology'
        }

    def save_gallery(self, save_path):
        """Serializes gallery embeddings and metadata."""
        torch.save({
            'embeddings': self.case_embeddings,
            'metadata': self.case_metadata
        }, save_path)
        print(f"[Similarity Engine] Saved gallery to: {save_path}")

    def load_gallery(self, load_path):
        """Loads serialized gallery embeddings and metadata."""
        data = torch.load(load_path, map_location='cpu')
        self.case_embeddings = data['embeddings']
        self.case_metadata = data['metadata']
        print(f"[Similarity Engine] Loaded gallery with {len(self.case_metadata)} cases from: {load_path}")


# =====================================================================
# 3. Latent Space Visualization & Manifold Analysis Utilities
# =====================================================================

def plot_latent_space_2d(vae_model, dataloader, class_names, output_path, method='tsne', device='cpu'):
    """
    Extracts latent vectors z for evaluation samples and creates a 2D scatter plot
    colored by skin condition class using t-SNE or PCA.
    """
    vae_model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            mu, _ = vae_model.encode(batch_x)
            latents.append(mu.cpu().numpy())
            labels.append(batch_y.numpy())
            if len(np.concatenate(labels)) >= 300:
                break
                
    X = np.concatenate(latents, axis=0)
    y = np.concatenate(labels, axis=0)
    
    try:
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(X)//5)), n_jobs=1)
            X_2d = reducer.fit_transform(X)
            title = "DermaVision AI — VAE Latent Space Clustering (t-SNE 2D Projection)"
        else:
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X)
            title = "DermaVision AI — VAE Latent Space Clustering (PCA 2D Projection)"
    except Exception:
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        title = "DermaVision AI — VAE Latent Space Clustering (PCA 2D Projection)"
        
    plt.figure(figsize=(10, 8))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
    
    for c_idx, c_name in enumerate(class_names):
        mask = (y == c_idx)
        if np.any(mask):
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                label=c_name,
                c=colors[c_idx % len(colors)],
                alpha=0.8,
                edgecolors='none',
                s=45
            )
            
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Latent Component 1", fontsize=12)
    plt.ylabel("Latent Component 2", fontsize=12)
    plt.legend(title="Condition", fontsize=11, title_fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_reconstruction_grid(vae_model, test_images, output_path, num_pairs=8, device='cpu'):
    """
    Creates a side-by-side comparison of original vs. VAE-reconstructed skin lesions.
    """
    vae_model.eval()
    test_batch = test_images[:num_pairs].to(device)
    
    with torch.no_grad():
        recon_batch, _, _ = vae_model(test_batch)
        
    orig_np = test_batch.cpu().permute(0, 2, 3, 1).numpy()
    recon_np = recon_batch.cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(2, num_pairs, figsize=(2 * num_pairs, 4.5))
    fig.suptitle("DermaVision AI — VAE Skin Lesion Reconstruction (Top: Original | Bottom: Reconstructed)", fontsize=13, fontweight='bold')
    
    for i in range(num_pairs):
        # Original
        axes[0, i].imshow(np.clip(orig_np[i], 0.0, 1.0))
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Orig #{i+1}", fontsize=9)
        
        # Reconstructed
        axes[1, i].imshow(np.clip(recon_np[i], 0.0, 1.0))
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Recon #{i+1}", fontsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_latent_interpolation(vae_model, img1, img2, output_path, steps=10, device='cpu'):
    """
    Performs continuous linear interpolation in latent space between two skin lesions (e.g. Nevus to Melanoma).
    z(alpha) = (1 - alpha) * z1 + alpha * z2
    """
    vae_model.eval()
    x1 = img1.unsqueeze(0).to(device)
    x2 = img2.unsqueeze(0).to(device)
    
    with torch.no_grad():
        mu1, _ = vae_model.encode(x1)
        mu2, _ = vae_model.encode(x2)
        
        alphas = np.linspace(0.0, 1.0, steps)
        interpolated_imgs = []
        for alpha in alphas:
            z_interp = (1.0 - alpha) * mu1 + alpha * mu2
            recon = vae_model.decode(z_interp)
            interpolated_imgs.append(recon.squeeze(0).cpu().permute(1, 2, 0).numpy())
            
    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2.5))
    fig.suptitle("DermaVision AI — Continuous Latent Manifold Traversal (Lesion Interpolation)", fontsize=13, fontweight='bold', y=1.02)
    
    for i in range(steps):
        axes[i].imshow(np.clip(interpolated_imgs[i], 0.0, 1.0))
        axes[i].axis('off')
        axes[i].set_title(f"{int(alphas[i]*100)}%", fontsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
