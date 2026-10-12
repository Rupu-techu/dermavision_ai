"""
DermaVision AI — Module 4: Standalone Training & Evaluation Script
===================================================================
Trains and evaluates:
1. SkinProgressionLSTM (Multi-horizon 1m, 3m, 6m progression forecaster)
2. SkinProgressionGRU (Recurrent architecture ablation)
3. ClinicalJournalBiLSTM (Dermatologist notes and patient journal NLP classifier)

Produces:
- Model checkpoints in models/
- Experiment results CSV: DermaVision_Module4_Experiment_Results.csv
- Visual performance curves and evaluation metrics

Author: Mouli Mukherjee (Module 4)
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.module4_models import SkinProgressionLSTM, SkinProgressionGRU, ClinicalJournalBiLSTM
from src.module4_sequence_dataset import (
    LongitudinalSkinDataset,
    build_longitudinal_dataset,
    ClinicalJournalDataset,
    build_vocab,
    generate_augmented_clinical_corpus,
    CLINICAL_JOURNAL_CORPUS
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_progression_model(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu'):
    """
    Multi-task training loop: MSE on continuous horizons (1m, 3m, 6m) + CrossEntropy on trend.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            seqs = batch['sequence'].to(device)
            t_1m = batch['target_1m'].to(device)
            t_3m = batch['target_3m'].to(device)
            t_6m = batch['target_6m'].to(device)
            t_trend = batch['trend_label'].to(device)
            
            optimizer.zero_grad()
            out = model(seqs)
            
            # Continuous normalized regression loss (targets divided by 100 for balance)
            loss_1m = criterion_reg(out['forecast_1m'], t_1m)
            loss_3m = criterion_reg(out['forecast_3m'], t_3m)
            loss_6m = criterion_reg(out['forecast_6m'], t_6m)
            loss_reg = (loss_1m + loss_3m + loss_6m) / 3.0
            
            # Trend classification loss
            loss_cls = criterion_cls(out['trend_logits'], t_trend)
            
            # Multi-task objective
            loss = loss_reg * 0.05 + loss_cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seqs = batch['sequence'].to(device)
                t_1m = batch['target_1m'].to(device)
                t_3m = batch['target_3m'].to(device)
                t_6m = batch['target_6m'].to(device)
                t_trend = batch['trend_label'].to(device)
                
                out = model(seqs)
                l_1m = criterion_reg(out['forecast_1m'], t_1m)
                l_3m = criterion_reg(out['forecast_3m'], t_3m)
                l_6m = criterion_reg(out['forecast_6m'], t_6m)
                l_reg = (l_1m + l_3m + l_6m) / 3.0
                l_cls = criterion_cls(out['trend_logits'], t_trend)
                
                b_loss = l_reg * 0.05 + l_cls
                val_loss += b_loss.item()
                
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict().copy()
            
    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, train_losses, val_losses

def evaluate_progression_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds_1m, all_true_1m = [], []
    all_preds_3m, all_true_3m = [], []
    all_preds_6m, all_true_6m = [], []
    all_trend_preds, all_trend_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            seqs = batch['sequence'].to(device)
            out = model(seqs)
            
            all_preds_1m.append(out['forecast_1m'].cpu().numpy())
            all_true_1m.append(batch['target_1m'].numpy())
            
            all_preds_3m.append(out['forecast_3m'].cpu().numpy())
            all_true_3m.append(batch['target_3m'].numpy())
            
            all_preds_6m.append(out['forecast_6m'].cpu().numpy())
            all_true_6m.append(batch['target_6m'].numpy())
            
            trend_p = torch.softmax(out['trend_logits'], dim=1).argmax(dim=1).cpu().numpy()
            all_trend_preds.extend(trend_p)
            all_trend_true.extend(batch['trend_label'].numpy())
            
    p1 = np.vstack(all_preds_1m)
    t1 = np.vstack(all_true_1m)
    p3 = np.vstack(all_preds_3m)
    t3 = np.vstack(all_true_3m)
    p6 = np.vstack(all_preds_6m)
    t6 = np.vstack(all_true_6m)
    
    # Calculate regression metrics across all horizons
    mae_1m = mean_absolute_error(t1, p1)
    mae_3m = mean_absolute_error(t3, p3)
    mae_6m = mean_absolute_error(t6, p6)
    overall_mae = (mae_1m + mae_3m + mae_6m) / 3.0
    
    r2_1m = r2_score(t1, p1)
    r2_3m = r2_score(t3, p3)
    r2_6m = r2_score(t6, p6)
    overall_r2 = (r2_1m + r2_3m + r2_6m) / 3.0
    
    trend_acc = accuracy_score(all_trend_true, all_trend_preds)
    
    return {
        'mae_1m': mae_1m,
        'mae_3m': mae_3m,
        'mae_6m': mae_6m,
        'overall_mae': overall_mae,
        'r2_1m': r2_1m,
        'r2_3m': r2_3m,
        'r2_6m': r2_6m,
        'overall_r2': overall_r2,
        'trend_accuracy': trend_acc,
        'trend_true': all_trend_true,
        'trend_preds': all_trend_preds
    }

def train_and_eval_bilstm(vocab, corpus, device='cpu', epochs=20, lr=0.002):
    """
    Trains and evaluates the Bidirectional LSTM on clinical notes.
    """
    texts = [t for t, l in corpus]
    labels = [l for t, l in corpus]
    
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    tr_txt, val_txt, tr_lbl, val_lbl = train_test_split(tr_txt, tr_lbl, test_size=0.15, random_state=42, stratify=tr_lbl)
    
    train_ds = ClinicalJournalDataset(tr_txt, tr_lbl, vocab)
    val_ds = ClinicalJournalDataset(val_txt, val_lbl, vocab)
    test_ds = ClinicalJournalDataset(te_txt, te_lbl, vocab)
    
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = ClinicalJournalBiLSTM(vocab_size=len(vocab), embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_ld:
            ids = batch['input_ids'].to(device)
            lbl = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_ld:
                ids = batch['input_ids'].to(device)
                lbl = batch['label'].to(device)
                logits = model(ids)
                val_loss += criterion(logits, lbl).item()
        val_loss /= len(val_ld)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict().copy()
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    # Evaluate
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_ld:
            ids = batch['input_ids'].to(device)
            logits = model(ids)
            p = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
            preds.extend(p)
            trues.extend(batch['label'].numpy())
            
    acc = accuracy_score(trues, preds)
    return model, acc, trues, preds


def main():
    print("=" * 70)
    print("DERMAVISION AI — MODULE 4: SEQUENTIAL DEEP LEARNING WORKFLOW")
    print("=" * 70)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Hardware Compute Device: {device.upper()}")
    
    # 1. Dataset Preparation from PAD-UFES-20
    print("\n1. Loading and formulating longitudinal progression sequences from PAD-UFES-20...")
    seqs, t1, t3, t6, trends = build_longitudinal_dataset(base_dir=BASE_DIR, seq_len=4, seed=42)
    print(f"   - Generated Sequences Shape: {seqs.shape} (N={len(seqs)} samples, T=4 visits, D=8 features)")
    print(f"   - Targets: 1M {t1.shape}, 3M {t3.shape}, 6M {t6.shape}")
    print(f"   - Trajectory Class Distribution: Improving={sum(trends==0)}, Stable={sum(trends==1)}, Worsening={sum(trends==2)}")
    
    # Splits (70% Train, 15% Val, 15% Test)
    idx_train, idx_temp = train_test_split(np.arange(len(seqs)), test_size=0.30, random_state=42, stratify=trends)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=trends[idx_temp])
    
    full_ds = LongitudinalSkinDataset(seqs, t1, t3, t6, trends)
    train_loader = DataLoader(Subset(full_ds, idx_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, idx_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(Subset(full_ds, idx_test), batch_size=32, shuffle=False)
    
    # 2. Train Model 1: SkinProgressionLSTM
    print("\n2. Training Architecture 1: SkinProgressionLSTM (2-layer Stacked LSTM + LayerNorm)...")
    lstm_model = SkinProgressionLSTM(input_dim=8, hidden_dim=64, num_layers=2, dropout=0.25)
    lstm_model, lstm_tr_loss, lstm_val_loss = train_progression_model(
        lstm_model, train_loader, val_loader, epochs=25, lr=0.001, device=device
    )
    lstm_eval = evaluate_progression_model(lstm_model, test_loader, device=device)
    print(f"   [LSTM RESULTS] Test MAE: {lstm_eval['overall_mae']:.4f} | R2: {lstm_eval['overall_r2']:.4f} | Trend Acc: {lstm_eval['trend_accuracy']*100:.2f}%")
    
    # 3. Train Model 2: SkinProgressionGRU (Ablation Comparison)
    print("\n3. Training Architecture 2: SkinProgressionGRU (Comparative Gated Recurrent Unit)...")
    gru_model = SkinProgressionGRU(input_dim=8, hidden_dim=64, num_layers=2, dropout=0.25)
    gru_model, gru_tr_loss, gru_val_loss = train_progression_model(
        gru_model, train_loader, val_loader, epochs=25, lr=0.001, device=device
    )
    gru_eval = evaluate_progression_model(gru_model, test_loader, device=device)
    print(f"   [GRU RESULTS]  Test MAE: {gru_eval['overall_mae']:.4f} | R2: {gru_eval['overall_r2']:.4f} | Trend Acc: {gru_eval['trend_accuracy']*100:.2f}%")
    
    # 4. Train Model 3: ClinicalJournalBiLSTM (NLP Notes Analyzer)
    print("\n4. Training Architecture 3: ClinicalJournalBiLSTM (Clinical Notes & Journal Analyzer)...")
    aug_corpus = generate_augmented_clinical_corpus(multiplier=25, seed=42)
    vocab = build_vocab(aug_corpus)
    print(f"   - Clinical Vocabulary Size: {len(vocab)} tokens")
    bilstm_model, bilstm_acc, bilstm_trues, bilstm_preds = train_and_eval_bilstm(
        vocab, aug_corpus, device=device, epochs=18, lr=0.002
    )
    print(f"   [BiLSTM RESULTS] Clinical Feedback Classification Accuracy: {bilstm_acc*100:.2f}%")
    
    # 5. Save Checkpoints & Artifacts
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    lstm_path = os.path.join(models_dir, 'module4_lstm_progression_best.pth')
    gru_path = os.path.join(models_dir, 'module4_gru_progression_best.pth')
    bilstm_path = os.path.join(models_dir, 'module4_bilstm_notes_best.pth')
    vocab_path = os.path.join(models_dir, 'module4_vocab.json')
    
    torch.save(lstm_model.state_dict(), lstm_path)
    torch.save(gru_model.state_dict(), gru_path)
    torch.save(bilstm_model.state_dict(), bilstm_path)
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
        
    print(f"\nSaved Module 4 Checkpoints to:")
    print(f" - {lstm_path}")
    print(f" - {gru_path}")
    print(f" - {bilstm_path}")
    print(f" - {vocab_path}")
    
    # 6. Save Comparison CSV
    results_data = [
        {
            "Model": "SkinProgressionLSTM",
            "Recurrent_Type": "LSTM",
            "Task": "Longitudinal Forecasting (1m, 3m, 6m)",
            "Overall_MAE": round(lstm_eval['overall_mae'], 4),
            "Overall_R2": round(lstm_eval['overall_r2'], 4),
            "Trend_Accuracy": f"{lstm_eval['trend_accuracy']*100:.2f}%"
        },
        {
            "Model": "SkinProgressionGRU",
            "Recurrent_Type": "GRU",
            "Task": "Longitudinal Forecasting (1m, 3m, 6m)",
            "Overall_MAE": round(gru_eval['overall_mae'], 4),
            "Overall_R2": round(gru_eval['overall_r2'], 4),
            "Trend_Accuracy": f"{gru_eval['trend_accuracy']*100:.2f}%"
        },
        {
            "Model": "ClinicalJournalBiLSTM",
            "Recurrent_Type": "BiLSTM",
            "Task": "Clinical Notes & Journal Feedback NLP",
            "Overall_MAE": "N/A",
            "Overall_R2": "N/A",
            "Trend_Accuracy": f"{bilstm_acc*100:.2f}%"
        }
    ]
    results_df = pd.DataFrame(results_data)
    csv_out = os.path.join(BASE_DIR, "DermaVision_Module4_Experiment_Results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"Saved results summary to: {csv_out}")
    
    # 7. Plot Curves
    plt.figure(figsize=(10, 4.5))
    plt.plot(lstm_tr_loss, label='LSTM Train Loss', color='#2563eb', linewidth=2)
    plt.plot(lstm_val_loss, label='LSTM Val Loss', color='#3b82f6', linestyle='--', linewidth=2)
    plt.plot(gru_tr_loss, label='GRU Train Loss', color='#10b981', linewidth=2)
    plt.plot(gru_val_loss, label='GRU Val Loss', color='#34d399', linestyle='--', linewidth=2)
    plt.title('DermaVision Module 4: Multi-Horizon Forecasting Loss (LSTM vs GRU)', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Multi-Task Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_png = os.path.join(BASE_DIR, 'Module4_Progression_Loss_Curves.png')
    plt.savefig(loss_png, dpi=300)
    plt.close()
    print(f"Saved performance plot to: {loss_png}")
    
    print("\n" + "=" * 70)
    print("MODULE 4 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    main()
