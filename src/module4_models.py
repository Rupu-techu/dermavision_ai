"""
DermaVision AI — Module 4: Sequential Deep Learning Models
============================================================
Architectures:
1. SkinProgressionLSTM: Multi-layer Long Short-Term Memory network for 
   longitudinal skin lesion tracking and multi-horizon forecasting (1, 3, 6 months).
2. SkinProgressionGRU: Multi-layer Gated Recurrent Unit network for comparative ablation.
3. ClinicalJournalBiLSTM: Bidirectional LSTM for analyzing dermatologist notes and
   patient skincare journals to evaluate treatment feedback and adverse reactions.

Author: Mouli Mukherjee (Module 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkinProgressionLSTM(nn.Module):
    """
    Multi-layer LSTM for temporal skin progression forecasting.
    Takes a sequence of past patient visits / scans:
    X shape: (batch_size, sequence_length, num_features)
    
    Produces:
    1. Multi-horizon continuous forecasts (1 month, 3 months, 6 months ahead)
       for: [Skin Health Score, Pigmentation Percentage, Clinical Severity Score]
    2. Trend classification logits: [Improving, Stable, Worsening]
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, dropout=0.25):
        super(SkinProgressionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Recurrent LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Temporal Attention / Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dense Shared Latent Layer
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-Horizon Forecasting Heads (each predicts 3 core metrics: [Health, Pigmentation %, Severity])
        # Forecast at +1 Month
        self.head_1m = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        # Forecast at +3 Months
        self.head_3m = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        # Forecast at +6 Months
        self.head_6m = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        # Trend Classification Head: 3 classes -> 0: Improving, 1: Stable, 2: Worsening
        self.trend_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        """
        Forward pass.
        x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            dict containing:
            - 'forecast_1m': (batch_size, 3)
            - 'forecast_3m': (batch_size, 3)
            - 'forecast_6m': (batch_size, 3)
            - 'trend_logits': (batch_size, 3)
        """
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the hidden representation of the final time step
        last_step = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        last_step = self.layer_norm(last_step)
        
        latent = self.fc_shared(last_step)
        
        # Generate multi-horizon forecasts
        pred_1m = self.head_1m(latent)
        pred_3m = self.head_3m(latent)
        pred_6m = self.head_6m(latent)
        
        # Generate trend logits
        trend_logits = self.trend_classifier(latent)
        
        return {
            'forecast_1m': pred_1m,
            'forecast_3m': pred_3m,
            'forecast_6m': pred_6m,
            'trend_logits': trend_logits
        }


class SkinProgressionGRU(nn.Module):
    """
    Multi-layer GRU (Gated Recurrent Unit) network for longitudinal skin forecasting.
    Provides an empirical comparison against LSTM for architectural ablation.
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, dropout=0.25):
        super(SkinProgressionGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Recurrent GRU Backbone
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.head_1m = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.head_3m = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.head_6m = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        self.trend_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        gru_out, hn = self.gru(x)
        last_step = gru_out[:, -1, :]
        last_step = self.layer_norm(last_step)
        
        latent = self.fc_shared(last_step)
        
        pred_1m = self.head_1m(latent)
        pred_3m = self.head_3m(latent)
        pred_6m = self.head_6m(latent)
        trend_logits = self.trend_classifier(latent)
        
        return {
            'forecast_1m': pred_1m,
            'forecast_3m': pred_3m,
            'forecast_6m': pred_6m,
            'trend_logits': trend_logits
        }


class ClinicalJournalBiLSTM(nn.Module):
    """
    Bidirectional LSTM (BiLSTM) for analyzing clinical notes and patient skincare journals.
    
    Processes text sequences to classify treatment feedback into 3 classes:
    0: Positive / Improving (Treatment working well, pigmentation fading)
    1: Neutral / Monitoring (Condition stable, ongoing monitoring)
    2: Adverse / Irritation (Erythema, burning, itching, peeling, negative reaction)
    """
    def __init__(self, vocab_size=5000, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.3):
        super(ClinicalJournalBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Bidirectional produces 2 * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths=None):
        """
        x: Tensor of token indices (batch_size, max_seq_len)
        lengths: Optional tensor of actual sequence lengths
        """
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        lstm_out, (hn, cn) = self.bilstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Global max-pooling across sequence length for robust salient feature extraction
        pooled, _ = torch.max(lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        logits = self.fc(pooled)  # (batch_size, num_classes)
        return logits
