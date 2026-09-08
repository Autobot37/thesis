import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ============================================================
# DLinear  (Zeng et al., 2023)
# ============================================================

class MovingAvg(nn.Module):
    """Moving average kernel for trend extraction."""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, L]
        front = x[:, :1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, self.kernel_size // 2)
        x_padded = torch.cat([front, x, end], dim=1)
        return self.avg(x_padded.unsqueeze(1)).squeeze(1)

class DLinear(nn.Module):
    """DLinear with proper moving-average trend decomposition and city embeddings."""
    def __init__(self, seq_len=30, pred_len=7, kernel_size=25, num_cities=10):
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_season = nn.Linear(seq_len, pred_len)
        self.city_emb = nn.Embedding(num_cities, pred_len)

    def forward(self, x, c):
        # x: [B, seq_len, 1]
        # c: [B]
        x = x.squeeze(-1)                                          # [B, L]
        trend = self.decomp(x)                                      # [B, L]
        season = x - trend                                          # [B, L]
        
        # Add city-specific bias to the prediction
        bias = self.city_emb(c)                                     # [B, H]
        out = self.linear_trend(trend) + self.linear_season(season) + bias
        return out.unsqueeze(-1)                                    # [B, H, 1]

# ============================================================
# PatchTST  (Nie et al., 2023)  simplified
# ============================================================

class PatchTST(nn.Module):
    """Simplified PatchTST baseline with city embeddings."""
    def __init__(self, seq_len=30, pred_len=7, patch_len=5, stride=5,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1, num_cities=10, use_spatial=True):
        super().__init__()
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # City embedding is added to every patch
        self.use_spatial = use_spatial
        if use_spatial:
            self.city_emb = nn.Embedding(num_cities, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.num_patches + 2, pred_len)

        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x, c, d=None):
        # x: [B, seq_len, 1]
        # c: [B]
        x = x.squeeze(-1)                                                       # [B, L]
        patches = x.unfold(-1, self.patch_len, self.stride)                      # [B, P, patch_len]
        
        c_emb = self.city_emb(c).unsqueeze(1) if self.use_spatial else 0                                    # [B, 1, d]
        h = self.patch_proj(patches) + self.pos_emb + c_emb                      # [B, P, d]
        h = self.transformer(h)                                                  # [B, P, d]
        
        h_flat = h.reshape(x.size(0), -1)
        if d is not None:
            h_flat = torch.cat([h_flat, d], dim=-1)
            
        pred = self.head(h_flat)                               # [B, H]
        return pred.unsqueeze(-1)                                                # [B, H, 1]

class PatchTST_Gaussian(nn.Module):
    """
    Direct parametric (mu, sigma) baseline without VAE sampling.
    """
    def __init__(self, seq_len=30, pred_len=7, patch_len=5, stride=5, d_model=64, n_heads=4, n_layers=2, num_cities=10, use_spatial=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        self.patch_emb = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Outputs mu and logvar
        self.head = nn.Linear(d_model * self.num_patches, pred_len * 2)
        self.use_spatial = use_spatial
        if use_spatial:
            self.city_bias = nn.Embedding(num_cities, pred_len * 2)
        
    def forward(self, x, city_id):
        # x: [B, seq_len, 1] -> [B, 1, seq_len]
        x = x.transpose(1, 2)
        
        # Patching
        B, C, L = x.shape
        x = x.unfold(-1, self.patch_len, self.stride) # [B, C, num_patches, patch_len]
        x = x.reshape(B * C, self.num_patches, self.patch_len)
        
        # Embed
        x = self.patch_emb(x)
        x = x + self.pos_emb
        
        # Attention
        x = self.transformer(x)
        
        # Flatten and project
        x = x.reshape(B, C, -1)
        x = self.head(x) # [B, 1, pred_len*2]
        
        # Add city bias
        bias = self.city_bias(city_id).unsqueeze(1) if self.use_spatial else 0 # [B, 1, pred_len*2]
        out = x + bias
        out = out.squeeze(1).view(B, self.pred_len, 2)
        
        mu, logvar = torch.chunk(out, 2, dim=-1)
        logvar = torch.clamp(logvar, -8.0, 8.0)
        return mu, logvar
        
    @torch.no_grad()
    def sample(self, x, city_id, n_samples=100):
        mu, logvar = self.forward(x, city_id)
        std = torch.exp(0.5 * logvar)
        
        # [B, H, 1]
        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(mu)
            samples.append(mu + eps * std)
            
        y_sample = torch.stack(samples, dim=0) # [n_samples, B, H, 1]
        return y_sample.cpu().numpy()
