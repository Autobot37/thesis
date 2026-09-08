import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchTST_CVAE(nn.Module):
    """
    PatchTST-based Conditional Variational Autoencoder (CVAE).
    Implements a probabilistic time series forecast using a conditional prior p(z|x)
    and posterior q(z|x,y) with NLL and KL divergence loss.
    """
    def __init__(self, seq_len=30, pred_len=7, patch_len=5, stride=5,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1, num_cities=10, z_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.z_dim = z_dim
        
        # --- ENCODER (Shared, Deterministic Backbone) ---
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_len = patch_len
        self.stride = stride
        
        self.use_spatial = use_spatial
        if use_spatial:
            self.use_spatial = use_spatial
        if use_spatial:
            self.city_emb = nn.Embedding(num_cities, d_model)
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        h_dim = d_model * self.num_patches
        
        # --- CONDITIONAL PRIOR p(z|x) ---
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim + 2, 64), nn.ReLU(),
            nn.Linear(64, z_dim * 2) # Outputs mu and logvar
        )
        
        # --- POSTERIOR q(z|x,y) (Training Only) ---
        self.future_encoder = nn.Sequential(
            nn.Linear(pred_len, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + 2 + d_model, 64), nn.ReLU(),
            nn.Linear(64, z_dim * 2) # Outputs mu and logvar
        )
        
        # --- DECODER p(y|x,z) ---
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + 2 + z_dim, 128), nn.ReLU(),
            nn.Linear(128, pred_len * 2) # Outputs mu and logvar per timestep
        )
        
    def encode_x(self, x, c, d=None):
        """Extracts context vector h from past sequence x."""
        x = x.squeeze(-1) # [B, L]
        patches = x.unfold(-1, self.patch_len, self.stride) # [B, P, patch_len]
        
        c_emb = self.city_emb(c).unsqueeze(1) if self.use_spatial else 0 if self.use_spatial else 0 # [B, 1, d]
        h = self.patch_proj(patches) + self.pos_emb + c_emb # [B, P, d]
        h = self.transformer(h) # [B, P, d]
        h_flat = h.reshape(x.size(0), -1)
        if d is not None:
            h_flat = torch.cat([h_flat, d], dim=-1)
        return h_flat
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, y, c, d=None):
        """
        Forward pass for training.
        Returns: NLL, KL_dimwise (before free bits)
        """
        # 1. Encode context
        h = self.encode_x(x, c, d)
        
        # 2. Prior p(z|x)
        prior_out = self.prior_net(h)
        mu_p, logvar_p = torch.chunk(prior_out, 2, dim=-1)
        logvar_p = torch.clamp(logvar_p, -8.0, 8.0)
        
        # 3. Posterior q(z|x,y)
        y_flat = y.squeeze(-1) # [B, H]
        h_y = self.future_encoder(y_flat) # [B, d_model]
        post_in = torch.cat([h, h_y], dim=-1)
        post_out = self.post_net(post_in)
        mu_q, logvar_q = torch.chunk(post_out, 2, dim=-1)
        logvar_q = torch.clamp(logvar_q, -8.0, 8.0)
        
        # 4. Sample z ~ q(z|x,y)
        z = self.reparameterize(mu_q, logvar_q)
        
        # 5. Decode p(y|x,z)
        dec_in = torch.cat([h, z], dim=-1)
        dec_out = self.decoder(dec_in)
        mu_y, logvar_y = torch.chunk(dec_out, 2, dim=-1) # [B, H]
        logvar_y = torch.clamp(logvar_y, -8.0, 8.0)
        
        # --- Loss Calculation ---
        import math
        
        # 1. Negative Log-Likelihood (or CRPS if implemented in training loop)
        nll = 0.5 * (math.log(2*math.pi) + logvar_y + ((y_flat - mu_y)**2) / torch.exp(logvar_y))
        
        # 2. KL Divergence
        kl_dim = 0.5 * (logvar_p - logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1)
        
        return nll, kl_dim, mu_y, logvar_y
        
    @torch.no_grad()
    def sample(self, x, c, d=None, n_samples=100, add_decoder_noise=True):
        """
        Inference: Generate N sample trajectories.
        add_decoder_noise: If False, returns mu_y directly (structural uncertainty only)
        """
        self.eval()
        B = x.size(0)
        h = self.encode_x(x, c, d)
        
        # Prior p(z|x)
        prior_out = self.prior_net(h)
        mu_p, logvar_p = torch.chunk(prior_out, 2, dim=-1)
        logvar_p = torch.clamp(logvar_p, -8.0, 8.0)
        
        # Vectorized sampling
        # Expand to [B, n_samples, dim]
        mu_p = mu_p.unsqueeze(1).expand(B, n_samples, self.z_dim)
        logvar_p = logvar_p.unsqueeze(1).expand(B, n_samples, self.z_dim)
        h_expanded = h.unsqueeze(1).expand(B, n_samples, -1)
        
        z = self.reparameterize(mu_p, logvar_p) # [B, n_samples, z_dim]
        
        dec_in = torch.cat([h_expanded, z], dim=-1)
        dec_out = self.decoder(dec_in) # [B, n_samples, H*2]
        
        mu_y, logvar_y = torch.chunk(dec_out, 2, dim=-1)
        logvar_y = torch.clamp(logvar_y, -8.0, 8.0)
        
        if add_decoder_noise:
            y_sample = self.reparameterize(mu_y, logvar_y) # [B, n_samples, H]
        else:
            y_sample = mu_y
            
        # Output shape should be [n_samples, B, H, 1]
        y_sample = y_sample.unsqueeze(-1).transpose(0, 1) # [n_samples, B, H, 1]
        
        return y_sample.cpu().numpy()

def gaussian_crps_loss(mu, logvar, y):
    """Differentiable CRPS for Gaussian N(mu, exp(logvar)) against true y."""
    import math
    sigma = torch.exp(0.5 * logvar)
    norm = torch.distributions.Normal(0, 1)
    z = (y - mu) / sigma
    cdf = norm.cdf(z)
    pdf = torch.exp(norm.log_prob(z))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))

def train_cvae(model, train_loader, val_loader, epochs=40, lr=1e-3, 
               free_bits=0.1, kl_anneal_epochs=12, lambda_var=0.0, use_crps=False, device='cuda', save_dir='/root/checkpoints', patience=5, model_name='best_CVAE.pt'):
    import os
    from tqdm import tqdm
    import sys
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_elbo = float('inf')
    epochs_no_improve = 0
    
    pbar = tqdm(range(epochs), file=sys.stdout, desc=f"Training {model_name}")
    for epoch in pbar:
        model.train()
        train_nll = 0
        train_kl = 0
        
        # KL Annealing (Linear ramp to 1.0)
        beta = min(1.0, epoch / max(1, kl_anneal_epochs))
        
        for x, y, c, d in train_loader:
            x, y, c, d = x.to(device), y.to(device), c.to(device), d.to(device)
            
            nll, kl_dim, mu_y, logvar_y = model(x, y, c, d)
            
            # 1. Main Data Loss
            if use_crps:
                y_flat = y.squeeze(-1)
                data_loss = gaussian_crps_loss(mu_y, logvar_y, y_flat).sum(dim=-1).mean()
            else:
                data_loss = nll.sum(dim=-1).mean()
                
            # 2. Variance Penalty
            var_penalty = lambda_var * logvar_y.mean()
            
            # 3. KL Divergence (with Free Bits)
            kl_dim_clamped = torch.clamp(kl_dim, min=free_bits)
            kl = kl_dim_clamped.sum(dim=-1).mean()
            
            # Total Loss
            loss = data_loss + beta * kl + var_penalty
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_nll += data_loss.item()
            train_kl += kl.item()
            
        train_nll /= len(train_loader)
        train_kl /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss_total = 0
        val_kl_total = 0
        with torch.no_grad():
            for x, y, c, d in val_loader:
                x, y, c, d = x.to(device), y.to(device), c.to(device), d.to(device)
                nll, kl_dim, mu_y, logvar_y = model(x, y, c, d)
                kl = kl_dim.sum(dim=-1).mean().item()
                
                if use_crps:
                    y_flat = y.squeeze(-1)
                    val_data_loss = gaussian_crps_loss(mu_y, logvar_y, y_flat).sum(dim=-1).mean().item()
                else:
                    val_data_loss = nll.sum(dim=-1).mean().item()
                    
                val_loss_total += val_data_loss
                val_kl_total += kl
                
        val_loss_total /= len(val_loader)
        val_kl_total /= len(val_loader)
        
        # Evaluate using true un-annealed ELBO (no free bits, beta=1) for model selection
        val_true_elbo = val_loss_total + val_kl_total
        
        pbar.set_postfix({'val_loss': f'{val_loss_total:.4f}', 'val_elbo': f'{val_true_elbo:.4f}', 'beta': f'{beta:.2f}'})
        
        if val_true_elbo < best_val_elbo:
            best_val_elbo = val_true_elbo
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"
Early stopping at epoch {epoch+1}")
                break
            
    return model
