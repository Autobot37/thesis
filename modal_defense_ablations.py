import modal

app = modal.App("thesis-defense-ablations")
volume = modal.Volume.from_name("thesis-data-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "pandas", "numpy", "scikit-learn", "properscoring", "matplotlib")
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
    timeout=3600
)
def run_defense():
    import sys
    sys.path.append("/root")
    import torch
    import torch.nn as nn
    import numpy as np
    import math
    import json
    import os
    
    from src.dataset import get_dataloaders
    from src.models import PatchTST, PatchTST_Gaussian
    from src.cvae import PatchTST_CVAE
    from src.evaluate import mae, rmse, crps_ensemble, coverage
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = "/data/checkpoints"
    
    train_loader, val_loader, test_loader, test_dataset = get_dataloaders(
        "/data/india_2000_2024_daily_weather.csv", batch_size=128, seq_len=30, pred_len=14
    )
    
    # Un-normalize scaler mapping
    id_to_scaler = {v['id']: (v['mean'], v['std']) for city, v in test_dataset.city_indices.items()}
    
    # ------------------
    # 1. Parameter Counts
    # ------------------
    pt_base = PatchTST(pred_len=14, d_model=64)
    cvae = PatchTST_CVAE(pred_len=14, d_model=64, z_dim=16)
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    p_base = count_params(pt_base)
    p_cvae = count_params(cvae)
    
    # Widen PatchTST to heavily exceed CVAE
    pt_large = PatchTST(pred_len=14, d_model=128)
    p_large = count_params(pt_large)
    
    # Auto-tune Gaussian baseline to exactly match CVAE parameters
    best_d = 64
    min_diff = float('inf')
    for d in range(64, 128, 4):
        g_test = PatchTST_Gaussian(pred_len=14, d_model=d, n_layers=2) # Enforce n_layers=2
        if abs(count_params(g_test) - p_cvae) < min_diff:
            min_diff = abs(count_params(g_test) - p_cvae)
            best_d = d
            
    gaussian = PatchTST_Gaussian(pred_len=14, d_model=best_d, n_layers=2)
    p_gauss = count_params(gaussian)
    
    print(f"PatchTST (Base): {p_base:,} parameters")
    print(f"CVAE (d=64, z=16): {p_cvae:,} parameters")
    print(f"PatchTST (Large, d=128): {p_large:,} parameters")
    print(f"PatchTST Gaussian (Capacity Matched, d={best_d}): {p_gauss:,} parameters")
    
    # ------------------
    # 2. Train Large Point Model
    # ------------------
    pt_large = pt_large.to(device)
    pt_large_ckpt = f"{ckpt_dir}/PatchTST_Large_14d.pt"
    if os.path.exists(pt_large_ckpt):
        print("Loading Large PatchTST...")
        pt_large.load_state_dict(torch.load(pt_large_ckpt, map_location=device))
    else:
        print("Training Large PatchTST...")
        optimizer = torch.optim.Adam(pt_large.parameters(), lr=1e-3)
        for epoch in range(15):
            pt_large.train()
            for x, y, c in train_loader:
                x, y, c = x.to(device), y.to(device), c.to(device)
                optimizer.zero_grad()
                pred = pt_large(x, c)
                loss = nn.MSELoss()(pred, y)
                loss.backward()
                optimizer.step()
        torch.save(pt_large.state_dict(), pt_large_ckpt)
        
    # ------------------
    # 3. Train Gaussian Head Model
    # ------------------
    gaussian = gaussian.to(device)
    gauss_ckpt = f"{ckpt_dir}/PatchTST_Gaussian_14d_Matched.pt"
    if os.path.exists(gauss_ckpt):
        print("Loading Gaussian Baseline...")
        gaussian.load_state_dict(torch.load(gauss_ckpt, map_location=device))
    else:
        print("Training Gaussian Baseline...")
        optimizer = torch.optim.Adam(gaussian.parameters(), lr=1e-3)
        for epoch in range(25):
            gaussian.train()
            for x, y, c in train_loader:
                x, y, c = x.to(device), y.to(device), c.to(device)
                optimizer.zero_grad()
                mu, logvar = gaussian(x, c)
                # y is [B, H, 1]. mu, logvar are [B, H, 1]
                nll = 0.5 * (math.log(2 * math.pi) + logvar + ((y - mu)**2) / torch.exp(logvar))
                loss = nll.mean()
                loss.backward()
                optimizer.step()
        torch.save(gaussian.state_dict(), gauss_ckpt)
        
    # ------------------
    # 4. Evaluation
    # ------------------
    cvae.load_state_dict(torch.load(f"{ckpt_dir}/cvae_z16_14d.pt", map_location=device))
    cvae.to(device).eval()
    pt_base.load_state_dict(torch.load(f"{ckpt_dir}/best_PatchTST_14d.pt", map_location=device))
    pt_base.to(device).eval()
    pt_large.eval()
    gaussian.eval()
    
    # Collect predictions
    all_y, all_c = [], []
    pt_base_preds, pt_large_preds = [], []
    cvae_samples, gauss_samples = [], []
    
    with torch.no_grad():
        for x, y, c in test_loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            
            all_y.append(y.cpu().numpy())
            all_c.append(c.cpu().numpy())
            
            pt_base_preds.append(pt_base(x, c).cpu().numpy())
            pt_large_preds.append(pt_large(x, c).cpu().numpy())
            
            cvae_samples.append(cvae.sample(x, c, n_samples=100))
            gauss_samples.append(gaussian.sample(x, c, n_samples=100))
            
    y_test = np.concatenate(all_y, axis=0)
    c_test = np.concatenate(all_c, axis=0)
    
    pt_base_test = np.concatenate(pt_base_preds, axis=0)
    pt_large_test = np.concatenate(pt_large_preds, axis=0)
    cvae_test = np.concatenate(cvae_samples, axis=1) # [n_samples, total_B, H, 1]
    gauss_test = np.concatenate(gauss_samples, axis=1) # [n_samples, total_B, H, 1]
    
    # Un-normalize
    means = np.array([id_to_scaler[c_val][0] for c_val in c_test])[:, None, None]
    stds = np.array([id_to_scaler[c_val][1] for c_val in c_test])[:, None, None]
    
    y_test = y_test * stds + means
    pt_base_test = pt_base_test * stds + means
    pt_large_test = pt_large_test * stds + means
    cvae_test = cvae_test * stds[None, ...] + means[None, ...]
    gauss_test = gauss_test * stds[None, ...] + means[None, ...]
    
    # Metrics
    cvae_median = np.median(cvae_test, axis=0)
    gauss_median = np.median(gauss_test, axis=0)
    
    results = {
        "params": {
            "PatchTST": p_base,
            "PatchTST_Large": p_large,
            "PatchTST_Gaussian": p_gauss,
            "CVAE": p_cvae
        },
        "mae": {
            "PatchTST": mae(y_test, pt_base_test),
            "PatchTST_Large": mae(y_test, pt_large_test),
            "PatchTST_Gaussian": mae(y_test, gauss_median),
            "CVAE": mae(y_test, cvae_median)
        },
        "crps": {
            "PatchTST_Gaussian": crps_ensemble(y_test.squeeze(-1), gauss_test.transpose(1, 2, 0, 3).squeeze(-1)),
            "CVAE": crps_ensemble(y_test.squeeze(-1), cvae_test.transpose(1, 2, 0, 3).squeeze(-1))
        },
        "coverage": {
            "PatchTST_Gaussian": coverage(y_test, gauss_test, 0.9),
            "CVAE": coverage(y_test, cvae_test, 0.9)
        }
    }
    
    return results

@app.local_entrypoint()
def main():
    print("Running thesis defense ablations...")
    res = run_defense.remote()
    
    print("\n--- PARAMETERS ---")
    for k, v in res["params"].items():
        print(f"{k}: {v:,}")
        
    print("\n--- MAE (°C) ---")
    for k, v in res["mae"].items():
        print(f"{k}: {v:.3f}")
        
    print("\n--- CRPS ---")
    for k, v in res["crps"].items():
        print(f"{k}: {v:.3f}")
        
    print("\n--- 90% COVERAGE ---")
    for k, v in res["coverage"].items():
        print(f"{k}: {v*100:.2f}%")
