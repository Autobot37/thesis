import modal
import os

app = modal.App("thesis-temporal-training")

image = modal.Image.debian_slim().pip_install(
    "torch", "pandas", "numpy", "scikit-learn", "tqdm", "properscoring"
).add_local_dir("src", remote_path="/root/src")

vol = modal.Volume.from_name("thesis-data-vol", create_if_missing=True)

@app.function(image=image, volumes={"/data": vol}, gpu="T4", timeout=1200)
def train_and_eval():
    import torch
    import torch.nn as nn
    import numpy as np
    from src.dataset import get_dataloaders
    from src.models import PatchTST
    from src.cvae import PatchTST_CVAE, train_cvae
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Dataset
    print("Loading Dataset with Temporal Features...")
    train_loader, val_loader, test_loader, test_dataset = get_dataloaders(
        '/data/india_2000_2024_daily_weather.csv', batch_size=128, seq_len=30, pred_len=14
    )
    
    id_to_scaler = {v['id']: (v['mean'], v['std']) for city, v in test_dataset.city_indices.items()}
    
    # ============================================================
    # Evaluate PatchTST Deterministic
    # ============================================================
    print("\nEvaluating PatchTST (Deterministic) with Temporal Features...")
    model_det = PatchTST(seq_len=30, pred_len=14, d_model=64, num_cities=10).to(device)
    
    # Skip training and load saved weights from volume
    model_det.load_state_dict(torch.load('/data/checkpoints/patchtst_temporal.pt', map_location=device))
    model_det.eval()
    mae_det = []
    
    with torch.no_grad():
        for x, y, c, d in test_loader:
            x, y, c, d = x.to(device), y.to(device), c.to(device), d.to(device)
            pred = model_det(x, c, d)
            
            x_np, y_np, pred_np, c_np = x.cpu().numpy(), y.cpu().numpy(), pred.cpu().numpy(), c.cpu().numpy()
            for i in range(x_np.shape[0]):
                mean, std = id_to_scaler[c_np[i]]
                y_real = y_np[i, :, 0] * std + mean
                pred_real = pred_np[i, :, 0] * std + mean
                mae_det.append(np.abs(y_real - pred_real).mean())
                
    det_mae_score = np.mean(mae_det)
    print(f"--> PatchTST (Temporal) Test MAE: {det_mae_score:.3f}")
    
    # ============================================================
    # Evaluate PatchTST-CVAE
    # ============================================================
    print("\nEvaluating PatchTST-CVAE with Temporal Features...")
    model_cvae = PatchTST_CVAE(seq_len=30, pred_len=14, d_model=64, z_dim=16, num_cities=10).to(device)
    
    # Skip training and load saved weights from volume
    model_cvae.load_state_dict(torch.load('/data/checkpoints/best_CVAE.pt', map_location=device))
    model_cvae.eval()
    
    import properscoring as ps
    mae_cvae = []
    crps_cvae = []
    coverage_cvae = []
    
    with torch.no_grad():
        for x, y, c, d in test_loader:
            x, y, c, d = x.to(device), y.to(device), c.to(device), d.to(device)
            samples = model_cvae.sample(x, c, d, n_samples=100) # [100, B, H, 1]
            
            x_np, y_np, c_np = x.cpu().numpy(), y.cpu().numpy(), c.cpu().numpy()
            
            for i in range(x_np.shape[0]):
                mean, std = id_to_scaler[c_np[i]]
                y_real = y_np[i, :, 0] * std + mean
                samples_real = samples[:, i, :, 0] * std + mean # [100, H]
                
                median_pred = np.median(samples_real, axis=0)
                mae_cvae.append(np.abs(y_real - median_pred).mean())
                
                # CRPS expects truth [H], samples [H, n_samples]
                crps_cvae.append(np.mean(ps.crps_ensemble(y_real, samples_real.T)))
                
                # 90% Coverage (PICP)
                p05 = np.percentile(samples_real, 5, axis=0)
                p95 = np.percentile(samples_real, 95, axis=0)
                coverage_cvae.append(np.mean((y_real >= p05) & (y_real <= p95)))
                
    cvae_mae_score = np.mean(mae_cvae)
    cvae_crps_score = np.mean(crps_cvae)
    cvae_cov_score = np.mean(coverage_cvae) * 100.0
    print(f"--> PatchTST-CVAE (Temporal) Test MAE: {cvae_mae_score:.3f}")
    print(f"--> PatchTST-CVAE (Temporal) Test CRPS: {cvae_crps_score:.3f}")
    print(f"--> PatchTST-CVAE (Temporal) 90% Coverage: {cvae_cov_score:.1f}%")
    
    return det_mae_score, cvae_mae_score, cvae_crps_score, cvae_cov_score

@app.local_entrypoint()
def main():
    import json
    print("Deploying temporal training to Modal...")
    det_mae, cvae_mae, cvae_crps, cvae_cov = train_and_eval.remote()
    
    print("\n\n====================================")
    print("RESULTS vs PREVIOUS (No Temporal):")
    print("Previous PatchTST MAE:  1.638")
    print(f"New PatchTST MAE:       {det_mae:.3f}")
    print("------------------------------------")
    print("Previous CVAE MAE:      1.651")
    print(f"New CVAE MAE:           {cvae_mae:.3f}")
    print("Previous CVAE CRPS:     1.188")
    print(f"New CVAE CRPS:          {cvae_crps:.3f}")
    print(f"CVAE 90% Coverage:      {cvae_cov:.1f}%")
    print("====================================\n")
    
    with open("brain/eda_results/temporal_results.md", "w") as f:
        f.write("# Temporal Cyclical Encoding Results\n\n")
        f.write("| Model | Previous MAE | New MAE | Previous CRPS | New CRPS | 90% Coverage |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(f"| PatchTST | 1.638 | {det_mae:.3f} | N/A | N/A | N/A |\n")
        f.write(f"| PatchTST-CVAE | 1.651 | {cvae_mae:.3f} | 1.188 | {cvae_crps:.3f} | {cvae_cov:.1f}% |\n")
