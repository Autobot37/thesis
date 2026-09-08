import modal
import os

app = modal.App("thesis-ablations")
volume = modal.Volume.from_name("thesis-data-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "tqdm", "properscoring")
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
    timeout=3600,
)
def run_ablations():
    import sys
    sys.path.append("/root")
    import torch
    import numpy as np
    import json
    
    from src.dataset import get_dataloaders
    from src.cvae import PatchTST_CVAE, train_cvae
    from src.evaluate import mae, rmse, crps_ensemble, coverage
    
    csv_path = "/data/india_2000_2024_daily_weather.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = "/data/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("Loading Data...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(csv_path, batch_size=128)
    
    # Extract test targets for eval
    all_y = []
    for _, y, _ in test_loader:
        all_y.append(y.cpu().numpy())
    y_test = np.concatenate(all_y, axis=0) # [B, H, 1]
    y_test_sq = y_test.squeeze(-1) # [B, H]
    
    def evaluate_cvae(model, add_noise=True):
        model.eval()
        samples = []
        with torch.no_grad():
            for x, _, c in test_loader:
                x, c = x.to(device), c.to(device)
                out = model.sample(x, c, n_samples=100, add_decoder_noise=add_noise)
                samples.append(out)
        cvae_test = np.concatenate(samples, axis=1) # [n_samples, B, H, 1]
        
        cvae_median = np.median(cvae_test, axis=0)
        mae_val = float(mae(cvae_median, y_test))
        crps_val = float(crps_ensemble(y_test_sq, cvae_test.squeeze(-1).transpose(1, 2, 0)))
        cov90 = float(coverage(cvae_test, y_test, 5, 95))
        
        # Empirical Width (mean difference between 95th and 5th percentiles)
        p95 = np.percentile(cvae_test, 95, axis=0)
        p5 = np.percentile(cvae_test, 5, axis=0)
        width = float(np.mean(p95 - p5))
        
        return mae_val, crps_val, cov90, width
        
    results = {}
    
    # 1. Baseline CVAE (Already Trained in /data/checkpoints/best_CVAE.pt)
    print("\n--- 1. Baseline CVAE ---")
    model_base = PatchTST_CVAE(num_cities=10, z_dim=8).to(device)
    model_base.load_state_dict(torch.load(f"{ckpt_dir}/best_CVAE.pt", map_location=device))
    results['Baseline'] = evaluate_cvae(model_base, add_noise=True)
    
    # 2. No Decoder Noise (Structural Uncertainty Only)
    print("\n--- 2. No Decoder Noise ---")
    results['No_Dec_Noise'] = evaluate_cvae(model_base, add_noise=False)
    
    # 3. Z-Dim = 2
    print("\n--- 3. Z-Dim = 2 ---")
    model_z2 = PatchTST_CVAE(num_cities=10, z_dim=2).to(device)
    ckpt_z2 = f"{ckpt_dir}/cvae_z2.pt"
    if not os.path.exists(ckpt_z2):
        model_z2 = train_cvae(model_z2, train_loader, val_loader, epochs=40, device=device, save_dir="/tmp")
        torch.save(model_z2.state_dict(), ckpt_z2)
    model_z2.load_state_dict(torch.load(ckpt_z2, map_location=device))
    results['Z_Dim_2'] = evaluate_cvae(model_z2)
    
    # 4. Z-Dim = 16
    print("\n--- 4. Z-Dim = 16 ---")
    model_z16 = PatchTST_CVAE(num_cities=10, z_dim=16).to(device)
    ckpt_z16 = f"{ckpt_dir}/cvae_z16.pt"
    if not os.path.exists(ckpt_z16):
        model_z16 = train_cvae(model_z16, train_loader, val_loader, epochs=40, device=device, save_dir="/tmp")
        torch.save(model_z16.state_dict(), ckpt_z16)
    model_z16.load_state_dict(torch.load(ckpt_z16, map_location=device))
    results['Z_Dim_16'] = evaluate_cvae(model_z16)
    
    # 5. Variance Penalty
    print("\n--- 5. Variance Penalty ---")
    model_var = PatchTST_CVAE(num_cities=10, z_dim=8).to(device)
    ckpt_var = f"{ckpt_dir}/cvae_varpenalty.pt"
    if not os.path.exists(ckpt_var):
        model_var = train_cvae(model_var, train_loader, val_loader, epochs=40, lambda_var=0.01, device=device, save_dir="/tmp")
        torch.save(model_var.state_dict(), ckpt_var)
    model_var.load_state_dict(torch.load(ckpt_var, map_location=device))
    results['Var_Penalty'] = evaluate_cvae(model_var)
    
    # 6. CRPS Training
    print("\n--- 6. CRPS Training ---")
    model_crps = PatchTST_CVAE(num_cities=10, z_dim=8).to(device)
    ckpt_crps = f"{ckpt_dir}/cvae_crps.pt"
    if not os.path.exists(ckpt_crps):
        model_crps = train_cvae(model_crps, train_loader, val_loader, epochs=40, use_crps=True, device=device, save_dir="/tmp")
        torch.save(model_crps.state_dict(), ckpt_crps)
    model_crps.load_state_dict(torch.load(ckpt_crps, map_location=device))
    results['CRPS_Loss'] = evaluate_cvae(model_crps)
    
    return results

@app.local_entrypoint()
def main():
    print("Starting Ablations on Modal...")
    out = run_ablations.remote()
    
    print("\n====== ABLATION RESULTS ======")
    print(f"{'Experiment':<15} | {'MAE':<7} | {'CRPS':<7} | {'Cov90':<7} | {'Width':<7}")
    print("-" * 55)
    for exp, (mae_v, crps_v, cov, width) in out.items():
        print(f"{exp:<15} | {mae_v:.4f} | {crps_v:.4f} | {cov:.4f} | {width:.4f}")
