import modal
import os

app = modal.App("thesis-cvae")
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
def run_cvae():
    import sys
    sys.path.append("/root")
    
    import torch
    import numpy as np
    import pickle
    import json
    
    from src.dataset import get_dataloaders
    from src.models import PatchTST
    from src.cvae import PatchTST_CVAE, train_cvae
    from src.train import train_point_model
    from src.evaluate import evaluate_models, plot_forecast_comparison
    
    csv_path = "/data/india_2000_2024_daily_weather.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running CVAE vs PatchTST Benchmark (14-DAY FORECAST) on Modal! Device: {device} ---")
    
    # 1. Load Data
    train_loader, val_loader, test_loader, test_dataset = get_dataloaders(
        csv_path, batch_size=128, seq_len=30, pred_len=14
    )
    
    # Persistent checkpoint directory on Modal volume
    ckpt_dir = "/data/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 2. Train baseline PatchTST
    print("\n--- Training Baseline PatchTST (14d) ---")
    pt_model = PatchTST(seq_len=30, pred_len=14, patch_len=5, stride=5, d_model=64, num_cities=10).to(device)
    pt_ckpt = f"{ckpt_dir}/best_PatchTST_14d.pt"
    if os.path.exists(pt_ckpt):
        print("Found existing PatchTST weights! Loading...")
        pt_model.load_state_dict(torch.load(pt_ckpt, map_location=device))
    else:
        pt_model = train_point_model(pt_model, train_loader, val_loader, epochs=40, lr=1e-3, device=device)
        torch.save(pt_model.state_dict(), pt_ckpt)
    
    # 3. Train CVAE (Best Configuration: Z-Dim = 16)
    print("\n--- Loading Best PatchTST-CVAE (Z=16, 14d) ---")
    cvae_model = PatchTST_CVAE(seq_len=30, pred_len=14, patch_len=5, stride=5, d_model=64, num_cities=10, z_dim=16).to(device)
    cvae_ckpt = f"{ckpt_dir}/cvae_z16_14d.pt"
    if os.path.exists(cvae_ckpt):
        print("Found existing Z=16 CVAE 14d weights! Loading...")
        cvae_model.load_state_dict(torch.load(cvae_ckpt, map_location=device))
    else:
        print("Training CVAE for 14 days...")
        cvae_model = train_cvae(cvae_model, train_loader, val_loader, epochs=40, lr=1e-3, free_bits=0.1, kl_anneal_epochs=12, device=device, save_dir=ckpt_dir)
        torch.save(cvae_model.state_dict(), cvae_ckpt)
    
    # 4. Evaluate and Compare
    print("\n--- Evaluating Models ---")
    pt_model.eval()
    cvae_model.eval()
    
    n_samples = 100
    all_x, all_y, all_c = [], [], []
    pt_preds = []
    cvae_samples = []
    
    with torch.no_grad():
        for x, y, c in test_loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            all_x.append(x.cpu().numpy())
            all_y.append(y.cpu().numpy())
            all_c.append(c.cpu().numpy())
            
            # Baseline PT
            pt_out = pt_model(x, c) # [B, H, 1]
            pt_preds.append(pt_out.cpu().numpy())
            
            # CVAE
            samples = cvae_model.sample(x, c, n_samples=n_samples) # [n_samples, B, H, 1]
            cvae_samples.append(samples)
            
    # Concatenate results
    x_test = np.concatenate(all_x, axis=0)
    y_test = np.concatenate(all_y, axis=0)
    pt_test = np.concatenate(pt_preds, axis=0)
    all_c_cat = np.concatenate(all_c, axis=0)
    
    # CVAE samples: concat along batch dimension
    cvae_test = np.concatenate(cvae_samples, axis=1) # [n_samples, total_B, H, 1]
    
    # --- UN-NORMALIZE to Real Units (Celsius) ---
    id_to_scaler = {v['id']: (v['mean'], v['std']) for city, v in test_dataset.city_indices.items()}
    means = np.array([id_to_scaler[c_val][0] for c_val in all_c_cat])[:, None, None] # [total_B, 1, 1]
    stds = np.array([id_to_scaler[c_val][1] for c_val in all_c_cat])[:, None, None] # [total_B, 1, 1]
    
    x_test = x_test * stds + means
    y_test = y_test * stds + means
    pt_test = pt_test * stds + means
    cvae_test = cvae_test * stds[None, ...] + means[None, ...]
    
    print("\n--- Metrics (in Real °C) ---")
    from src.evaluate import mae, rmse, crps_ensemble, coverage
    
    cvae_median = np.median(cvae_test, axis=0) # [B, H, 1]
    
    pt_mae = float(mae(pt_test, y_test))
    pt_rmse = float(rmse(pt_test, y_test))
    
    cvae_mae = float(mae(cvae_median, y_test))
    cvae_rmse = float(rmse(cvae_median, y_test))
    
    # CRPS expects truth=[B,H], samples=[B,H,n_samples]
    cvae_crps = float(crps_ensemble(y_test.squeeze(-1), cvae_test.squeeze(-1).transpose(1, 2, 0)))
    
    cvae_cov90 = float(coverage(cvae_test, y_test, 5, 95))
    
    print(f"PatchTST MAE: {pt_mae:.4f} | RMSE: {pt_rmse:.4f}")
    print(f"CVAE MAE:     {cvae_mae:.4f} | RMSE: {cvae_rmse:.4f}")
    print(f"CVAE CRPS:    {cvae_crps:.4f} | 90% Cov: {cvae_cov90:.4f}")
    
    # 5. Plot Comparison
    print("\n--- Generating Comparison Plots ---")
    plot_data = {
        'x': x_test,
        'y': y_test,
        'dl': pt_test, # Just use pt_test for DLinear slot to simplify
        'pt': pt_test,
        'td': cvae_test # CVAE samples
    }
    
    os.makedirs("/root/results_cvae_14d", exist_ok=True)
    plot_forecast_comparison(plot_data, save_dir="/root/results_cvae_14d", n_windows=4)
    
    with open("/root/results_cvae_14d/metrics.json", "w") as f:
        json.dump({
            "PatchTST_MAE": pt_mae,
            "CVAE_MAE": cvae_mae,
            "CVAE_CRPS": cvae_crps,
            "CVAE_Cov90": cvae_cov90
        }, f, indent=4)
        
    print("Evaluation complete. Downloading results...")
    
    # Read files to return
    plots = {}
    for i in range(1, 5):
        try:
            with open(f"/root/results_cvae_14d/forecast_window_{i}.png", "rb") as f:
                plots[f"window_{i}"] = f.read()
        except FileNotFoundError:
            pass
            
    with open("/root/results_cvae_14d/forecast_comparison.png", "rb") as f:
        plots["grid"] = f.read()
        
    return {
        "metrics": {
            "PatchTST_MAE": pt_mae,
            "CVAE_MAE": cvae_mae,
            "CVAE_CRPS": cvae_crps,
            "CVAE_Cov90": cvae_cov90
        },
        "plots": plots
    }

@app.local_entrypoint()
def main():
    print("Starting CVAE Benchmark on Modal...")
    out = run_cvae.remote()
    if "error" in out:
        print(f"ERROR: {out['error']}")
        return
        
    print("\n====== METRICS ======")
    for k, v in out["metrics"].items():
        print(f"{k}: {v:.4f}")
        
        
    os.makedirs("results_cvae_14d", exist_ok=True)
    
    for name, b in out["plots"].items():
        fname = f"results_cvae_14d/forecast_comparison_{name}.png" if name == "grid" else f"results_cvae_14d/forecast_{name}.png"
        with open(fname, "wb") as f:
            f.write(b)
        print(f"Saved {fname}!")
