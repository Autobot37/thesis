import modal

app = modal.App("thesis-print-samples")
volume = modal.Volume.from_name("thesis-data-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "pandas", "numpy", "scikit-learn", "properscoring")
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
)
def run_samples():
    import sys
    sys.path.append("/root")
    import torch
    import numpy as np
    
    from src.dataset import get_dataloaders
    from src.models import PatchTST
    from src.cvae import PatchTST_CVAE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = "/data/checkpoints"
    
    train_loader, val_loader, test_loader, test_dataset = get_dataloaders(
        "/data/india_2000_2024_daily_weather.csv", batch_size=128, seq_len=30, pred_len=14
    )
    
    id_to_scaler = {v['id']: (v['mean'], v['std'], city) for city, v in test_dataset.city_indices.items()}
    
    pt_model = PatchTST(num_cities=10, pred_len=14).to(device)
    pt_model.load_state_dict(torch.load(f"{ckpt_dir}/best_PatchTST_14d.pt", map_location=device))
    pt_model.eval()
    
    cvae_model = PatchTST_CVAE(num_cities=10, z_dim=16, pred_len=14).to(device)
    cvae_model.load_state_dict(torch.load(f"{ckpt_dir}/cvae_z16_14d.pt", map_location=device))
    cvae_model.eval()
    
    # Grab one batch
    x, y, c = next(iter(test_loader))
    x, y, c = x.to(device), y.to(device), c.to(device)
    
    with torch.no_grad():
        pt_preds = pt_model(x, c)
        cvae_samples = cvae_model.sample(x, c, n_samples=100) # [100, B, H, 1]
    
    results = []
    # Pick 20 evenly spaced samples from the batch
    indices = np.linspace(0, len(x)-1, 20, dtype=int)
    
    for i in indices:
        c_id = c[i].item()
        mean, std, city_name = id_to_scaler[c_id]
        
        sample_data = {"city": city_name, "days": {}}
        
        # We will look at Days 3, 6, 9, 12 (0-indexed: 2, 5, 8, 11)
        for day, day_idx in [(3, 2), (6, 5), (9, 8), (12, 11)]:
            true_val = y[i, day_idx, 0].item() * std + mean
            pt_val = pt_preds[i, day_idx, 0].item() * std + mean
            
            cvae_dist = cvae_samples[:, i, day_idx, 0] * std + mean
            cvae_med = np.median(cvae_dist)
            cvae_p5 = np.percentile(cvae_dist, 5)
            cvae_p95 = np.percentile(cvae_dist, 95)
            
            sample_data["days"][day] = {
                "true": round(true_val, 1),
                "pt": round(pt_val, 1),
                "cvae_med": round(cvae_med, 1),
                "cvae_p5": round(cvae_p5, 1),
                "cvae_p95": round(cvae_p95, 1)
            }
        
        results.append(sample_data)
        
    return results

@app.local_entrypoint()
def main():
    res = run_samples.remote()
    for i, r in enumerate(res):
        print(f"\n=========================================")
        print(f"Sample {i+1} ({r['city']})")
        print(f"=========================================")
        for day, d_res in r["days"].items():
            print(f"Day {day} Forecast:")
            print(f"  Actual Temp: {d_res['true']} C")
            print(f"  PatchTST:    {d_res['pt']} C")
            print(f"  CVAE:        {d_res['cvae_med']} C (90% confident it's between {d_res['cvae_p5']} C and {d_res['cvae_p95']} C)")
