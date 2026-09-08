import modal
import os

app = modal.App("thesis-foundation-models")

# We use an image with PyTorch, HuggingFace Transformers, and Chronos-forecasting
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch", 
        "pandas", 
        "numpy", 
        "transformers", 
        "properscoring", 
        "accelerate",
        "git+https://github.com/amazon-science/chronos-forecasting.git"
    )
    .add_local_dir("src", remote_path="/root/src")
)

volume = modal.Volume.from_name("thesis-data-vol")

@app.function(
    image=image, 
    volumes={"/data": volume}, 
    gpu="T4", 
    timeout=3600
)
def evaluate_foundation_models():
    import torch
    import numpy as np
    import pandas as pd
    import properscoring as ps
    from chronos import ChronosPipeline
    import sys
    import os
    
    # Need to add src to path to import dataset
    sys.path.append("/root")
    from src.dataset import WeatherDataset
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    csv_path = "/data/india_2000_2024_daily_weather.csv"
    test_dataset = WeatherDataset(csv_path, seq_len=30, pred_len=14, split='test')
    
    indices = np.arange(len(test_dataset))
    
    results = {}
    
    models_to_test = [
        "amazon/chronos-t5-small", # 46M parameters
        "amazon/chronos-t5-base"   # 200M parameters
    ]
    
    for model_name in models_to_test:
        print(f"\n--- Loading {model_name} ---")
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        
        all_mae = []
        all_crps = []
        
        batch_size = 16 # T4 has 16GB, but 100 autoregressive samples blows up memory quickly
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            
            batch_x_real = []
            batch_y_real = []
            
            for idx in batch_idx:
                x, y, c = test_dataset[idx]
                city = test_dataset.cities[c.item()]
                meta = test_dataset.city_indices[city]
                mean, std = meta['mean'], meta['std']
                
                # Un-normalize: Foundation models require actual real-world values!
                x_real = x.numpy() * std + mean
                y_real = y.numpy() * std + mean
                
                batch_x_real.append(torch.tensor(x_real).squeeze(-1))
                batch_y_real.append(y_real.squeeze(-1))
            
            # Predict with 100 samples (matching our CVAE metric calculations)
            with torch.no_grad():
                forecast = pipeline.predict(
                    batch_x_real, 
                    prediction_length=14, 
                    num_samples=100
                )
            
            forecast_np = forecast.cpu().numpy() # [B, 100, 14]
            y_true_np = np.stack(batch_y_real)   # [B, 14]
            
            # Median-based MAE
            y_pred_median = np.median(forecast_np, axis=1) 
            mae = np.abs(y_pred_median - y_true_np).mean()
            all_mae.append(mae)
            
            # CRPS (properscoring expects samples on the last axis: [B, 14, 100])
            forecast_transposed = np.transpose(forecast_np, (0, 2, 1))
            crps = ps.crps_ensemble(y_true_np, forecast_transposed).mean()
            all_crps.append(crps)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(indices)} | Batch MAE: {mae:.3f} | Batch CRPS: {crps:.3f}")
        
        mean_mae = np.mean(all_mae)
        mean_crps = np.mean(all_crps)
        results[model_name] = {"MAE": mean_mae, "CRPS": mean_crps}
        
        print(f"Finished {model_name} -> MAE: {mean_mae:.3f}, CRPS: {mean_crps:.3f}")
        
        # Free memory before loading next model
        del pipeline
        torch.cuda.empty_cache()
        
    return results

@app.local_entrypoint()
def main():
    print("Initiating Foundation Model Benchmark on Modal GPU...")
    results = evaluate_foundation_models.remote()
    
    print("\n==============================================")
    print("         FOUNDATION MODEL COMPARISON          ")
    print("==============================================")
    for model, mets in results.items():
        print(f"\n{model}")
        print(f"  MAE (°C):  {mets['MAE']:.3f}")
        print(f"  CRPS:      {mets['CRPS']:.3f}")
        
    print("\n[Baseline (Custom PatchTST-CVAE)]")
    print("  MAE (°C):  1.651")
    print("  CRPS:      1.188")
    print("==============================================")
