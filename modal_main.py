import modal
import os

app = modal.App("thesis-timediff-pipeline")
volume = modal.Volume.from_name("thesis-data-vol")

# Create the Modal image with all our required packages
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "tqdm", "properscoring")
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
    timeout=7200,  # 2 hours max
)
def run_thesis_pipeline():
    import sys
    sys.path.append("/root") # Make /root/src importable
    
    import torch
    from src.dataset import get_dataloaders
    from src.models import DLinear, PatchTST, DiffusionTS
    from src.train import train_point_model, train_diffusion
    from src.evaluate import evaluate_models, plot_forecast_comparison, plot_extreme_event, plot_calibration, save_results_json
    
    csv_path = "/data/india_2000_2024_daily_weather.csv"
    
    if not os.path.exists(csv_path):
        return {"error": f"Dataset not found at {csv_path} in Modal volume."}
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running on Modal! Using device: {device} ---")
    
    # 1. Load Data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader, test_dataset = get_dataloaders(
        csv_path, batch_size=128, seq_len=30, pred_len=7
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    
    # 2. Initialize Models (10 Indian cities)
    num_cities = 10
    dlinear = DLinear(seq_len=30, pred_len=7, num_cities=num_cities).to(device)
    patchtst = PatchTST(seq_len=30, pred_len=7, num_cities=num_cities).to(device)
    diffusion = DiffusionTS(seq_len=30, pred_len=7, d_model=128, n_steps=100, num_cities=num_cities).to(device)
    
    # 3. Train Models
    epochs_point = 25
    epochs_ar = 10
    epochs_diff = 50
    
    print(f"\n--- Training DLinear ({epochs_point} epochs) ---")
    dlinear = train_point_model(dlinear, train_loader, val_loader, epochs=epochs_point, lr=1e-3, device=device)
    
    print(f"\n--- Training PatchTST ({epochs_point} epochs) ---")
    patchtst = train_point_model(patchtst, train_loader, val_loader, epochs=epochs_point, lr=1e-3, device=device)
    
    print(f"\n--- Training DiffusionTS ---")
    diffusion = train_diffusion(diffusion, train_loader, val_loader, 
                                epochs_ar=epochs_ar, epochs_diff=epochs_diff, 
                                lr=1e-3, device=device)
    
    # 4. Evaluate & Visualize
    print("\n--- Evaluating Models & Generating Plots ---")
    # Increase samples to 50 for a much better point-estimate via median
    results, plot_data = evaluate_models(dlinear, patchtst, diffusion, test_loader, test_dataset, 
                                         device=device, n_samples=50, ddim_steps=20)
    
    os.makedirs('/root/results', exist_ok=True)
    plot_forecast_comparison(plot_data, save_dir='/root/results')
    plot_extreme_event(plot_data, save_dir='/root/results')
    plot_calibration(plot_data, save_dir='/root/results')
    save_results_json(results, save_dir='/root/results')
    
    # Read the generated files to return them locally
    files_to_return = {}
    for filename in ['forecast_comparison.png', 'extreme_event.png', 'calibration.png', 'metrics.json']:
        filepath = f'/root/results/{filename}'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                files_to_return[filename] = f.read()
                
    return {
        "results": results,
        "files": files_to_return
    }

@app.local_entrypoint()
def main():
    print("Starting execution on Modal (T4 GPU)...")
    out = run_thesis_pipeline.remote()
    
    if "error" in out:
        print(f"ERROR: {out['error']}")
        return
        
    print("\n====== FINAL RESULTS ======")
    results = out["results"]
    for k, v in results.items():
        print(f"{k.ljust(20)}: {v:.4f}")
    
    os.makedirs("results", exist_ok=True)
    for filename, file_bytes in out["files"].items():
        with open(f"results/{filename}", "wb") as f:
            f.write(file_bytes)
        print(f"Saved downstream file to: results/{filename}")
    print("\nPipeline execution completed successfully!")
