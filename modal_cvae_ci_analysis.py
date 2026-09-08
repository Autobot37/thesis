import modal
import os

app = modal.App("thesis-cvae-ci")
volume = modal.Volume.from_name("thesis-data-vol")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "pandas", "numpy", "scikit-learn")
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=3600)
def analyze_ci():
    import sys
    sys.path.append("/root")
    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from src.cvae import PatchTST_CVAE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = "/data/india_2000_2024_daily_weather.csv"
    
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Summer'
        elif month in [6, 7, 8, 9]: return 'Monsoon'
        else: return 'Post-Monsoon'

    print("1. Loading dataset and calculating variance...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['season'] = df['date'].dt.month.apply(get_season)
    var_df = df.groupby(['city', 'season'])['temperature_2m_max'].var().reset_index(name='variance')
    
    class WeatherDatasetWithSeasons(Dataset):
        def __init__(self, df, seq_len=30, pred_len=14, split='test', train_ratio=0.8, val_ratio=0.05):
            self.seq_len = seq_len
            self.pred_len = pred_len
            
            df = df.sort_values(['city', 'date'])
            self.cities = df['city'].unique()
            self.city_to_id = {city: i for i, city in enumerate(self.cities)}
            
            self.X, self.Y, self.C = [], [], []
            self.meta = []
            
            for city in self.cities:
                city_df = df[df['city'] == city].copy()
                data = city_df['temperature_2m_max'].values.astype(np.float32)
                seasons = city_df['season'].values
                
                n = len(data)
                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)
                
                city_mean = data[:train_end].mean()
                city_std = data[:train_end].std()
                if city_std < 1e-6: city_std = 1.0
                data_norm = (data - city_mean) / city_std
                
                if split == 'train':
                    split_data, split_seasons = data_norm[:train_end], seasons[:train_end]
                elif split == 'val':
                    split_data, split_seasons = data_norm[train_end - seq_len : val_end], seasons[train_end - seq_len : val_end]
                else:
                    split_data, split_seasons = data_norm[val_end - seq_len :], seasons[val_end - seq_len :]
                    
                c_id = self.city_to_id[city]
                
                for i in range(len(split_data) - seq_len - pred_len + 1):
                    self.X.append(split_data[i : i + seq_len])
                    self.Y.append(split_data[i + seq_len : i + seq_len + pred_len])
                    self.C.append(c_id)
                    self.meta.append({
                        'city': city,
                        'target_season': split_seasons[i + seq_len],
                        'mean': city_mean,
                        'std': city_std
                    })
                    
            self.X = np.stack(self.X, axis=0)[..., np.newaxis]
            self.Y = np.stack(self.Y, axis=0)[..., np.newaxis]
            self.C = np.array(self.C, dtype=np.int64)

        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx]), torch.tensor(self.C[idx])

    print("2. Building Test Loader...")
    test_dataset = WeatherDatasetWithSeasons(df, seq_len=30, pred_len=14, split='test')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("3. Loading CVAE weights...")
    cvae = PatchTST_CVAE(pred_len=14, d_model=64, z_dim=16).to(device)
    cvae.load_state_dict(torch.load("/data/checkpoints/cvae_z16_14d.pt", map_location=device))
    cvae.eval()
    
    print("4. Calculating 90% Confidence Intervals...")
    ci_widths = []
    
    idx_offset = 0
    with torch.no_grad():
        for x, y, c in test_loader:
            x, c = x.to(device), c.to(device)
            
            # samples shape: [n_samples, B, H, 1]
            samples = cvae.sample(x, c, n_samples=100)
            
            batch_size = x.size(0)
            for b in range(batch_size):
                meta = test_dataset.meta[idx_offset + b]
                mean, std = meta['mean'], meta['std']
                
                # shape [100, H]
                item_samples = samples[:, b, :, 0]
                
                # Un-normalize samples for this specific sequence
                item_samples_real = item_samples * std + mean
                
                # 90% CI width (95th percentile - 5th percentile)
                p95 = np.percentile(item_samples_real, 95, axis=0)
                p05 = np.percentile(item_samples_real, 5, axis=0)
                width = (p95 - p05).mean()
                
                ci_widths.append({
                    'city': meta['city'],
                    'season': meta['target_season'],
                    'ci_width_90': width
                })
                
            idx_offset += batch_size

    print("5. Formatting and Merging Results...")
    ci_df = pd.DataFrame(ci_widths)
    ci_agg = ci_df.groupby(['city', 'season'])['ci_width_90'].mean().reset_index()
    
    merged = pd.merge(ci_agg, var_df, on=['city', 'season'])
    corr = merged['variance'].corr(merged['ci_width_90'])
    
    return merged, corr

@app.local_entrypoint()
def main():
    print("Deploying CI analysis to Modal...")
    merged_df, correlation = analyze_ci.remote()
    
    print("\n=== PATCHTST-CVAE 90% CONFIDENCE INTERVAL ANALYSIS ===")
    print(merged_df.to_string(index=False))
    print("\n=======================================================")
    print(f"CORRELATION (Variance vs 90% CI Width): {correlation:.4f}")
    print("=======================================================")
    
    os.makedirs("brain/eda_results", exist_ok=True)
    merged_df.to_csv("brain/eda_results/cvae_ci_analysis.csv", index=False)
    with open("brain/eda_results/cvae_ci_analysis.md", "w") as f:
        f.write("# PatchTST-CVAE Uncertainty Analysis\n\n")
        f.write(f"**Pearson Correlation (Intrinsic Variance vs 90% CI Width):** {correlation:.4f}\n\n")
        f.write(merged_df.to_markdown(index=False))
    print("Saved detailed results to brain/eda_results/cvae_ci_analysis.md")
