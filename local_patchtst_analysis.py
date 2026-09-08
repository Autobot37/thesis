import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['no_emb', 'with_emb', 'eval'], required=True, help="Mode to run")
args = parser.parse_args()

CSV_PATH = 'data/india_2000_2024_daily_weather.csv'
TARGET_COL = 'temperature_2m_max'
SEQ_LEN = 30
PRED_LEN = 14
EPOCHS = 50
BATCH_SIZE = 128
SAVE_DIR = 'brain/eda_results'

os.makedirs(SAVE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'Post-Monsoon'

print(f"--- 1. LOADING DATA (Mode: {args.mode}) ---")
df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['season'] = df['month'].apply(get_season)

class WeatherDatasetLocal(Dataset):
    def __init__(self, df, seq_len=30, pred_len=14, split='train', train_ratio=0.8, val_ratio=0.05):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        df = df.sort_values(['city', 'date'])
        self.cities = df['city'].unique()
        self.num_cities = len(self.cities)
        self.city_to_id = {city: i for i, city in enumerate(self.cities)}
        
        self.X, self.Y, self.C = [], [], []
        self.meta = []
        
        for city in self.cities:
            city_df = df[df['city'] == city].copy()
            data = city_df[TARGET_COL].values.astype(np.float32)
            dates = city_df['date'].values
            seasons = city_df['season'].values
            
            n = len(data)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            city_mean = data[:train_end].mean()
            city_std = data[:train_end].std()
            if city_std < 1e-6: city_std = 1.0
            data_norm = (data - city_mean) / city_std
            
            if split == 'train':
                split_data = data_norm[:train_end]
                split_seasons = seasons[:train_end]
            elif split == 'val':
                split_data = data_norm[train_end - seq_len : val_end]
                split_seasons = seasons[train_end - seq_len : val_end]
            else:
                split_data = data_norm[val_end - seq_len :]
                split_seasons = seasons[val_end - seq_len :]
                
            c_id = self.city_to_id[city]
            
            for i in range(len(split_data) - seq_len - pred_len + 1):
                self.X.append(split_data[i : i + seq_len])
                self.Y.append(split_data[i + seq_len : i + seq_len + pred_len])
                self.C.append(c_id)
                if split == 'test':
                    self.meta.append({
                        'city': city,
                        'target_season': split_seasons[i + seq_len],
                        'city_mean': city_mean,
                        'city_std': city_std
                    })
                    
        self.X = np.stack(self.X, axis=0)[..., np.newaxis]
        self.Y = np.stack(self.Y, axis=0)[..., np.newaxis]
        self.C = np.array(self.C, dtype=np.int64)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx]), torch.tensor(self.C[idx])

train_dataset = WeatherDatasetLocal(df, seq_len=SEQ_LEN, pred_len=PRED_LEN, split='train')
val_dataset = WeatherDatasetLocal(df, seq_len=SEQ_LEN, pred_len=PRED_LEN, split='val')
test_dataset = WeatherDatasetLocal(df, seq_len=SEQ_LEN, pred_len=PRED_LEN, split='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_cities = train_dataset.num_cities

class EarlyStopping:
    def __init__(self, patience=4, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class PatchTST_NoEmb(nn.Module):
    def __init__(self, seq_len=30, pred_len=14, patch_len=5, stride=5,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.num_patches, pred_len)
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        x = x.squeeze(-1)
        patches = x.unfold(-1, self.patch_len, self.stride)
        h = self.patch_proj(patches) + self.pos_emb
        h = self.transformer(h)
        pred = self.head(h.reshape(x.size(0), -1))
        return pred.unsqueeze(-1)


class PatchTST_WithEmb(nn.Module):
    def __init__(self, seq_len=30, pred_len=14, patch_len=5, stride=5,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1, num_cities=10):
        super().__init__()
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        self.city_emb = nn.Embedding(num_cities, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.num_patches, pred_len)
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x, c):
        x = x.squeeze(-1)
        patches = x.unfold(-1, self.patch_len, self.stride)
        c_emb = self.city_emb(c).unsqueeze(1)
        h = self.patch_proj(patches) + self.pos_emb + c_emb
        h = self.transformer(h)
        pred = self.head(h.reshape(x.size(0), -1))
        return pred.unsqueeze(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if args.mode == 'no_emb':
    print("\n--- TRAINING PATCHTST (NO EMBEDDINGS) ---")
    model = PatchTST_NoEmb(seq_len=SEQ_LEN, pred_len=PRED_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=4)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y, c in tqdm(train_loader, desc=f"No-Emb Train {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, c in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        early_stop(v_loss)
        if early_stop.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break
            
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'patchtst_no_emb.pt'))
    print("Saved No-Emb model!")

elif args.mode == 'with_emb':
    print("\n--- TRAINING PATCHTST (WITH EMBEDDINGS) ---")
    model = PatchTST_WithEmb(seq_len=SEQ_LEN, pred_len=PRED_LEN, num_cities=num_cities).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=4)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y, c in tqdm(train_loader, desc=f"With-Emb Train {epoch+1}/{EPOCHS}"):
            x, y, c = x.to(device), y.to(device), c.to(device)
            optimizer.zero_grad()
            pred = model(x, c)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, c in val_loader:
                x, y, c = x.to(device), y.to(device), c.to(device)
                pred = model(x, c)
                val_loss += criterion(pred, y).item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        early_stop(v_loss)
        if early_stop.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break
            
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'patchtst_with_emb.pt'))
    print("Saved With-Emb model!")

elif args.mode == 'eval':
    print("\n--- EVALUATION ---")
    model_no = PatchTST_NoEmb(seq_len=SEQ_LEN, pred_len=PRED_LEN).to(device)
    model_no.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'patchtst_no_emb.pt'), map_location=device))
    model_no.eval()
    
    model_with = PatchTST_WithEmb(seq_len=SEQ_LEN, pred_len=PRED_LEN, num_cities=num_cities).to(device)
    model_with.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'patchtst_with_emb.pt'), map_location=device))
    model_with.eval()

    preds_no, preds_with, all_trues = [], [], []
    with torch.no_grad():
        for x, y, c in tqdm(test_loader, desc="Evaluating"):
            x, c = x.to(device), c.to(device)
            preds_no.append(model_no(x).cpu().numpy())
            preds_with.append(model_with(x, c).cpu().numpy())
            all_trues.append(y.numpy())

    preds_no = np.concatenate(preds_no, axis=0)
    preds_with = np.concatenate(preds_with, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    results = []
    for i in range(len(test_dataset)):
        meta = test_dataset.meta[i]
        city, season = meta['city'], meta['target_season']
        mean, std = meta['city_mean'], meta['city_std']
        
        true_real = all_trues[i] * std + mean
        pred_no_real = preds_no[i] * std + mean
        pred_with_real = preds_with[i] * std + mean
        
        mse_no = np.mean((true_real - pred_no_real)**2)
        mse_with = np.mean((true_real - pred_with_real)**2)
        
        results.append({
            'city': city,
            'season': season,
            'mse_no_emb': mse_no,
            'mse_with_emb': mse_with
        })

    res_df = pd.DataFrame(results)
    agg_res = res_df.groupby(['city', 'season']).mean().reset_index()

    print("\n[PatchTST Comparison MSE - City x Season]")
    print(agg_res.to_string(index=False))

    plt.figure(figsize=(16, 6))
    plot_df = agg_res.melt(id_vars=['city', 'season'], value_vars=['mse_no_emb', 'mse_with_emb'], var_name='Model', value_name='MSE')
    plot_df['Model'] = plot_df['Model'].map({'mse_no_emb': 'No City Embeddings', 'mse_with_emb': 'With City Embeddings'})
    plot_df['City_Season'] = plot_df['city'] + " (" + plot_df['season'] + ")"

    sns.barplot(data=plot_df, x='City_Season', y='MSE', hue='Model', palette='Set2')
    plt.title('PatchTST: Impact of City Embeddings on MSE (City x Season)')
    plt.ylabel('Mean Squared Error (°C²)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'patchtst_embedding_comparison.png'), dpi=150)
    plt.close()

    with open(os.path.join(SAVE_DIR, 'patchtst_embedding_analysis.md'), 'w') as f:
        f.write("# PatchTST City Embeddings Ablation Analysis\n\n")
        f.write(agg_res.to_markdown(index=False))

    print(f"\nSaved plots and markdown to {SAVE_DIR}/")
