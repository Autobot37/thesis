import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

print("--- 1. LOADING AND EDA ---")
df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['season'] = df['month'].apply(get_season)

print("\n--- 2. PREPARING DATA ---")
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
            
            # Normalize
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
test_dataset = WeatherDatasetLocal(df, seq_len=SEQ_LEN, pred_len=PRED_LEN, split='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_cities = train_dataset.num_cities

print("\n--- 3. MODEL DEFINITIONS ---")
class MovingAvg(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, :1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, self.kernel_size // 2)
        x_padded = torch.cat([front, x, end], dim=1)
        return self.avg(x_padded.unsqueeze(1)).squeeze(1)

class DLinear_NoEmb(nn.Module):
    def __init__(self, seq_len=30, pred_len=14, kernel_size=25):
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_season = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = x.squeeze(-1)
        trend = self.decomp(x)
        season = x - trend
        out = self.linear_trend(trend) + self.linear_season(season)
        return out.unsqueeze(-1)

class DLinear_WithEmb(nn.Module):
    def __init__(self, seq_len=30, pred_len=14, kernel_size=25, num_cities=10):
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_season = nn.Linear(seq_len, pred_len)
        self.city_emb = nn.Embedding(num_cities, pred_len)

    def forward(self, x, c):
        x = x.squeeze(-1)
        trend = self.decomp(x)
        season = x - trend
        bias = self.city_emb(c)
        out = self.linear_trend(trend) + self.linear_season(season) + bias
        return out.unsqueeze(-1)


print("\n--- 4. TRAINING DLINEAR (NO EMBEDDINGS) ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_no_emb = DLinear_NoEmb(seq_len=SEQ_LEN, pred_len=PRED_LEN).to(device)
optimizer_no = torch.optim.Adam(model_no_emb.parameters(), lr=1e-3)
criterion = nn.MSELoss()

model_no_emb.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for x, y, c in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer_no.zero_grad()
        pred = model_no_emb(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer_no.step()
        epoch_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"No-Emb Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

print("\n--- 5. TRAINING DLINEAR (WITH EMBEDDINGS) ---")
model_with_emb = DLinear_WithEmb(seq_len=SEQ_LEN, pred_len=PRED_LEN, num_cities=num_cities).to(device)
optimizer_with = torch.optim.Adam(model_with_emb.parameters(), lr=1e-3)

model_with_emb.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for x, y, c in train_loader:
        x, y, c = x.to(device), y.to(device), c.to(device)
        optimizer_with.zero_grad()
        pred = model_with_emb(x, c)
        loss = criterion(pred, y)
        loss.backward()
        optimizer_with.step()
        epoch_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"With-Emb Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

print("\n--- 6. EVALUATION ---")
model_no_emb.eval()
model_with_emb.eval()

preds_no, preds_with, all_trues = [], [], []
with torch.no_grad():
    for x, y, c in test_loader:
        x, c = x.to(device), c.to(device)
        preds_no.append(model_no_emb(x).cpu().numpy())
        preds_with.append(model_with_emb(x, c).cpu().numpy())
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

print("\n[Comparison MSE - City x Season]")
print(agg_res.to_string(index=False))

# Plotting Comparison
plt.figure(figsize=(16, 6))
plot_df = agg_res.melt(id_vars=['city', 'season'], value_vars=['mse_no_emb', 'mse_with_emb'], var_name='Model', value_name='MSE')
plot_df['Model'] = plot_df['Model'].map({'mse_no_emb': 'No City Embeddings', 'mse_with_emb': 'With City Embeddings'})
plot_df['City_Season'] = plot_df['city'] + " (" + plot_df['season'] + ")"

sns.barplot(data=plot_df, x='City_Season', y='MSE', hue='Model', palette='Set1')
plt.title('DLinear: Impact of City Embeddings on MSE (City x Season)')
plt.ylabel('Mean Squared Error (°C²)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'embedding_comparison.png'), dpi=150)
plt.close()

with open(os.path.join(SAVE_DIR, 'embedding_analysis.md'), 'w') as f:
    f.write("# City Embeddings Ablation Analysis\n\n")
    f.write(agg_res.to_markdown(index=False))

print(f"\nSaved plots and markdown to {SAVE_DIR}/")
