import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class WeatherDataset(Dataset):
    def __init__(self, csv_path, seq_len=30, pred_len=7, target_col='temperature_2m_max', split='train', train_ratio=0.8, val_ratio=0.05):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        
        # Load data
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['city', 'date'])
        
        self.cities = df['city'].unique()
        
        self.X, self.Y, self.C = [], [], []
        self.D = [] # For temporal cyclical features (sin_doy, cos_doy)
        self.city_indices = {}  # for test evaluation per city
        self.city_to_id = {city: i for i, city in enumerate(self.cities)}
        self.num_cities = len(self.cities)
        
        current_idx = 0
        
        for city in self.cities:
            city_df = df[df['city'] == city]
            data = city_df[self.target_col].values.astype(np.float32)
            
            # Extract Day of Year
            doy = city_df['date'].dt.dayofyear.values.astype(np.float32)
            sin_doy = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
            cos_doy = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
            
            # Split sizes
            n = len(data)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            # Fit scaler on train ONLY
            city_mean = data[:train_end].mean()
            city_std = data[:train_end].std()
            if city_std < 1e-6:
                city_std = 1.0
                
            data_norm = (data - city_mean) / city_std
            
            # Get split
            if split == 'train':
                split_data = data_norm[:train_end]
                split_sin = sin_doy[:train_end]
                split_cos = cos_doy[:train_end]
            elif split == 'val':
                split_data = data_norm[train_end - seq_len : val_end]
                split_sin = sin_doy[train_end - seq_len : val_end]
                split_cos = cos_doy[train_end - seq_len : val_end]
            else:
                split_data = data_norm[val_end - seq_len :]
                split_sin = sin_doy[val_end - seq_len :]
                split_cos = cos_doy[val_end - seq_len :]
                
            # Create sliding windows
            X_city, Y_city, D_city = [], [], []
            for i in range(len(split_data) - seq_len - pred_len + 1):
                X_city.append(split_data[i : i + seq_len])
                Y_city.append(split_data[i + seq_len : i + seq_len + pred_len])
                # Append sin/cos of the very first day of the prediction horizon
                D_city.append([split_sin[i + seq_len], split_cos[i + seq_len]])
                
            if len(X_city) > 0:
                self.X.append(np.stack(X_city))
                self.Y.append(np.stack(Y_city))
                self.D.append(np.array(D_city, dtype=np.float32))
                
                # City IDs
                c_id = self.city_to_id[city]
                self.C.append(np.full((len(X_city),), c_id, dtype=np.int64))
                
                # Record indices for city mapping
                self.city_indices[city] = {
                    'start': current_idx,
                    'end': current_idx + len(X_city),
                    'mean': city_mean,
                    'std': city_std,
                    'id': c_id
                }
                current_idx += len(X_city)
                
        self.X = np.concatenate(self.X, axis=0)[..., np.newaxis] # [N, seq_len, 1]
        self.Y = np.concatenate(self.Y, axis=0)[..., np.newaxis] # [N, pred_len, 1]
        self.C = np.concatenate(self.C, axis=0)                  # [N]
        self.D = np.concatenate(self.D, axis=0)                  # [N, 2]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx]), torch.tensor(self.C[idx]), torch.tensor(self.D[idx])

def get_dataloaders(csv_path, batch_size=64, seq_len=30, pred_len=7):
    train_dataset = WeatherDataset(csv_path, seq_len, pred_len, split='train')
    val_dataset = WeatherDataset(csv_path, seq_len, pred_len, split='val')
    test_dataset = WeatherDataset(csv_path, seq_len, pred_len, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, test_dataset
