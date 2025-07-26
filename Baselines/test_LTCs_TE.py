import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json
import joblib
import time
import os

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=216, input_dim=8, time_embed_dim=5, dropout_rate=0.2):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + time_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, hidden_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, h, x, time_embed):
        input = torch.cat([h, x, time_embed], dim=-1)
        return self.net(input)

class LTC(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=216, output_dim=216, seq_len=4, dt=6.0, dropout_rate=0.2):
        super(LTC, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.dt = dt
        self.ode_func = ODEFunc(hidden_dim, input_dim, time_embed_dim=5, dropout_rate=dropout_rate)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_seq, time_embed_seq):
        b, t, c, H, W = x_seq.shape
        x_seq = x_seq.permute(0, 3, 4, 1, 2).reshape(b * H * W, self.seq_len, -1)
        time_embed_seq = time_embed_seq.unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1, 1).reshape(b * H * W, self.seq_len, 5)
        h = torch.zeros(b * H * W, self.hidden_dim).to(x_seq.device)
        dt = self.dt * 0.1
        for k in range(self.seq_len):
            x_k = x_seq[:, k, :]
            t_k = time_embed_seq[:, k, :]
            if torch.isnan(h).any():
                h = torch.where(torch.isnan(h), torch.zeros_like(h), h)
            k1 = self.ode_func(h, x_k, t_k)
            k2 = self.ode_func(h + 0.5 * dt * k1, x_k, t_k)
            k3 = self.ode_func(h + 0.5 * dt * k2, x_k, t_k)
            k4 = self.ode_func(h + dt * k3, x_k, t_k)
            k1 = torch.clamp(k1, -10, 10)
            k2 = torch.clamp(k2, -10, 10)
            k3 = torch.clamp(k3, -10, 10)
            k4 = torch.clamp(k4, -10, 10)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            h = torch.clamp(h, -100, 100)
        out = self.output_layer(h)
        out = out.reshape(b, H, W, -1).permute(0, 3, 1, 2)
        return out

# GatedFusion模块（修改）
class GatedFusion(nn.Module):
    def __init__(self, C):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(C, C // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(C // 2, C, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ltc_out):
        gate = self.gate(ltc_out)
        output = gate * ltc_out
        return output

# MLP模块（修改）
class MLP(nn.Module):
    def __init__(self, input_dim=216, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(512, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(384, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(192, 1, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# 完整模型（修改）
class WindSpeedPredictor(nn.Module):
    def __init__(self, H, W, tigge_features=8, dropout_rate=0.2, ltc_hidden_dim=216, cbam_reduction=16):
        super(WindSpeedPredictor, self).__init__()
        self.H = H
        self.W = W
        self.ltc = LTC(input_dim=tigge_features, hidden_dim=ltc_hidden_dim, output_dim=ltc_hidden_dim, dropout_rate=dropout_rate)
        self.gated_fusion = GatedFusion(ltc_hidden_dim)
        self.mlp = MLP(input_dim=ltc_hidden_dim, dropout_rate=dropout_rate)
        self.time_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ltc_hidden_dim)
        )
        for m in self.time_embed:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self,  tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq):  # 修改点 27：忽略 tigge_spatial 和 dem_spatial
        b = tigge_seq.size(0)
        ltc_out = self.ltc(tigge_seq, time_features_seq)
        fused = self.gated_fusion(ltc_out)
        pred = self.mlp(fused)
        return pred.squeeze(1)
# 数据集类（无变化）
class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)
        tigge_min = float(self.ds['X_tigge'].min().values)
        tigge_max = float(self.ds['X_tigge'].max().values)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] tigge_features range: [{tigge_min}, {tigge_max}]")
        dem_min = float(self.ds['X_dem'].min().values)
        dem_max = float(self.ds['X_dem'].max().values)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] dem_features range: [{dem_min}, {dem_max}]")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normalizing time features")
        time_data = self.ds['time_features'].values
        self.time_scaler = StandardScaler()
        normalized_time = self.time_scaler.fit_transform(time_data)
        self.ds['time_features_normalized'] = xr.DataArray(
            normalized_time,
            dims=self.ds['time_features'].dims,
            coords={'sample': self.ds['time_features'].coords['sample']}
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating time index")
        times = pd.to_datetime({
            'year': time_data[:, 0],
            'month': time_data[:, 1],
            'day': time_data[:, 2],
            'hour': time_data[:, 3]
        })
        self.ds = self.ds.assign_coords(time=("sample", times)).sortby('time')
        self.time_points = np.unique(self.ds.time.values)
        self.T = len(self.time_points)
        self.samples_per_time = H * W
        self.sample_indices = np.arange(self.T - self.seq_len + 1)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset initialized with {len(self.sample_indices)} samples")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        actual_idx = self.sample_indices[idx]
        t = actual_idx + self.seq_len - 1
        seq_times = self.time_points[t - self.seq_len + 1: t + 1]
        seq_data = []
        time_features_seq = []
        for time in seq_times:
            mask = self.ds.time == time
            tigge_data = self.ds['X_tigge'].sel(sample=mask).values.reshape(self.H, self.W, 8)
            seq_data.append(tigge_data)
            time_features = self.ds['time_features_normalized'].sel(sample=mask).values[0]
            time_features_seq.append(time_features)
        tigge_seq = np.stack(seq_data)
        time_features_seq = np.stack(time_features_seq)
        time_t = self.time_points[t]
        mask_t = self.ds.time == time_t
        tigge_spatial = self.ds['X_tigge'].sel(sample=mask_t).values.reshape(self.H, self.W, 8)
        dem_spatial = self.ds['X_dem'].sel(sample=mask_t).values.reshape(self.H, self.W, 3)
        target = self.ds['y'].sel(sample=mask_t).values.reshape(self.H, self.W)
        time_features_t = time_features_seq[-1]
        return {
            'tigge_spatial': torch.from_numpy(tigge_spatial).float().permute(2, 0, 1),
            'dem_spatial': torch.from_numpy(dem_spatial).float().permute(2, 0, 1),
            'tigge_seq': torch.from_numpy(tigge_seq).float().permute(0, 3, 1, 2),
            'time_features_t': torch.from_numpy(time_features_t).float(),
            'time_features_seq': torch.from_numpy(time_features_seq).float(),
            'target': torch.from_numpy(target).float()
        }
# 指标计算函数（无变化）
def calculate_metrics(pred, target, tigge_wind):
    pred = pred.flatten()
    target = target.flatten()
    tigge_wind = tigge_wind.flatten()
    FA_pred = ((pred - target).abs() < 1).float().mean().item() * 100
    FA_tigge = ((tigge_wind - target).abs() < 1).float().mean().item() * 100
    RMSE_pred = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    RMSE_tigge = torch.sqrt(torch.mean((tigge_wind - target) ** 2)).item()
    MAE_pred = torch.mean((pred - target).abs()).item()
    MAE_tigge = torch.mean((tigge_wind - target).abs()).item()
    mean_target = torch.mean(target).item()
    rRMSE_pred = (RMSE_pred / mean_target) * 100
    rRMSE_tigge = (RMSE_tigge / mean_target) * 100
    rMAE_pred = (MAE_pred / mean_target) * 100
    rMAE_tigge = (MAE_tigge / mean_target) * 100
    R_pred = torch.corrcoef(torch.stack([pred, target]))[0, 1].item()
    R_tigge = torch.corrcoef(torch.stack([tigge_wind, target]))[0, 1].item()
    ss_tot = torch.sum((target - mean_target) ** 2).item()
    ss_res_pred = torch.sum((target - pred) ** 2).item()
    ss_res_tigge = torch.sum((target - tigge_wind) ** 2).item()
    R2_pred = 1 - (ss_res_pred / ss_tot) if ss_tot > 0 else 0
    R2_tigge = 1 - (ss_res_tigge / ss_tot) if ss_tot > 0 else 0
    return {
        'FA_pred': FA_pred, 'RMSE_pred': RMSE_pred, 'MAE_pred': MAE_pred,
        'rRMSE_pred': rRMSE_pred, 'rMAE_pred': rMAE_pred, 'R_pred': R_pred, 'R2_pred': R2_pred,
        'FA_tigge': FA_tigge, 'RMSE_tigge': RMSE_tigge, 'MAE_tigge': MAE_tigge,
        'rRMSE_tigge': rRMSE_tigge, 'rMAE_tigge': rMAE_tigge, 'R_tigge': R_tigge, 'R2_tigge': R2_tigge
    }

def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_tigge_wind = []
    all_dates = []
    criterion = nn.SmoothL1Loss()

    # 加载标准化器（无变化）
    scaler_target = joblib.load('./show_relevance_visualization/target_scaler.pkl')
    scaler_tigge = joblib.load('./show_relevance_visualization/tigge_feature_scaler.pkl')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Scaler files loaded successfully")
    target_data_min = scaler_target.data_min_[0]
    target_range = 1 / scaler_target.scale_[0]
    tigge_data_min = scaler_tigge.data_min_[26]
    tigge_range = 1 / scaler_tigge.scale_[26]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Target scaler: data_min={target_data_min}, range={target_range}")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Tigge scaler (wind speed, index 26): data_min={tigge_data_min}, range={tigge_range}")

    with torch.no_grad():
        test_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(test_loader):
            try:
                tigge_spatial = batch['tigge_spatial'].to(device)
                dem_spatial = batch['dem_spatial'].to(device)
                tigge_seq = batch['tigge_seq'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                time_features_seq = batch['time_features_seq'].to(device)
                target = batch['target'].to(device)
                output = model(tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq)
                loss = criterion(output, target)
                test_loss += loss.item() * tigge_seq.size(0)
                batch_count += 1
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
                all_tigge_wind.append(batch['tigge_spatial'][:, 0, :, :].cpu())

                # 日期提取逻辑
                start_idx = batch_idx * test_loader.batch_size
                end_idx = min((batch_idx + 1) * test_loader.batch_size, len(test_loader.dataset))
                batch_dates = test_loader.dataset.time_points[start_idx:end_idx]
                all_dates.extend(batch_dates)

                if batch_idx % 100 == 0 or batch_idx == len(test_loader) - 1:
                    progress = (batch_idx + 1) / len(test_loader) * 100
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test progress: {progress:.1f}%")

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing test batch {batch_idx}: {str(e)}")
                continue

        test_loss = test_loss / len(test_loader.dataset) if batch_count > 0 else float('inf')
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Test Loss: {test_loss:.4f}')

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_tigge_wind = torch.cat(all_tigge_wind, dim=0).numpy()
        all_dates = np.array(all_dates[:len(all_preds)])

        all_preds_orig = (all_preds * target_range) + target_data_min
        all_targets_orig = (all_targets * target_range) + target_data_min
        all_tigge_wind_orig = (all_tigge_wind * tigge_range) + tigge_data_min
        all_preds_orig = np.clip(all_preds_orig, 0, 100)
        all_targets_orig = np.clip(all_targets_orig, 0, 100)
        all_tigge_wind_orig = np.clip(all_tigge_wind_orig, 0, 100)

        np.save('LTCs_test_preds.npy', all_preds_orig)
        np.save('LTCs_test_targets.npy', all_targets_orig)
        np.save('LTCs_test_tigge_wind.npy', all_tigge_wind_orig)
        np.save('LTCs_test_dates.npy', all_dates)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test predictions, targets, TIGGE wind, and dates saved as LTCs_test_*.npy files.")

        # 计算指标
        metrics_all = calculate_metrics(
            torch.from_numpy(all_preds_orig),
            torch.from_numpy(all_targets_orig),
            torch.from_numpy(all_tigge_wind_orig)
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Yearly Test Metrics (LTCs):", metrics_all)

        seasons = {
            'Spring': (3, 5),
            'Summer': (6, 8),
            'Autumn': (9, 11),
            'Winter': (12, 2)
        }
        seasonal_metrics = {}
        for season, (start_month, end_month) in seasons.items():
            if start_month < end_month:
                mask = (pd.to_datetime(all_dates).month >= start_month) & (pd.to_datetime(all_dates).month <= end_month)
            else:
                mask = (pd.to_datetime(all_dates).month >= start_month) | (pd.to_datetime(all_dates).month <= end_month)
            season_preds = all_preds_orig[mask]
            season_targets = all_targets_orig[mask]
            season_tigge = all_tigge_wind_orig[mask]
            seasonal_metrics[season] = calculate_metrics(
                torch.from_numpy(season_preds),
                torch.from_numpy(season_targets),
                torch.from_numpy(season_tigge)
            )
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {season} Test Metrics (LTCs):",
                  seasonal_metrics[season])

        all_metrics = {'Yearly': metrics_all, **seasonal_metrics}
        with open('LTCs_test_metrics.json', 'w') as f:
            json.dump(all_metrics, f)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test metrics saved as LTCs_test_metrics.json")

        monthly_metrics = {}
        months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = list(range(1, 13))
        for month in range(1, 13):
            month_mask = (pd.to_datetime(all_dates).month == month)
            if np.sum(month_mask) > 0:
                month_preds = all_preds_orig[month_mask]
                month_targets = all_targets_orig[month_mask]
                month_tigge = all_tigge_wind_orig[month_mask]
                monthly_metrics[month] = calculate_metrics(
                    torch.from_numpy(month_preds),
                    torch.from_numpy(month_targets),
                    torch.from_numpy(month_tigge)
                )
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Month {months_names[month - 1]} Test Metrics (LTCs):",
                    monthly_metrics[month])

        monthly_metrics_dict = {str(k): v for k, v in monthly_metrics.items()}
        with open('LTCs_test_monthly_metrics.json', 'w') as f:
            json.dump(monthly_metrics_dict, f)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monthly test metrics saved as LTCs_test_monthly_metrics.json")

if __name__ == "__main__":
    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets...")
    batch_size = 16
    test_ds = WindDataset("./show_relevance_visualization/test.nc", H, W)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Datasets loaded successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
    model = WindSpeedPredictor(
        H, W,
        tigge_features=8,
        dropout_rate=0.2,
        ltc_hidden_dim=216
    ).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model_LTCs.pth'))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Testing...")
    test_model(model, test_loader, device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing completed!")