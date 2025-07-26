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
import os


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        global_feat = self.global_pool(x)
        global_feat = self.fc(global_feat.view(x.size(0), -1))
        global_att = self.sigmoid(global_feat).view(x.size(0), -1, 1, 1)
        x = x + x * global_att
        return x

class ResNetCBAM(nn.Module):
    def __init__(self, in_channels=11, dropout_rate=0.3):
        super(ResNetCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 56, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(56)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, 56, kernel_size=1)
        self.conv2 = nn.Conv2d(56, 56, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(56)
        self.conv3 = nn.Conv2d(56, 56, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(56)
        self.conv4 = nn.Conv2d(56, 56, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(56)
        self.conv5 = nn.Conv2d(56, 56, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(56)
        self.cbam = CBAM(56, reduction=16)
        self.dropout = nn.Dropout(p=dropout_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, time_emb):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out) + time_emb
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        identity = out
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        out = self.relu(out)
        identity = out
        out = self.conv4(out)
        out = self.bn4(out)
        out = out + identity
        out = self.relu(out)
        identity = out
        out = self.conv5(out)
        out = self.bn5(out)
        out = out + identity
        out = self.relu(out)
        out = self.dropout(out)
        out = self.cbam(out)
        return out


class GatedFusion(nn.Module):
    def __init__(self, C1):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(C1, C1 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(C1 // 2, C1, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, resnet_out):
        gate = self.gate(resnet_out)
        output = gate * resnet_out
        return output

class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
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

class WindSpeedPredictor(nn.Module):
    def __init__(self, H, W, tigge_features=8, dropout_rate=0.3, cbam_reduction=16):
        super(WindSpeedPredictor, self).__init__()
        self.H = H
        self.W = W
        self.resnet = ResNetCBAM(in_channels=tigge_features + 3, dropout_rate=dropout_rate)  # 修改点 5：使用ResNetCBAM
        self.gated_fusion = GatedFusion(56)
        self.mlp = MLP(56, dropout_rate=dropout_rate)
        self.time_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 56)
        )
        for m in self.time_embed:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self, tigge_spatial, dem_spatial, time_features_t):
        b = tigge_spatial.size(0)
        spatial_input = torch.cat([tigge_spatial, dem_spatial], dim=1)
        time_emb = self.time_embed(time_features_t).view(b, 56, 1, 1)
        resnet_out = self.resnet(spatial_input, time_emb)
        fused = self.gated_fusion(resnet_out)
        pred = self.mlp(fused)
        return pred.squeeze(1)

class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        self.ds = xr.open_dataset(ds_path, cache=False)
        tigge_min = float(self.ds['X_tigge'].min().values)
        tigge_max = float(self.ds['X_tigge'].max().values)
        print(f"tigge_features range: [{tigge_min}, {tigge_max}]")
        dem_min = float(self.ds['X_dem'].min().values)
        dem_max = float(self.ds['X_dem'].max().values)
        print(f"dem_features range: [{dem_min}, {dem_max}]")
        time_data = self.ds['time_features'].values
        self.time_scaler = StandardScaler()
        normalized_time = self.time_scaler.fit_transform(time_data)
        self.ds['time_features_normalized'] = xr.DataArray(
            normalized_time,
            dims=self.ds['time_features'].dims,
            coords={'sample': self.ds['time_features'].coords['sample']}
        )
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

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        actual_idx = self.sample_indices[idx]
        t = actual_idx + self.seq_len - 1
        time_t = self.time_points[t]
        mask_t = self.ds.time == time_t
        tigge_spatial = self.ds['X_tigge'].sel(sample=mask_t).values.reshape(self.H, self.W, 8)
        dem_spatial = self.ds['X_dem'].sel(sample=mask_t).values.reshape(self.H, self.W, 3)
        target = self.ds['y'].sel(sample=mask_t).values.reshape(self.H, self.W)
        time_features_t = self.ds['time_features_normalized'].sel(sample=mask_t).values[0]
        return {
            'tigge_spatial': torch.from_numpy(tigge_spatial).float().permute(2, 0, 1),
            'dem_spatial': torch.from_numpy(dem_spatial).float().permute(2, 0, 1),
            'time_features_t': torch.from_numpy(time_features_t).float(),
            'target': torch.from_numpy(target).float()
        }

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

def validate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_tigge_wind = []
    all_dates = []
    criterion = nn.SmoothL1Loss()

    # 加载标准化器
    scaler_target = joblib.load('./show_relevance_visualization/target_scaler.pkl')
    scaler_tigge = joblib.load('./show_relevance_visualization/tigge_feature_scaler.pkl')
    print("Scaler files loaded successfully")
    target_data_min = scaler_target.data_min_[0]
    target_range = 1 / scaler_target.scale_[0]
    tigge_data_min = scaler_tigge.data_min_[26]
    tigge_range = 1 / scaler_tigge.scale_[26]
    print(f"Target scaler: data_min={target_data_min}, range={target_range}")
    print(f"Tigge scaler (wind speed, index 26): data_min={tigge_data_min}, range={tigge_range}")

    with torch.no_grad():
        val_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(val_loader):
            try:
                tigge_spatial = batch['tigge_spatial'].to(device)
                dem_spatial = batch['dem_spatial'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                target = batch['target'].to(device)
                output = model(tigge_spatial, dem_spatial, time_features_t)
                loss = criterion(output, target)
                val_loss += loss.item() * tigge_spatial.size(0)
                batch_count += 1
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
                all_tigge_wind.append(tigge_spatial[:, 0, :, :].cpu())
                start_idx = batch_idx * val_loader.batch_size
                end_idx = min((batch_idx + 1) * val_loader.batch_size, len(val_loader.dataset))
                batch_dates = val_loader.dataset.time_points[
                              start_idx + val_loader.dataset.seq_len - 1:end_idx + val_loader.dataset.seq_len - 1]
                all_dates.extend(batch_dates)
            except Exception as e:
                print(f"Error processing validation batch {batch_idx}: {str(e)}")
                continue

        val_loss = val_loss / len(val_loader.dataset) if batch_count > 0 else float('inf')
        print(f'Validation Loss: {val_loss:.4f}')

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

        np.save('ResNetCBAMTE_val_preds.npy', all_preds_orig)
        np.save('ResNetCBAMTE_val_targets.npy', all_targets_orig)
        np.save('ResNetCBAMTE_val_tigge_wind.npy', all_tigge_wind_orig)
        np.save('ResNetCBAMTE_val_dates.npy', all_dates)
        print("Validation predictions, targets, TIGGE wind saved as ResNetCBAMTE_*.npy files.")

        metrics_all = calculate_metrics(
            torch.from_numpy(all_preds_orig),
            torch.from_numpy(all_targets_orig),
            torch.from_numpy(all_tigge_wind_orig)
        )
        print("Yearly Validation Metrics (ResNetCBAMTE):", metrics_all)

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
            print(f"{season} Validation Metrics (ResNetCBAMTE):", seasonal_metrics[season])

        all_metrics = {'Yearly': metrics_all, **seasonal_metrics}

        with open('ResNetCBAMTE_val_metrics.json', 'w') as f:
            json.dump(all_metrics, f)
        print("Validation metrics saved as ResNetCBAMTE_val_metrics.json")

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
                print(f"Month {months_names[month - 1]} Validation Metrics (ResNetCBAMTE):", monthly_metrics[month])

        monthly_metrics_dict = {str(k): v for k, v in monthly_metrics.items()}

        with open('ResNetCBAMTE_val_monthly_metrics.json', 'w') as f:
            json.dump(monthly_metrics_dict, f)
        print("Monthly validation metrics saved as ResNetCBAMTE_val_monthly_metrics.json")

# 主函数
if __name__ == "__main__":
    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    batch_size = 24
    val_ds = WindDataset("./show_relevance_visualization/val.nc", H, W)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("Datasets loaded successfully")

    print("Initializing model...")
    model = WindSpeedPredictor(
        H, W,
        tigge_features=8,
        dropout_rate=0.3,
        cbam_reduction=16
    ).to(device)

    model.load_state_dict(torch.load('checkpoints/best_model_ResNetCBAM_TE.pth'))
    print("Model loaded successfully")

    print("Starting Validation...")
    validate_model(model, val_loader, device)
    print("Validation completed!")