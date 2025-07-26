import torch
import torch.nn as nn
from matplotlib import gridspec
from matplotlib.lines import Line2D
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import NearestNDInterpolator
from sklearn.preprocessing import StandardScaler
import json
import joblib
import os
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns

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


# ResNetCBAM模块（更新 dropout_rate）
class ResNetCBAM(nn.Module):
    def __init__(self, in_channels=11, dropout_rate=0.24733185479083603):
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


# ODEFunc模块（更新 hidden_dim 和 dropout_rate）
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=216, input_dim=8, dropout_rate=0.24733185479083603):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + 5, 512),
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

    def forward(self, h, x, time_features):
        input = torch.cat([h, x, time_features], dim=-1)
        return self.net(input)


# LTC模块（更新 hidden_dim 和 dropout_rate）
class LTC(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=216, output_dim=216, seq_len=4, dt=6.0,
                 dropout_rate=0.24733185479083603):
        super(LTC, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.dt = dt
        self.ode_func = ODEFunc(hidden_dim, input_dim, dropout_rate)
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

    def forward(self, x_seq, time_features_seq):
        b, t, c, H, W = x_seq.shape
        x_seq = x_seq.permute(0, 3, 4, 1, 2).reshape(b * H * W, self.seq_len, -1)
        time_features_seq = time_features_seq.unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1, 1).reshape(b * H * W,
                                                                                                      self.seq_len, 5)
        h = torch.zeros(b * H * W, self.hidden_dim).to(x_seq.device)
        dt = self.dt * 0.1
        for k in range(self.seq_len):
            x_k = x_seq[:, k, :]
            t_k = time_features_seq[:, k, :]
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


# GatedFusion模块（无变化）
class GatedFusion(nn.Module):
    def __init__(self, C1, C2):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(C1 + C2, (C1 + C2) // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d((C1 + C2) // 2, C1 + C2, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, resnet_out, ltc_out):
        fused = torch.cat([resnet_out, ltc_out], dim=1)
        gate = self.gate(fused)
        output = gate * fused
        return output


# MLP模块（更新 dropout_rate）
class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.24733185479083603):
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


# WindSpeedPredictor模块（更新参数）
class WindSpeedPredictor(nn.Module):
    def __init__(self, H, W, tigge_features=8, dropout_rate=0.24733185479083603, ltc_hidden_dim=216,
                 cbam_reduction=16):
        super(WindSpeedPredictor, self).__init__()
        self.H = H
        self.W = W
        self.resnet = ResNetCBAM(in_channels=tigge_features + 3, dropout_rate=dropout_rate)
        self.ltc = LTC(input_dim=tigge_features, hidden_dim=ltc_hidden_dim, output_dim=ltc_hidden_dim,
                       dropout_rate=dropout_rate)
        self.gated_fusion = GatedFusion(56, ltc_hidden_dim)
        self.mlp = MLP(56 + ltc_hidden_dim, dropout_rate=dropout_rate)
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

    def forward(self, tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq):
        b = tigge_spatial.size(0)
        spatial_input = torch.cat([tigge_spatial, dem_spatial], dim=1)
        time_emb = self.time_embed(time_features_t).view(b, 56, 1, 1)
        resnet_out = self.resnet(spatial_input, time_emb)
        ltc_out = self.ltc(tigge_seq, time_features_seq)
        fused = self.gated_fusion(resnet_out, ltc_out)
        pred = self.mlp(fused)
        return pred.squeeze(1)


# WindDataset类（无变化）
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


def calculate_spatial_metrics(pred, target, tigge_wind):
    H, W = pred.shape[1], pred.shape[2]
    spatial_metrics = {
        'RMSE_pred': np.zeros((H, W)),
        'RMSE_tigge': np.zeros((H, W)),
        'MAE_pred': np.zeros((H, W)),
        'MAE_tigge': np.zeros((H, W)),
        'rRMSE_pred': np.zeros((H, W)),
        'rRMSE_tigge': np.zeros((H, W)),
        'rMAE_pred': np.zeros((H, W)),
        'rMAE_tigge': np.zeros((H, W)),
        'R_pred': np.zeros((H, W)),
        'R_tigge': np.zeros((H, W)),
        'R2_pred': np.zeros((H, W)),
        'R2_tigge': np.zeros((H, W)),
        'FA_pred': np.zeros((H, W)),
        'FA_tigge': np.zeros((H, W))
    }

    # 计算每个网格点的指标
    for i in range(H):
        for j in range(W):
            pred_ij = torch.from_numpy(pred[:, i, j])
            target_ij = torch.from_numpy(target[:, i, j])
            tigge_ij = torch.from_numpy(tigge_wind[:, i, j])

            # 计算均方根误差 (RMSE)
            spatial_metrics['RMSE_pred'][i, j] = torch.sqrt(torch.mean((pred_ij - target_ij) ** 2)).item()
            spatial_metrics['RMSE_tigge'][i, j] = torch.sqrt(torch.mean((tigge_ij - target_ij) ** 2)).item()

            # 计算平均绝对误差 (MAE)
            spatial_metrics['MAE_pred'][i, j] = torch.mean(torch.abs(pred_ij - target_ij)).item()
            spatial_metrics['MAE_tigge'][i, j] = torch.mean(torch.abs(tigge_ij - target_ij)).item()

            # 计算相对均方根误差 (rRMSE)
            mean_target_ij = torch.mean(target_ij).item()
            if mean_target_ij > 0:
                spatial_metrics['rRMSE_pred'][i, j] = (spatial_metrics['RMSE_pred'][i, j] / mean_target_ij) * 100
                spatial_metrics['rRMSE_tigge'][i, j] = (spatial_metrics['RMSE_tigge'][i, j] / mean_target_ij) * 100

                # 计算相对平均绝对误差 (rMAE)
                spatial_metrics['rMAE_pred'][i, j] = (spatial_metrics['MAE_pred'][i, j] / mean_target_ij) * 100
                spatial_metrics['rMAE_tigge'][i, j] = (spatial_metrics['MAE_tigge'][i, j] / mean_target_ij) * 100

            # 计算相关系数 (R)
            if torch.std(pred_ij) > 0 and torch.std(target_ij) > 0:
                spatial_metrics['R_pred'][i, j] = torch.corrcoef(torch.stack([pred_ij, target_ij]))[0, 1].item()
            if torch.std(tigge_ij) > 0 and torch.std(target_ij) > 0:
                spatial_metrics['R_tigge'][i, j] = torch.corrcoef(torch.stack([tigge_ij, target_ij]))[0, 1].item()

            # 计算决定系数 (R²)
            ss_tot = torch.sum((target_ij - mean_target_ij) ** 2).item()
            if ss_tot > 0:
                ss_res_pred = torch.sum((target_ij - pred_ij) ** 2).item()
                ss_res_tigge = torch.sum((target_ij - tigge_ij) ** 2).item()
                spatial_metrics['R2_pred'][i, j] = 1 - (ss_res_pred / ss_tot)
                spatial_metrics['R2_tigge'][i, j] = 1 - (ss_res_tigge / ss_tot)

            # 计算准确度 (FA)
            spatial_metrics['FA_pred'][i, j] = ((torch.abs(pred_ij - target_ij) < 1).float().mean().item()) * 100
            spatial_metrics['FA_tigge'][i, j] = ((torch.abs(tigge_ij - target_ij) < 1).float().mean().item()) * 100

    return spatial_metrics


def plot_spatial_metrics(spatial_metrics, lat_range=(35.13, 47), lon_range=(103, 126.88), save_dir='spatial_metrics',
                         map_image_path=None):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 定义绘图参数
    metrics_display = {
        'RMSE': 'RMSE (m/s)',
        'MAE': 'MAE (m/s)',
        'rRMSE': 'rRMSE (%)',
        'rMAE': 'rMAE (%)',
        'R': 'R',
        'R2': 'R²',
        'FA': 'FA (%)'
    }

    # 设置颜色方案
    cmap_diff = plt.cm.RdBu_r  # 红蓝配色，蓝色
    # 设置颜色方案
    cmap_dict = {
        'RMSE': 'coolwarm',
        'MAE': 'coolwarm',
        'rRMSE': 'coolwarm',
        'rMAE': 'coolwarm',
        'R': 'viridis',
        'R2': 'viridis',
        'FA': 'YlGnBu'
    }

    # 设置值范围
    vmin_dict = {
        'RMSE': None,
        'MAE': None,
        'rRMSE': None,
        'rMAE': None,
        'R': -1,
        'R2': 0,
        'FA': 0
    }

    vmax_dict = {
        'RMSE': None,
        'MAE': None,
        'rRMSE': None,
        'rMAE': None,
        'R': 1,
        'R2': 1,
        'FA': 100
    }

    # 华北地区边界
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # 创建华北地区掩膜
    H, W = spatial_metrics['RMSE_pred'].shape
    north_china_mask = create_north_china_mask(H, W, lat_min, lat_max, lon_min, lon_max)

    # 应用掩膜到所有指标
    for metric in metrics_display.keys():
        for model in ['pred', 'tigge']:
            key = f'{metric}_{model}'
            if key in spatial_metrics:
                spatial_metrics[key] = np.where(north_china_mask, spatial_metrics[key], np.nan)

    # 计算经纬度网格
    lats = np.linspace(lat_min, lat_max, H)
    lons = np.linspace(lon_min, lon_max, W)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 获取掩膜边界
    contours = measure.find_contours(north_china_mask.reshape(H, W), 0.5)

    # 加载自定义地图图像
    custom_map = None
    # 为每个指标创建一个图
    for metric in metrics_display.keys():
        # 获取两个模型的数据
        pred_data = spatial_metrics[f'{metric}_pred']
        tigge_data = spatial_metrics[f'{metric}_tigge']

        # 确定颜色范围
        vmin = vmin_dict[metric]
        vmax = vmax_dict[metric]
        if vmin is None or vmax is None:
            combined_data = np.concatenate([pred_data[~np.isnan(pred_data)], tigge_data[~np.isnan(tigge_data)]])
            if len(combined_data) > 0:
                if vmin is None:
                    vmin = np.nanpercentile(combined_data, 1)  # 使用1%分位数避免极端值
                if vmax is None:
                    vmax = np.nanpercentile(combined_data, 99)  # 使用99%分位数避免极端值

        # 创建图形
        fig = plt.figure(figsize=(20, 10))

        # 设置标题
        fig.suptitle(f'Spatial Distribution of {metrics_display[metric]}: Proposed vs ECMWF', fontsize=16)

        # 创建子图
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # 关闭所有spine边框 - 修复双边框问题
        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        rect1 = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                              fill=False,
                              edgecolor='black',
                              linewidth=1.5,
                              zorder=10)
        ax1.add_patch(rect1)

        rect2 = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                              fill=False,
                              edgecolor='black',
                              linewidth=1.5,
                              zorder=10)
        ax2.add_patch(rect2)

        ax1.set_title(f'Proposed Model: {metrics_display[metric]}', fontsize=14)
        ax1.set_xlabel('Longitude (°E)', fontsize=12)
        ax1.set_ylabel('Latitude (°N)', fontsize=12)
        ax1.set_xlim(lon_min, lon_max)
        ax1.set_ylim(lat_min, lat_max)
        ax1.grid(False)  # 去除网格线
        ax1.set_aspect('auto')

        # 在左侧添加直方图
        divider1 = make_axes_locatable(ax1)
        hist_ax1 = divider1.append_axes("left", size="20%", pad=0.1)
        hist_data1 = pred_data[~np.isnan(pred_data)].flatten()
        if len(hist_data1) > 0:
            hist_ax1.hist(hist_data1, bins=30, orientation='horizontal', color='blue', alpha=0.7)
            hist_ax1.set_ylim(vmin, vmax)
            hist_ax1.set_xticks([])
            hist_ax1.set_yticks([])
            hist_ax1.spines['top'].set_visible(False)
            hist_ax1.spines['right'].set_visible(False)
            hist_ax1.spines['bottom'].set_visible(False)
            hist_ax1.spines['left'].set_visible(False)

        ax2.set_title(f'ECMWF: {metrics_display[metric]}', fontsize=14)
        ax2.set_xlabel('Longitude (°E)', fontsize=12)
        ax2.set_ylabel('Latitude (°N)', fontsize=12)
        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)
        ax2.grid(False)  # 去除网格线

        # 在左侧添加直方图
        divider2 = make_axes_locatable(ax2)
        hist_ax2 = divider2.append_axes("left", size="20%", pad=0.1)
        hist_data2 = tigge_data[~np.isnan(tigge_data)].flatten()
        if len(hist_data2) > 0:
            hist_ax2.hist(hist_data2, bins=30, orientation='horizontal', color='orange', alpha=0.7)
            hist_ax2.set_ylim(vmin, vmax)
            hist_ax2.set_xticks([])
            hist_ax2.set_yticks([])
            hist_ax2.spines['top'].set_visible(False)
            hist_ax2.spines['right'].set_visible(False)
            hist_ax2.spines['bottom'].set_visible(False)
            hist_ax2.spines['left'].set_visible(False)

        # 添加颜色条
        if custom_map is not None:
            # 对于contourf，直接使用返回的对象
            cbar = fig.colorbar(sc1, ax=[ax1, ax2], orientation='horizontal', pad=0.05, aspect=40)
        else:
            # 对于scatter，直接使用返回的对象
            cbar = fig.colorbar(sc1, ax=[ax1, ax2], orientation='horizontal', pad=0.05, aspect=40)
        cbar.set_label(metrics_display[metric], fontsize=12)

        # 保存图形
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3, hspace=0.3)
        plt.savefig(os.path.join(save_dir, f'test_spatial_{metric}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存 {metric} 空间分布图")


def plot_hourly_scatter_comparison(all_preds, all_targets, all_tigge_wind, all_dates, save_dir='hourly_metrics'):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 转换日期为pandas datetime对象
    dates = pd.to_datetime(all_dates)

    # 提取小时信息
    hours = dates.hour

    # 时间点名称和图注标签
    hour_names = {
        0: '00:00',
        6: '06:00',
        12: '12:00',
        18: '18:00'
    }

    hour_labels = {
        0: '(a) 00:00 UTC',
        6: '(b) 06:00 UTC',
        12: '(c) 12:00 UTC',
        18: '(d) 18:00 UTC'
    }

    # 设置全局字体属性 - 使用Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    font_size = 14

    # 创建2x2的大图
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    # 为每个时间点创建散点图
    for i, hour in enumerate([0, 6, 12, 18]):
        hour_mask = (hours == hour)

        # 跳过没有数据的时间点
        if not np.any(hour_mask):
            print(f"警告: 没有 {hour_names[hour]} 的数据")
            axes[i].text(0.5, 0.5, f"No data for {hour_names[hour]}",
                         ha='center', va='center', fontsize=font_size, fontweight='bold')
            continue

        # 获取该时间点的数据
        hour_preds = all_preds[hour_mask].flatten()
        hour_targets = all_targets[hour_mask].flatten()
        hour_tigge = all_tigge_wind[hour_mask].flatten()

        # 数据采样 - 大幅减少点的数量以提高可视性
        np.random.seed(42 + i)  # 为每个子图使用不同的随机种子

        sample_size_tigge = min(1500, len(hour_targets))
        sample_size_pred = min(2500, len(hour_targets))

        # 分别采样ECMWF和模型预测点
        sample_indices_tigge = np.random.choice(len(hour_targets), sample_size_tigge, replace=False)
        sample_indices_pred = np.random.choice(len(hour_targets), sample_size_pred, replace=False)

        # 获取采样后的数据
        hour_targets_tigge = hour_targets[sample_indices_tigge]
        hour_tigge_sampled = hour_tigge[sample_indices_tigge]

        hour_targets_pred = hour_targets[sample_indices_pred]
        hour_preds_sampled = hour_preds[sample_indices_pred]

        # 计算该时间点的指标 (使用全部数据计算指标，而不是采样后的数据)
        metrics = calculate_metrics(
            torch.from_numpy(hour_preds),
            torch.from_numpy(hour_targets),
            torch.from_numpy(hour_tigge)
        )

        # 设置当前子图
        ax = axes[i]

        # 设置图形边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 设置点的透明度和大小
        point_alpha = 0.5
        point_size = 8

        # 设置线的粗细 - 两种线使用相同的虚线和粗度
        line_width = 2.5

        ax.scatter(hour_targets_tigge, hour_tigge_sampled, alpha=point_alpha, s=point_size, c='blue',
                   label='ECMWF-TIGGE')

        # 绘制模型预测散点图 - 红色
        ax.scatter(hour_targets_pred, hour_preds_sampled, alpha=point_alpha, s=point_size, c='red', label='MTRCL')

        # 添加对角线 - 使用实线
        max_val = max(np.max(hour_targets), np.max(hour_preds), np.max(hour_tigge))
        min_val = min(np.min(hour_targets), np.min(hour_preds), np.min(hour_tigge))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5)

        z_pred = np.sum(hour_targets * hour_preds) / np.sum(hour_targets * hour_targets)
        p_pred = lambda x: z_pred * x

        z_tigge = np.sum(hour_targets * hour_tigge) / np.sum(hour_targets * hour_targets)
        p_tigge = lambda x: z_tigge * x

        # 修改：两种拟合线都使用相同的虚线和粗度
        ax.plot(np.sort(hour_targets_pred), p_pred(np.sort(hour_targets_pred)), "r--", linewidth=line_width)
        ax.plot(np.sort(hour_targets_tigge), p_tigge(np.sort(hour_targets_tigge)), "b--", linewidth=line_width)

        # 修改：交换指标位置 - MTRCL指标放在右下角
        metrics_text_pred = f"MTRCL: y={z_pred:.2f}x"
        metrics_text_pred_2 = f"N={len(hour_targets)}"
        metrics_text_pred_3 = f"RMSE={metrics['RMSE_pred']:.3f}"
        metrics_text_pred_4 = f"MAE={metrics['MAE_pred']:.3f}"
        metrics_text_pred_5 = f"FA={metrics['FA_pred']:.2f}%"
        metrics_text_pred_6 = f"R={metrics['R_pred']:.3f}"

        # 放置文本 - 右下角，红色，字体更大
        ax.text(0.95, 0.30, metrics_text_pred, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.25, metrics_text_pred_2, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.20, metrics_text_pred_3, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.15, metrics_text_pred_4, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.10, metrics_text_pred_5, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.05, metrics_text_pred_6, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')

        metrics_text_tigge = f"ECMWF-TIGGE: y={z_tigge:.2f}x"
        metrics_text_tigge_2 = f"N={len(hour_targets)}"  # 添加N值
        metrics_text_tigge_3 = f"RMSE={metrics['RMSE_tigge']:.3f}"
        metrics_text_tigge_4 = f"MAE={metrics['MAE_tigge']:.3f}"
        metrics_text_tigge_5 = f"FA={metrics['FA_tigge']:.2f}%"
        metrics_text_tigge_6 = f"R={metrics['R_tigge']:.3f}"

        # 放置文本 - 左上角，蓝色，字体更大
        ax.text(0.05, 0.95, metrics_text_tigge, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.90, metrics_text_tigge_2, transform=ax.transAxes,  # 添加N值
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.85, metrics_text_tigge_3, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.80, metrics_text_tigge_4, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.75, metrics_text_tigge_5, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.70, metrics_text_tigge_6, transform=ax.transAxes,
                fontsize=font_size, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')

        # 设置轴标签
        ax.set_xlabel('Observed wind speed (m/s)', fontsize=font_size + 2, fontweight='bold')
        ax.set_ylabel('Estimated wind speed (m/s)', fontsize=font_size + 2, fontweight='bold')

        # 修改：将图注放回左上方，稍微向下移动一点
        ax.text(0.01, 0.99, hour_labels[hour], transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left')

        # 去除网格线
        ax.grid(False)

        # 修改：将图例移到右上方
        legend = ax.legend(loc='upper right', fontsize=font_size + 2, frameon=False)

        # 单独设置图例文本样式
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # 设置轴范围
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        # 设置刻度字体
        ax.tick_params(axis='both', which='major', width=1.5, labelsize=font_size)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    # 调整布局
    plt.tight_layout()

    # 保存组合图
    plt.savefig(os.path.join(save_dir, 'test_scatter_plots_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("已保存组合风速散点图")

    # 单独保存每个时间点的图
    for i, hour in enumerate([0, 6, 12, 18]):
        hour_mask = (hours == hour)

        # 跳过没有数据的时间点
        if not np.any(hour_mask):
            continue

        # 创建单独的图
        fig, ax = plt.subplots(figsize=(10, 10))

        # 设置图形边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 获取该时间点的数据
        hour_preds = all_preds[hour_mask].flatten()
        hour_targets = all_targets[hour_mask].flatten()
        hour_tigge = all_tigge_wind[hour_mask].flatten()

        # 数据采样
        np.random.seed(42)
        sample_size_tigge = min(1500, len(hour_targets))
        sample_size_pred = min(2500, len(hour_targets))

        sample_indices_tigge = np.random.choice(len(hour_targets), sample_size_tigge, replace=False)
        sample_indices_pred = np.random.choice(len(hour_targets), sample_size_pred, replace=False)

        hour_targets_tigge = hour_targets[sample_indices_tigge]
        hour_tigge_sampled = hour_tigge[sample_indices_tigge]

        hour_targets_pred = hour_targets[sample_indices_pred]
        hour_preds_sampled = hour_preds[sample_indices_pred]

        # 计算指标
        metrics = calculate_metrics(
            torch.from_numpy(hour_preds),
            torch.from_numpy(hour_targets),
            torch.from_numpy(hour_tigge)
        )

        # 绘制散点图
        ax.scatter(hour_targets_tigge, hour_tigge_sampled, alpha=0.35, s=12, c='blue', label='ECMWF-TIGGE')
        ax.scatter(hour_targets_pred, hour_preds_sampled, alpha=0.35, s=12, c='red', label='MTRCL')

        # 添加对角线
        max_val = max(np.max(hour_targets), np.max(hour_preds), np.max(hour_tigge))
        min_val = min(np.min(hour_targets), np.min(hour_preds), np.min(hour_tigge))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5)

        # 添加拟合线
        z_pred = np.sum(hour_targets * hour_preds) / np.sum(hour_targets * hour_targets)
        p_pred = lambda x: z_pred * x

        z_tigge = np.sum(hour_targets * hour_tigge) / np.sum(hour_targets * hour_targets)
        p_tigge = lambda x: z_tigge * x

        ax.plot(np.sort(hour_targets_pred), p_pred(np.sort(hour_targets_pred)), "r--", linewidth=2.5)
        ax.plot(np.sort(hour_targets_tigge), p_tigge(np.sort(hour_targets_tigge)), "b--", linewidth=2.5)

        # 添加指标文本 - MTRCL在右下角
        metrics_text_pred = f"MTRCL: y={z_pred:.2f}x"
        metrics_text_pred_2 = f"N={len(hour_targets)}"
        metrics_text_pred_3 = f"RMSE={metrics['RMSE_pred']:.3f}"
        metrics_text_pred_4 = f"MAE={metrics['MAE_pred']:.3f}"
        metrics_text_pred_5 = f"FA={metrics['FA_pred']:.2f}%"
        metrics_text_pred_6 = f"R={metrics['R_pred']:.3f}"

        ax.text(0.95, 0.30, metrics_text_pred, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.25, metrics_text_pred_2, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.20, metrics_text_pred_3, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.15, metrics_text_pred_4, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.10, metrics_text_pred_5, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')
        ax.text(0.95, 0.05, metrics_text_pred_6, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                color='red')

        # 修改：ECMWF-TIGGE指标放在左上角，添加N值
        metrics_text_tigge = f"ECMWF-TIGGE: y={z_tigge:.2f}x"
        metrics_text_tigge_2 = f"N={len(hour_targets)}"  # 添加N值
        metrics_text_tigge_3 = f"RMSE={metrics['RMSE_tigge']:.3f}"
        metrics_text_tigge_4 = f"MAE={metrics['MAE_tigge']:.3f}"
        metrics_text_tigge_5 = f"FA={metrics['FA_tigge']:.2f}%"
        metrics_text_tigge_6 = f"R={metrics['R_tigge']:.3f}"

        # 放置文本 - 左上角，蓝色，字体更大
        ax.text(0.05, 0.95, metrics_text_tigge, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.90, metrics_text_tigge_2, transform=ax.transAxes,  # 添加N值
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.85, metrics_text_tigge_3, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.80, metrics_text_tigge_4, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.75, metrics_text_tigge_5, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')
        ax.text(0.05, 0.70, metrics_text_tigge_6, transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                color='blue')

        # 设置轴标签
        ax.set_xlabel('Observed wind speed (m/s)', fontsize=font_size + 2, fontweight='bold')
        ax.set_ylabel('Estimated wind speed (m/s)', fontsize=font_size + 2, fontweight='bold')

        # 修改：将图注放回左上方，稍微向下移动一点
        ax.text(0.01, 0.99, hour_labels[hour], transform=ax.transAxes,
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left')

        # 去除网格线
        ax.grid(False)

        # 修改：将图例移到右上方
        legend = ax.legend(loc='upper right', fontsize=font_size + 2, frameon=False)
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # 设置轴范围
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        # 设置刻度字体
        ax.tick_params(axis='both', which='major', width=1.5, labelsize=font_size)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # 保存单独图片
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'test_scatter_plot_{hour:02d}00.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存 {hour_names[hour]} 时间点的风速散点图")


def plot_china_wind_distribution(preds, targets, tigge_wind, save_dir='china_wind_distribution'):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 计算平均风速分布
    mean_pred = np.mean(preds, axis=0)
    mean_target = np.mean(targets, axis=0)
    mean_tigge = np.mean(tigge_wind, axis=0)

    # 华北地区边界 - 确保使用正确的经纬度范围
    lat_min, lat_max = 35.13, 47.0
    lon_min, lon_max = 103.0, 126.88  

    # 创建经纬度网格
    H, W = mean_pred.shape

    # 格点间距
    lon_step = (lon_max - lon_min) / (W - 1)
    lat_step = (lat_max - lat_min) / (H - 1)

    # 创建格点中心坐标
    lons = np.linspace(lon_min, lon_max, W)
    lats = np.linspace(lat_min, lat_max, H)

    # 创建华北地区掩膜
    north_china_mask = create_north_china_mask(H, W, lat_min, lat_max, lon_min, lon_max)

    # 应用掩膜
    mean_pred = np.where(north_china_mask, mean_pred, np.nan)
    mean_target = np.where(north_china_mask, mean_target, np.nan)
    mean_tigge = np.where(north_china_mask, mean_tigge, np.nan)

    # 设置全局字体属性
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    font_size = 16  # 与空间指标分布图一致的字体大小

    # 加载自定义地图图像
    try:
    
        has_map = True

        # 获取地图图像的宽高比
        map_height, map_width = custom_map.shape[:2]
        map_aspect_ratio = map_width / map_height
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 地图宽高比: {map_aspect_ratio}")

        # 处理地图 - 创建一个用于叠加的线条图
        # 首先确保地图数据在[0,1]范围内
        if custom_map.max() > 1.0:
            custom_map = custom_map / 255.0

        # 创建线条掩码 - 检测非白色区域
        # 白色像素RGB值接近(1,1,1)
        is_white = np.all(custom_map > 0.9, axis=2) if custom_map.shape[2] >= 3 else np.ones(custom_map.shape[:2],
                                                                                             dtype=bool)
        is_line = ~is_white

        # 创建一个只包含线条的RGBA图像
        lines_map = np.ones((*custom_map.shape[:2], 4))  # 初始化为全白色且完全透明

        # 设置线条颜色和透明度 - 更深的线条
        for i in range(min(3, custom_map.shape[2])):  # 复制RGB通道
            # 只复制线条区域的颜色，非线条区域保持白色
            lines_map[..., i] = np.where(is_line, 0.1, 1.0)  # 使用更深的黑色

        # 设置透明度通道 - 只有线条区域不透明
        lines_map[..., 3] = np.where(is_line, 1.0, 0.0)  # 完全不透明的线条

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]  {str(e)}")
        has_map = False
        map_aspect_ratio = (lon_max - lon_min) / (lat_max - lat_min)  # 默认宽高比

    # 创建边界网格 - 关键修改：将格点网格扩展至精确匹配边界
    lons_edges = np.linspace(lon_min, lon_max, W + 1)
    lats_edges = np.linspace(lat_min, lat_max, H + 1)

    # 数据处理 - 填充NaN值
    # 创建有效数据的掩码
    valid_mask_pred = ~np.isnan(mean_pred)
    valid_mask_target = ~np.isnan(mean_target)
    valid_mask_tigge = ~np.isnan(mean_tigge)

    # 创建用于插值的索引网格
    y_indices, x_indices = np.mgrid[0:H, 0:W]

    # 对预测数据进行处理
    valid_points_pred = np.column_stack((y_indices[valid_mask_pred], x_indices[valid_mask_pred]))
    valid_values_pred = mean_pred[valid_mask_pred]

    # 为预测数据创建插值函数
    from scipy.interpolate import NearestNDInterpolator
    interp_pred = NearestNDInterpolator(valid_points_pred, valid_values_pred)

    # 生成待插值的所有点
    all_points = np.column_stack((y_indices.ravel(), x_indices.ravel()))

    # 生成填充后的预测数据
    mean_pred_filled = np.copy(mean_pred)
    nan_mask_pred = np.isnan(mean_pred)
    if np.any(nan_mask_pred):  # 如果存在NaN值
        mean_pred_filled[nan_mask_pred] = interp_pred(all_points[nan_mask_pred.ravel()]).reshape(-1)

    # 对观测数据同样处理
    valid_points_target = np.column_stack((y_indices[valid_mask_target], x_indices[valid_mask_target]))
    valid_values_target = mean_target[valid_mask_target]

    interp_target = NearestNDInterpolator(valid_points_target, valid_values_target)

    mean_target_filled = np.copy(mean_target)
    nan_mask_target = np.isnan(mean_target)
    if np.any(nan_mask_target):  # 如果存在NaN值
        mean_target_filled[nan_mask_target] = interp_target(all_points[nan_mask_target.ravel()]).reshape(-1)

    # 对ECMWF数据同样处理
    valid_points_tigge = np.column_stack((y_indices[valid_mask_tigge], x_indices[valid_mask_tigge]))
    valid_values_tigge = mean_tigge[valid_mask_tigge]

    interp_tigge = NearestNDInterpolator(valid_points_tigge, valid_values_tigge)

    mean_tigge_filled = np.copy(mean_tigge)
    nan_mask_tigge = np.isnan(mean_tigge)
    if np.any(nan_mask_tigge):  # 如果存在NaN值
        mean_tigge_filled[nan_mask_tigge] = interp_tigge(all_points[nan_mask_tigge.ravel()]).reshape(-1)

    # 设置颜色范围为0-6
    vmin = 0
    vmax = 6

    # 确保所有数据都在有效范围内
    mean_pred_filled = np.clip(mean_pred_filled, vmin, vmax)
    mean_target_filled = np.clip(mean_target_filled, vmin, vmax)
    mean_tigge_filled = np.clip(mean_tigge_filled, vmin, vmax)

    # 根据地图宽高比计算图形大小
    main_height = 5
    main_width = main_height * map_aspect_ratio

    hist_width_ratio = 0.2
    colorbar_width_ratio = 0.02
    spacing_ratio = 0.05  # 图形之间的间距比例

    # 计算总宽度（两个主图 + 两个直方图 + 间距 + 颜色条）
    total_width = 2 * main_width * (
            1 + hist_width_ratio) + main_width * spacing_ratio + main_width * colorbar_width_ratio

    # 总高度：两行主图 + 行间距
    total_height = 2 * main_height + main_height * 0.3  # 行间距

    # 创建图形，设置总宽度和高度
    fig = plt.figure(figsize=(total_width, total_height))


    # 基础尺寸参数
    hist_width = main_width * hist_width_ratio / total_width  # 直方图宽度比例
    main_img_width = main_width / total_width  # 主图宽度比例
    spacing_width = main_width * spacing_ratio / total_width  # 图间距宽度比例
    cbar_width = main_width * colorbar_width_ratio * 2 / total_width  # 颜色条宽度比例

    # 第一行位置参数 (可调整)
    row1_bottom = 0.55  # 第一行底部位置 (0-1, 增加→上移, 减少→下移)
    row1_height = 0.4  # 第一行高度 (0-1, 增加→变高, 减少→变矮)

    # 第二行位置参数 (可调整)
    row2_bottom = 0.05  # 第二行底部位置 (0-1, 增加→上移, 减少→下移)
    row2_height = 0.4  # 第二行高度 (0-1, 增加→变高, 减少→变矮)

    # 第一行左侧图位置参数 (可调整)
    left_start_x = 0.05  # 左侧图起始x位置 (0-1, 增加→右移, 减少→左移)
    hist_main_gap1 = 0.03  # 第一行直方图与主图间距 (增加→间距变大, 减少→间距变小)

    # 第一行右侧图位置参数 (可调整)
    middle_gap = 0.03  # 左右两组图之间的间距 (增加→间距变大, 减少→间距变小)
    hist_main_gap2 = 0.03  # 第二个直方图与主图间距 (增加→间距变大, 减少→间距变小)
    main_cbar_gap = 0.01  # 主图与颜色条间距 (增加→间距变大, 减少→间距变小)

    # 第二行居中图位置参数 (可调整)
    left_start_x = 0.05  # 控制(a)图和(c)图的左边界位置
    hist_main_gap3 = 0.03  # 控制(c)图直方图与主图间距
    main_cbar_gap = 0.01  # 控制(c)图主图与颜色条间距
    # ================= 实际位置计算 =================

    # 第一行左侧图
    hist_ax1 = fig.add_axes([left_start_x, row1_bottom, hist_width, row1_height])
    ax1 = fig.add_axes([left_start_x + hist_width + hist_main_gap1, row1_bottom, main_img_width, row1_height])

    # 第一行右侧图
    right_start = left_start_x + hist_width + hist_main_gap1 + main_img_width + middle_gap
    hist_ax2 = fig.add_axes([right_start, row1_bottom, hist_width, row1_height])
    ax2 = fig.add_axes([right_start + hist_width + hist_main_gap2, row1_bottom, main_img_width, row1_height])

    # 第一行颜色条
    cbar_ax = fig.add_axes([right_start + hist_width + hist_main_gap2 + main_img_width + main_cbar_gap,
                            row1_bottom, cbar_width, row1_height])

    # 第二行中间图（居中）
    # total_second_row_width = hist_width + hist_main_gap3 + main_img_width + main_cbar_gap + cbar_width
    # center_x = (1.0 - total_second_row_width) / 2 + row2_center_offset
    center_x = left_start_x

    hist_ax3 = fig.add_axes([center_x, row2_bottom, hist_width, row2_height])
    ax3 = fig.add_axes([center_x + hist_width + hist_main_gap3, row2_bottom, main_img_width, row2_height])

    # 第二行颜色条 (新增)
    cbar_ax2 = fig.add_axes([center_x + hist_width + hist_main_gap3 + main_img_width + main_cbar_gap,
                             row2_bottom, cbar_width, row2_height])

    # 自定义坐标刻度 - 确保使用精确的值
    lon_ticks = [103, 110, 115, 120, 126]
    lat_ticks = np.arange(36, 48, 3)  # [36, 39, 42, 45]

    # 创建cmap - 使用RdYlBu颜色映射
    cmap = plt.cm.RdYlBu_r  # 使用RdYlBu_r获得更鲜明的颜色对比

    # 绘制第一个图 - 预测风速 (MTRCL)
    _draw_wind_plot(ax1, hist_ax1, mean_pred_filled, mean_pred, lons_edges, lats_edges,
                    lon_min, lon_max, lat_min, lat_max, lines_map, has_map, cmap, vmin, vmax,
                    "(a) MTRCL", lon_ticks, lat_ticks, font_size, map_aspect_ratio)

    # 绘制第二个图 - ECMWF风速
    im2 = _draw_wind_plot(ax2, hist_ax2, mean_tigge_filled, mean_tigge, lons_edges, lats_edges,
                          lon_min, lon_max, lat_min, lat_max, lines_map, has_map, cmap, vmin, vmax,
                          "(b) ECMWF-TIGGE", lon_ticks, lat_ticks, font_size, map_aspect_ratio)

    # 绘制第三个图 - 观测风速
    im3 = _draw_wind_plot(ax3, hist_ax3, mean_target_filled, mean_target, lons_edges, lats_edges,
                          lon_min, lon_max, lat_min, lat_max, lines_map, has_map, cmap, vmin, vmax,
                          "(c) ERA5_wind_speed", lon_ticks, lat_ticks, font_size, map_aspect_ratio)

    # 添加第一行颜色条
    cbar = plt.colorbar(im2, cax=cbar_ax, extend='neither')  # 使用neither参数去掉尖尖
    cbar.set_label('Wind Speed (m/s)', fontsize=font_size+7, fontweight='bold')

    # 设置第一行颜色条刻度标签字体
    cbar.ax.tick_params(labelsize=font_size+5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(font_size+5)

    # 增加刻度线宽度
    cbar.ax.tick_params(width=1.5)

    # 颜色条轮廓线宽度
    cbar.outline.set_linewidth(1.5)

    # 添加第二行颜色条 (新增)
    cbar2 = plt.colorbar(im3, cax=cbar_ax2, extend='neither')
    cbar2.set_label('Wind Speed (m/s)', fontsize=font_size+7, fontweight='bold')

    # 设置第二行颜色条刻度标签字体
    cbar2.ax.tick_params(labelsize=font_size+5)
    for label in cbar2.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(font_size+5)

    cbar2.ax.tick_params(width=1.5)
    cbar2.outline.set_linewidth(1.5)

    # 保存图形
    plt.savefig(os.path.join(save_dir, 'china_wind_distribution_3plots.png'),
                dpi=1000,
                bbox_inches='tight',
                pad_inches=0.05)
    plt.close()

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 三图中国风速分布图已保存至: {save_dir}/china_wind_distribution_3plots.png")


def _draw_wind_plot(ax, hist_ax, data_filled, data_original, lons_edges, lats_edges,
                    lon_min, lon_max, lat_min, lat_max, lines_map, has_map, cmap, vmin, vmax,
                    title, lon_ticks, lat_ticks, font_size, map_aspect_ratio):
    """
    绘制单个风速分布图的辅助函数
    """
    # 设置直方图样式
    for spine in hist_ax.spines.values():
        spine.set_linewidth(1.5)

    # 绘制直方图
    hist_data = data_original[~np.isnan(data_original)].flatten()
    if len(hist_data) > 0:
        hist, bins = np.histogram(hist_data, bins=np.linspace(vmin, vmax, 11))
        width = 0.85 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        hist_ax.barh(center, hist, height=width, color='dimgray', alpha=0.8)
        hist_ax.set_ylim(vmin, vmax)

        hist_ax.set_xlabel('Number of samples', fontsize=font_size+7, fontweight='bold')
        hist_ax.set_ylabel('Wind speed (m/s)', fontsize=font_size+7, fontweight='bold')

        # 设置x轴刻度，让数值更简洁（类似于(a)和(b)图）
        max_samples = np.max(hist)
        if max_samples > 100:
            # 设置简洁的x轴刻度
            if max_samples <= 1000:
                x_ticks = [0, 200, 400, 600, 800, 1000]
            elif max_samples <= 2000:
                x_ticks = [0, 500, 1000, 1500, 2000]
            else:
                # 动态设置刻度
                step = int(max_samples // 4 / 100) * 100  # 取整到百位
                x_ticks = list(range(0, int(max_samples) + step, step))

            # 过滤掉超出范围的刻度
            x_ticks = [x for x in x_ticks if x <= max_samples * 1.1]
            hist_ax.set_xticks(x_ticks)

            # 添加科学计数法标注
            hist_ax.set_title('1e2', fontsize=font_size +7, loc='right', fontweight='bold')

        for label in hist_ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(font_size+5)
        for label in hist_ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(font_size+5)

        hist_ax.tick_params(width=1.5)

    # 设置主图框架
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    ax.axis('on')

    # 绘制数据层
    im = ax.pcolormesh(lons_edges, lats_edges, data_filled,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       alpha=0.75, zorder=2, shading='flat')

    # 叠加地图线条
    if has_map:
        ax.imshow(lines_map, extent=[lon_min, lon_max, lat_min, lat_max],
                  aspect='auto', origin='upper', zorder=3, interpolation='nearest')

    # 设置标题
    ax.set_title(title, fontsize=font_size + 10, fontweight='bold',
                 loc='left', pad=8)

    # 设置刻度
    ax.set_xticks(lon_ticks)
    lon_labels = [f"{int(x)}°E" for x in lon_ticks]
    ax.set_xticklabels(lon_labels)

    ax.set_yticks(lat_ticks)
    lat_labels = [f"{int(y)}°N" for y in lat_ticks]
    ax.set_yticklabels(lat_labels)

    # 设置刻度标签字体
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(font_size+5)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(font_size+5)

    ax.tick_params(width=1.5)

    # 设置坐标轴范围
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(False)


    return im


def plot_combined_time_analysis(all_preds, all_targets, all_tigge_wind, all_dates, save_dir='combined_analysis'):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 转换日期为pandas datetime对象
    dates = pd.to_datetime(all_dates)

    # 提取小时信息
    hours = dates.hour

    # 时间点名称和图注标签
    hour_values = [0, 6, 12, 18]
    hour_labels = {
        0: '(a) 00:00 UTC',
        6: '(c) 06:00 UTC',
        12: '(e) 12:00 UTC',
        18: '(g) 18:00 UTC'
    }

    scatter_labels = {
        0: '(b) 00:00 UTC',
        6: '(d) 06:00 UTC',
        12: '(f) 12:00 UTC',
        18: '(h) 18:00 UTC'
    }

    # 设置全局字体属性 - 使用Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    # 调小散点图中的指标文字的字体大小，从16改为14
    font_size = 16
    metrics_font_size = 14  # 指标文字的字体大小

    # 新的配色方案
    colors = {
        'JRA55': 'black',  # 黑色 - 观测值
        'MTRCL': 'red',  # 红色 - 模型预测
        'ECMWF': 'blue'  # 蓝色 - ECMWF预测
    }

    # 创建4行2列的大图，调整布局
    fig = plt.figure(figsize=(20, 18))


    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[1.8, 1])


    # 为每个时间点创建一行（折线图+散点图）
    for row, hour in enumerate(hour_values):
        # ================ 左侧：折线图 ================
        ax_line = fig.add_subplot(gs[row, 0])

        # 获取该时间点的折线图数据
        hour_mask = (dates.hour == hour)
        hour_dates = dates[hour_mask]
        daily_preds = all_preds[hour_mask].mean(axis=(1, 2))
        daily_targets = all_targets[hour_mask].mean(axis=(1, 2))
        daily_tigge = all_tigge_wind[hour_mask].mean(axis=(1, 2))

        days_of_year = hour_dates.dayofyear.values

        # 设置图形边框线宽
        for spine in ax_line.spines.values():
            spine.set_linewidth(1.5)

        # 使用指定的颜色绘制折线
        ax_line.plot(days_of_year, daily_targets, label='ERA5_wind_speed',
                     color=colors['JRA55'], linewidth=1.5, zorder=3)
        ax_line.plot(days_of_year, daily_preds, label='MTRCL',
                     color=colors['MTRCL'], linewidth=1.5, zorder=2)
        ax_line.plot(days_of_year, daily_tigge, label='ECMWF-TIGGE',
                     color=colors['ECMWF'], linewidth=1.5, zorder=1)

        # 设置x轴标签（只在最底部的折线图显示）
        if row == 3:
            ax_line.set_xlabel('Day of year(2024)', fontsize=font_size + 4, fontweight='bold')

        # 设置y轴标签
        ax_line.set_ylabel('Wind speed (m/s)', fontsize=font_size + 4, fontweight='bold')

        # 设置x轴刻度
        ax_line.set_xlim(0, 370)
        ax_line.set_xticks(np.arange(0, 371, 50))

        # 设置刻度字体
        ax_line.tick_params(axis='both', which='major', width=1.5, labelsize=font_size)
        for label in ax_line.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax_line.get_yticklabels():
            label.set_fontweight('bold')

        # 只在第一个折线图显示图例，保持原有位置
        if row == 0:
            # 保持图例在折线图内部的右上角
            legend = ax_line.legend(loc='upper right', bbox_to_anchor=(1.0, 1.02),
                                    ncol=3, fontsize=16, frameon=False)

            for text in legend.get_texts():
                text.set_fontweight('bold')

            # 图注放在左上角，保持原有位置
            ax_line.text(0.01, 0.96, hour_labels[hour], transform=ax_line.transAxes,
                         fontsize=font_size + 2, fontweight='bold', va='top', ha='left')
        else:
            # 其他行只添加图注
            ax_line.text(0.01, 0.96, hour_labels[hour], transform=ax_line.transAxes,
                         fontsize=font_size + 2, fontweight='bold', va='top', ha='left')

        # ================ 右侧：散点图 ================
        ax_scatter = fig.add_subplot(gs[row, 1])

        # 获取该时间点的散点图数据
        hour_mask = (hours == hour)
        hour_preds = all_preds[hour_mask].flatten()
        hour_targets = all_targets[hour_mask].flatten()
        hour_tigge = all_tigge_wind[hour_mask].flatten()

        # 数据采样 - 减少点的数量以提高可视性
        np.random.seed(42 + row)
        # 修改：大幅减少采样点数量，解决重叠问题
        sample_size_tigge = min(300, len(hour_targets))
        sample_size_pred = min(500, len(hour_targets))

        # 修改：使用分层采样，确保各个区域的点都能被采样到
        # 根据目标值大小将数据分成几个区间
        bins = np.linspace(np.min(hour_targets), np.max(hour_targets), 10)
        digitized = np.digitize(hour_targets, bins)

        # 在每个区间内采样
        sample_indices_tigge = []
        sample_indices_pred = []

        for i in range(1, len(bins) + 1):
            bin_indices = np.where(digitized == i)[0]
            if len(bin_indices) > 0:
                # 计算每个区间应采样的点数
                bin_size_tigge = int(sample_size_tigge * len(bin_indices) / len(hour_targets))
                bin_size_pred = int(sample_size_pred * len(bin_indices) / len(hour_targets))

                # 确保每个区间至少有一些点
                bin_size_tigge = max(bin_size_tigge, min(5, len(bin_indices)))
                bin_size_pred = max(bin_size_pred, min(5, len(bin_indices)))

                # 在区间内随机采样
                if len(bin_indices) > bin_size_tigge:
                    tigge_indices = np.random.choice(bin_indices, bin_size_tigge, replace=False)
                    sample_indices_tigge.extend(tigge_indices)
                else:
                    sample_indices_tigge.extend(bin_indices)

                if len(bin_indices) > bin_size_pred:
                    pred_indices = np.random.choice(bin_indices, bin_size_pred, replace=False)
                    sample_indices_pred.extend(pred_indices)
                else:
                    sample_indices_pred.extend(bin_indices)

        # 转换为numpy数组
        sample_indices_tigge = np.array(sample_indices_tigge)
        sample_indices_pred = np.array(sample_indices_pred)

        hour_targets_tigge = hour_targets[sample_indices_tigge]
        hour_tigge_sampled = hour_tigge[sample_indices_tigge]

        hour_targets_pred = hour_targets[sample_indices_pred]
        hour_preds_sampled = hour_preds[sample_indices_pred]

        # 计算该时间点的指标
        metrics = calculate_metrics(
            torch.from_numpy(hour_preds),
            torch.from_numpy(hour_targets),
            torch.from_numpy(hour_tigge)
        )

        # 设置图形边框线宽
        for spine in ax_scatter.spines.values():
            spine.set_linewidth(1.5)

        # 调整点的透明度和大小
        point_alpha = 0.8
        point_size = 6

        # 设置线的粗细
        line_width = 2.0

        # 绘制散点图，但不显示图例
        ax_scatter.scatter(hour_targets_tigge, hour_tigge_sampled, alpha=point_alpha, s=point_size, c='blue')
        ax_scatter.scatter(hour_targets_pred, hour_preds_sampled, alpha=point_alpha, s=point_size, c='red')

        # 添加对角线
        max_val = max(np.max(hour_targets), np.max(hour_preds), np.max(hour_tigge))
        min_val = min(np.min(hour_targets), np.min(hour_preds), np.min(hour_tigge))
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5)

        # 添加线性拟合 - 强制通过原点
        z_pred = np.sum(hour_targets * hour_preds) / np.sum(hour_targets * hour_targets)
        p_pred = lambda x: z_pred * x

        z_tigge = np.sum(hour_targets * hour_tigge) / np.sum(hour_targets * hour_targets)
        p_tigge = lambda x: z_tigge * x

        ax_scatter.plot(np.sort(hour_targets_pred), p_pred(np.sort(hour_targets_pred)), "r--", linewidth=line_width)
        ax_scatter.plot(np.sort(hour_targets_tigge), p_tigge(np.sort(hour_targets_tigge)), "b--", linewidth=line_width)

        spacing = 0.06

        # 设置轴标签（只在最底部的散点图显示x轴标签）
        if row == 3:
            ax_scatter.set_xlabel('Observed wind speed (m/s)', fontsize=font_size + 3, fontweight='bold')
        ax_scatter.set_ylabel('Estimated wind speed (m/s)', fontsize=font_size + 3, fontweight='bold')

        # 添加图注 - 与左侧折线图的图注保持相同的垂直位置
        ax_scatter.text(0.01, 0.96, scatter_labels[hour], transform=ax_scatter.transAxes,
                        fontsize=font_size + 2, fontweight='bold', verticalalignment='top', horizontalalignment='left')

        # 去除网格线
        ax_scatter.grid(False)

        # 设置轴范围
        max_val_for_axis = max(max_val, 12)  # 确保最大值至少为12，以便能有足够的刻度
        ax_scatter.set_xlim(0, max_val_for_axis)
        ax_scatter.set_ylim(0, max_val_for_axis)

        # 修改: 设置相同的刻度间隔（2为间隔）
        ticks = np.arange(0, max_val_for_axis + 1, 2)
        ax_scatter.set_xticks(ticks)
        ax_scatter.set_yticks(ticks)

        # 修改: 只在原点显示0，其他刻度保持数值但不显示0
        x_labels = [str(int(tick)) if tick > 0 else '0' for tick in ticks]
        y_labels = [str(int(tick)) if tick > 0 else '0' for tick in ticks]

        ax_scatter.set_xticklabels(x_labels)
        ax_scatter.set_yticklabels(y_labels)

        # 设置刻度字体
        ax_scatter.tick_params(axis='both', which='major', width=1.5, labelsize=font_size)
        for label in ax_scatter.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax_scatter.get_yticklabels():
            label.set_fontweight('bold')

        # 确保散点图是正方形
        ax_scatter.set_aspect('equal')

    # 在所有子图创建完成后，手动调整它们的位置
    for row in range(4):
        # 获取当前行的折线图和散点图
        ax_line = fig.axes[row * 2]
        ax_scatter = fig.axes[row * 2 + 1]

        # 获取当前位置
        line_pos = ax_line.get_position()
        scatter_pos = ax_scatter.get_position()

        # 调整散点图位置，使其更靠近折线图
        # 减小left值使散点图向左移动，靠近折线图
        new_left = line_pos.x1 + 0.04  # 可以调整这个值，使散点图更靠近折线图
        ax_scatter.set_position([new_left, scatter_pos.y0,
                                 scatter_pos.width, scatter_pos.height])

    # 保存组合图 - 将DPI提高到1000
    plt.savefig(os.path.join(save_dir, 'time_analysis_combined.png'), dpi=1000, bbox_inches='tight')
    plt.close()

    print("已保存时间点分析组合图，DPI=1000")


def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_tigge_wind = []
    all_dates = []
    criterion = nn.SmoothL1Loss()  # 修改为与训练一致，提升鲁棒性

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

                test_loss += loss.item() * tigge_spatial.size(0)
                batch_count += 1

                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
                all_tigge_wind.append(tigge_spatial[:, 0, :, :].cpu())

                start_idx = batch_idx * test_loader.batch_size
                end_idx = min((batch_idx + 1) * test_loader.batch_size, len(test_loader.dataset))
                batch_dates = test_loader.dataset.time_points[
                              start_idx + test_loader.dataset.seq_len - 1:end_idx + test_loader.dataset.seq_len - 1]
                all_dates.extend(batch_dates)

            except Exception as e:
                print(f"Error processing test batch {batch_idx}: {str(e)}")
                continue

        test_loss = test_loss / len(test_loader.dataset) if batch_count > 0 else float('inf')
        print(f'test Loss: {test_loss:.4f}')

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

        np.save('test_preds.npy', all_preds_orig)
        np.save('test_targets.npy', all_targets_orig)
        np.save('test_dates.npy', all_dates)
        print("test predictions, targets, TIGGE wind,noisy TIGGE and dates saved as .npy files.")

       
        print("\n===== 四个时间点指标对比 =====")
        hours = ['00', '06', '12', '18']

        # 定义指标及其显示名称
        metrics_display = {'FA': 'FA (%)', 'RMSE': 'RMSE (m/s)', 'MAE': 'MAE (m/s)',
                           'rRMSE': 'rRMSE (%)', 'rMAE': 'rMAE (%)', 'R': 'R', 'R2': 'R²'}
        metrics_names = ['FA', 'RMSE', 'MAE', 'rRMSE', 'rMAE', 'R', 'R2']

        # 为每个时间点计算指标
        hourly_metrics = {}
        for hour in hours:
            hour_mask = (pd.to_datetime(all_dates).hour == int(hour))
            if np.sum(hour_mask) > 0:  # 确保当前时间点有数据
                hour_preds = all_preds_orig[hour_mask]
                hour_targets = all_targets_orig[hour_mask]
                hour_tigge = all_tigge_wind_orig[hour_mask]
                hourly_metrics[hour] = calculate_metrics(
                    torch.from_numpy(hour_preds),
                    torch.from_numpy(hour_targets),
                    torch.from_numpy(hour_tigge)
                )

        # 打印表头
        header = f"{'指标':<10} | {'模型':<8} | "
        for hour in hours:
            header += f"{hour}:00  | "
        print(header)
        print("-" * (10 + 10 + 10 * len(hours)))

        # 打印每个指标的值
        for metric in metrics_names:
            metric_display = metrics_display[metric]

            # Proposed模型行
            row = f"{metric_display:<10} | {'Proposed':<8} | "
            for hour in hours:
                if hour in hourly_metrics:
                    row += f"{hourly_metrics[hour][f'{metric}_pred']:<8.4f} | "
                else:
                    row += f"{'N/A':<8} | "
            print(row)

            # ECMWF行
            row = f"{'':<10} | {'ECMWF':<8} | "
            for hour in hours:
                if hour in hourly_metrics:
                    row += f"{hourly_metrics[hour][f'{metric}_tigge']:<8.4f} | "
                else:
                    row += f"{'N/A':<8} | "
            print(row)

            # 添加分隔线
            if metric != metrics_names[-1]:
                print("-" * (10 + 10 + 10 * len(hours)))

        # 保存时间点指标到文件
        with open('test_hourly_metrics.json', 'w') as f:
            json.dump({hour: {k: float(v) for k, v in metrics.items()}
                       for hour, metrics in hourly_metrics.items()}, f, indent=2)
        print("时间点指标已保存至 test_hourly_metrics.json")
        metrics_all = calculate_metrics(torch.from_numpy(all_preds_orig), torch.from_numpy(all_targets_orig),
                                        torch.from_numpy(all_tigge_wind_orig))
        print("Yearly test Metrics:", metrics_all)

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
            seasonal_metrics[season] = calculate_metrics(torch.from_numpy(season_preds),
                                                         torch.from_numpy(season_targets),
                                                         torch.from_numpy(season_tigge))
            print(f"{season} test Metrics:", seasonal_metrics[season])

        all_metrics = {'Yearly': metrics_all, **seasonal_metrics}
        with open('test_metrics.json', 'w') as f:
            json.dump(all_metrics, f)
        print("test metrics saved as test_metrics.json")

        monthly_metrics = {}
        months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        metrics_names = ['FA', 'RMSE', 'MAE', 'rRMSE', 'rMAE', 'R', 'R2']  # 7个要可视化的指标
        months = list(range(1, 13))

        # 第一步：计算每个月的指标
        for month in range(1, 13):
            month_mask = (pd.to_datetime(all_dates).month == month)
            if np.sum(month_mask) > 0:  # 确保当月有数据
                month_preds = all_preds_orig[month_mask]
                month_targets = all_targets_orig[month_mask]
                month_tigge = all_tigge_wind_orig[month_mask]
                monthly_metrics[month] = calculate_metrics(torch.from_numpy(month_preds),
                                                           torch.from_numpy(month_targets),
                                                           torch.from_numpy(month_tigge))
                print(f"Month {months_names[month - 1]} test Metrics:", monthly_metrics[month])

        # 保存月度指标到单独的文件
        monthly_metrics_dict = {str(k): v for k, v in monthly_metrics.items()}
        with open('test_monthly_metrics.json', 'w') as f:
            json.dump(monthly_metrics_dict, f)
        print("Monthly test metrics saved as test_monthly_metrics.json")

        print("\n===== 全年指标对比 =====")
        model_names = ['Proposed'] + list(baseline_metrics.keys()) + ['ECMWF']

        metrics_display = {'FA': 'FA (%)', 'RMSE': 'RMSE (m/s)', 'MAE': 'MAE (m/s)',
                           'rRMSE': 'rRMSE (%)', 'rMAE': 'rMAE (%)', 'R': 'R', 'R2': 'R²'}

        # 打印表头
        header = f"{'指标':<10} | "
        for model in model_names:
            header += f"{model:<12} | "
        print(header)
        print("-" * (10 + 15 * len(model_names)))

        # 打印每个指标的值
        for metric in metrics_names:
            metric_display = metrics_display[metric]
            row = f"{metric_display:<10} | "

            # 主模型的值
            row += f"{metrics_all[f'{metric}_pred']:<12.4f} | "

            # 各baseline模型的值
            for model_name in baseline_metrics.keys():
                try:
                    row += f"{baseline_metrics[model_name]['Yearly'][f'{metric}_pred']:<12.4f} | "
                except (KeyError, TypeError):
                    row += f"{'N/A':<12} | "

            # TIGGE的值
            row += f"{metrics_all[f'{metric}_tigge']:<12.4f} | "

            print(row)

        print("\n创建季节性指标对比可视化...")

        # 更新模型名称映射 (与月度指标对比中的相同)
        model_name_mapping = {
            'Proposed': 'MTRCL',
            'LTCs_TE': 'TE-LTCs',
            'ResNet_TE': 'TE-ResNet',
            'ResNet_TE_CBAM': 'TC-ResNet',
            'ECMWF': 'ECMWF-TIGGE'
        }

        display_model_names = [model_name_mapping.get(name, name) for name in model_names]

        # 设置Times New Roman字体并加粗
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴线宽
        plt.rcParams['xtick.labelsize'] = 14  # 设置x轴刻度标签字体大小
        plt.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度标签字体大小


        # 为每个指标创建一个季节性对比图
        for metric in metrics_names:
            plt.figure(figsize=(14, 8))

            # 设置季节标签和位置
            seasons_list = list(seasons.keys())
            x = np.arange(len(seasons_list))
            width = 0.1  # 柱状图宽度

            # 计算每个模型的柱状图位置偏移
            offsets = np.linspace(-0.35, 0.35, len(model_names))

            # 为每个模型绘制柱状图
            for i, model_name in enumerate(model_names):
                display_name = model_name_mapping.get(model_name, model_name)
                color = model_colors.get(display_name, 'gray')

                values = []

                for season in seasons_list:
                    if model_name == 'Proposed':
                        values.append(seasonal_metrics[season][f'{metric}_pred'])
                    elif model_name == 'ECMWF':
                        # 使用调整后的ECMWF值
                        values.append(seasonal_metrics[season][f'{metric}_tigge'])
                    else:
                        try:
                            values.append(baseline_metrics[model_name][season][f'{metric}_pred'])
                        except (KeyError, TypeError):
                            values.append(np.nan)  # 使用NaN表示缺失数据

                plt.bar(x + offsets[i], values, width, label=display_name, color=color)

            # 设置图表标签
            plt.ylabel(metrics_display[metric], fontsize=16, fontweight='bold')
            plt.xticks(x, seasons_list, fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=12)  # 移除fontweight参数

            # 只保留左侧和底部轴
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            # 增加可见轴的线宽
            plt.gca().spines['left'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)

            # 保存图表
            plt.tight_layout()
            plt.savefig(f'test_seasonal_comparison_{metric}.png', dpi=1000)
            plt.close()
            print(f"季节性{metric}对比图已保存为 test_seasonal_comparison_{metric}.png")

        print("\n创建月度指标对比可视化...")

        # 更新模型名称映射
        model_name_mapping = {
            'Proposed': 'MTRCL',
            'LTCs_TE': 'TE-LTCs',
            'ResNet_TE': 'TE-ResNet',
            'ResNet_TE_CBAM': 'TC-ResNet',
            'ECMWF': 'ECMWF-TIGGE'
        }

        # 为每个模型定义不同的标记
        marker = 'o'

        # 更新模型名称列表
        display_model_names = [model_name_mapping.get(name, name) for name in model_names]

        # 设置Times New Roman字体，并加粗
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴线宽
        plt.rcParams['xtick.labelsize'] = 14  # 设置x轴刻度标签字体大小
        plt.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度标签字体大小

        # 为每个指标创建一个月度对比折线图
        for metric in metrics_names:
            plt.figure(figsize=(14, 8))

            # 为每个模型绘制折线图
            for i, model_name in enumerate(model_names):
                values = []
                display_name = model_name_mapping.get(model_name, model_name)
                color = model_colors.get(display_name, 'gray')

                for month in months:
                    if model_name == 'Proposed':
                        if month in monthly_metrics:
                            values.append(monthly_metrics[month][f'{metric}_pred'])
                        else:
                            values.append(np.nan)
                    elif model_name == 'ECMWF':
                        if month in monthly_metrics:
                            # 使用调整后的ECMWF值
                            values.append(monthly_metrics[month][f'{metric}_tigge'])
                        else:
                            values.append(np.nan)
                    else:
                        try:
                            # 注意baseline_monthly_metrics中键是字符串
                            if str(month) in baseline_monthly_metrics[model_name]:
                                values.append(baseline_monthly_metrics[model_name][str(month)][f'{metric}_pred'])
                            else:
                                values.append(np.nan)
                        except (KeyError, TypeError):
                            values.append(np.nan)

                # 所有线条都使用实线，只用标记来区分
                plt.plot(months, values, '-', marker=marker, linewidth=2.5, markersize=8,
                         label=display_name, color=color)

            # 设置图表标签
            plt.ylabel(metrics_display[metric], fontsize=16, fontweight='bold')
            plt.xticks(months, months_names, fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=12)  # 移除fontweight参数
            # 只保留左侧和底部轴
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            # 增加可见轴的线宽
            plt.gca().spines['left'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)
            # 保存图表
            plt.tight_layout()
            plt.savefig(f'test_monthly_comparison_{metric}.png', dpi=1000)
            plt.close()
            print(f"月度{metric}对比图已保存为 test_monthly_comparison_{metric}.png")

        hours = ['00', '06', '12', '18']
        hour_indices = {hour: i for i, hour in enumerate(hours)}
        hour_labels = ['(a) 00:00 UTC', '(b) 06:00 UTC', '(c) 12:00 UTC', '(d) 18:00 UTC']

        # 设置全局字体属性 - 使用Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        font_size = 14  # 与空间指标分布图一致的字体大小

        # 存储所有小时的数据，用于合并图
        all_data = []

        # 生成单独的图表
        for i, hour in enumerate(hours):
            hour_mask = (pd.to_datetime(all_dates).hour == int(hour))
            hour_dates = pd.to_datetime(all_dates[hour_mask])
            daily_preds = all_preds_orig[hour_mask].mean(axis=(1, 2))
            daily_targets = all_targets_orig[hour_mask].mean(axis=(1, 2))
            daily_tigge = all_tigge_wind_orig[hour_mask].mean(axis=(1, 2))

            days_of_year = hour_dates.dayofyear.values

            # 保存数据用于大图
            all_data.append({
                'days': days_of_year,
                'targets': daily_targets,
                'preds': daily_preds,
                'tigge': daily_tigge
            })

            fig, ax = plt.subplots(figsize=(12, 6))

            # 设置图形边框线宽
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # 使用指定的颜色绘制折线
            ax.plot(days_of_year, daily_targets, label='ERA5 hourly data on single levels',
                    color=colors['ERA5'], linewidth=1.5,zorder = 3)
            ax.plot(days_of_year, daily_preds, label='MTRCL',
                    color=colors['MTRCL'], linewidth=1.5,zorder = 2)
            ax.plot(days_of_year, daily_tigge, label='ECMWF-TIGGE',
                    color=colors['ECMWF'], linewidth=1.5,zorder = 1)

            # 设置x轴标签
            ax.set_xlabel('Day of Year', fontsize=font_size+4, fontweight='bold')

            # 设置y轴标签
            ax.set_ylabel('Wind Speed (m/s)', fontsize=font_size, fontweight='bold')

            # 添加图注在左上角，稍微下移
            ax.text(0.01, 0.96, hour_labels[i], transform=ax.transAxes,
                    fontsize=font_size+4, fontweight='bold', va='top', ha='left')

            # 设置x轴刻度，确保从原点开始
            ax.set_xlim(0, 370)
            ax.set_xticks(np.arange(0, 370, 50))

            # 设置刻度字体
            ax.tick_params(axis='both', which='major', width=1.5, labelsize=font_size)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')

            # 将图例放在图的上方，与图注在同一行但靠右
            legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=3,
                               fontsize=font_size, frameon=False)

            # 设置图例文本字体粗细
            for text in legend.get_texts():
                text.set_fontweight('bold')

            # 保存单独的图片，不使用tight_layout
            fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
            plt.savefig(f'test_wind_speed_2024_{hour}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Wind speed plot saved as test_wind_speed_2024_{hour}.png')
        for name in metrics_names:
            plt.figure(figsize=(10, 6))
            yearly_pred = metrics_all[f'{name}_pred']
            yearly_tigge = metrics_all[f'{name}_tigge']
            seasonal_pred = [seasonal_metrics[season][f'{name}_pred'] for season in seasons]
            seasonal_tigge = [seasonal_metrics[season][f'{name}_tigge'] for season in seasons]

            bar_width = 0.35
            index = np.arange(5)
            plt.bar(index - bar_width / 2, [yearly_pred] + seasonal_pred, bar_width, label='Predicted', color='blue')
            plt.bar(index + bar_width / 2, [yearly_tigge] + seasonal_tigge, bar_width, label='ECMWF', color='orange')
            plt.xlabel('Period')
            plt.ylabel(name)
            plt.title(f'test {name} Comparison (Yearly and Seasonal)')
            plt.xticks(index, ['Yearly', 'Spring', 'Summer', 'Autumn', 'Winter'])
            plt.legend()
            plt.savefig(f'test_metrics_{name}.png')
            plt.close()
            print(f'Metrics plot saved as test_metrics_{name}.png')

        print("\n计算并绘制空间指标分布...")


        if len(all_preds_orig.shape) == 3:  # [N, H, W]
            spatial_metrics = calculate_spatial_metrics(
                all_preds_orig,
                all_targets_orig,
                all_tigge_wind_orig
            )

            # 华北地区的经纬度范围
            lat_range = (35.13, 47.0)
            lon_range = (103.0, 126.88)
        else:

            try:

                N, H, W = all_preds_orig.shape
                print(f"重塑数据形状: 从 {all_preds_orig.shape} 到 [N, H, W]")

                # 如果已经是正确形状，不需要重塑
                spatial_metrics = calculate_spatial_metrics(
                    all_preds_orig,
                    all_targets_orig,
                    all_tigge_wind_orig
                )

                # 华北地区的经纬度范围
                lat_range = (35.13, 47.0)
                lon_range = (103.0, 126.88)

                # 绘制空间指标分布图
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                # 在主函数中调用plot_spatial_metrics时，确保传递正确的地图路径
                plot_spatial_metrics(spatial_metrics,
                                     lat_range=(35.13, 47),
                                     lon_range=(103, 126.88),
                                     save_dir='spatial_metrics')
                print("空间指标分布图已保存到 spatial_metrics 目录")

                # 创建更精细的空间指标可视化
                plot_spatial_metrics_paper_style(
                    all_preds_orig,
                    all_targets_orig,
                    all_tigge_wind_orig,
                    lat_range,
                    lon_range,
                    save_path='spatial_metrics_paper_style.png'
                )
                print("论文风格的空间指标分布图已保存")
            except Exception as e:
                print(f"无法处理形状为 {all_preds_orig.shape} 的数据: {str(e)}")
                print(f"警告: 预测数据形状不正确，无法计算空间指标分布。当前形状: {all_preds_orig.shape}")

# 主函数
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
        dropout_rate=0.24733185479083603,
        ltc_hidden_dim=216,
        cbam_reduction=16
    ).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Testing...")
    test_model(model, test_loader, device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing completed!")
