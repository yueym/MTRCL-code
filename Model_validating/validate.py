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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
import matplotlib.dates as mdates
import matplotlib.font_manager as fm


# CBAM模块（更新 reduction 参数）
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


# === 新增: 计算空间分布指标 ===
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

def load_all_baseline_metrics():
    """
    加载所有baseline模型的指标数据
    """
    # 模型名称和对应的指标文件路径
    baseline_files = {
        'ConvLSTM': ('ConvLSTM_val_metrics.json', 'ConvLSTM_val_monthly_metrics.json'),
        'LTCs_TE': ('LTCs_val_metrics.json', 'LTCs_val_monthly_metrics.json'),
        'ResNet': ('ResNet_val_metrics.json', 'ResNet_val_monthly_metrics.json'),
        'ResNet_TE': ('ResNetTE_val_metrics.json', 'ResNetTE_val_monthly_metrics.json'),
        'ResNet_TE_CBAM': ('ResNetCBAMTE_val_metrics.json', 'ResNetCBAMTE_val_monthly_metrics.json')
    }

    # 存储所有指标
    all_metrics = {}
    all_monthly_metrics = {}

    # 加载每个模型的指标
    for model_name, (metrics_file, monthly_file) in baseline_files.items():
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics[model_name] = metrics

            with open(monthly_file, 'r') as f:
                monthly_metrics = json.load(f)
                all_monthly_metrics[model_name] = monthly_metrics

            print(f"成功加载 {model_name} 模型的指标")
        except FileNotFoundError:
            print(f"警告: 无法找到 {model_name} 的指标文件 ({metrics_file} 或 {monthly_file})")
            continue

    return all_metrics, all_monthly_metrics

def plot_hourly_scatter_comparison(all_preds, all_targets, all_tigge_wind, all_dates, save_dir='hourly_metrics'):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 转换日期为pandas datetime对象
    dates = pd.to_datetime(all_dates)

    # 提取小时信息
    hours = dates.hour

    # 时间点名称
    hour_names = {
        0: '00:00',
        6: '06:00',
        12: '12:00',
        18: '18:00'
    }

    # 为每个时间点创建散点图
    for hour in [0, 6, 12, 18]:
        hour_mask = (hours == hour)

        # 跳过没有数据的时间点
        if not np.any(hour_mask):
            print(f"警告: 没有 {hour_names[hour]} 的数据")
            continue

        # 获取该时间点的数据
        hour_preds = all_preds[hour_mask].flatten()
        hour_targets = all_targets[hour_mask].flatten()
        hour_tigge = all_tigge_wind[hour_mask].flatten()

        # 数据采样 - 大幅减少点的数量以提高可视性
        np.random.seed(42)  # 设置随机种子以确保可重复性

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

        metrics = calculate_metrics(
            torch.from_numpy(hour_preds),
            torch.from_numpy(hour_targets),
            torch.from_numpy(hour_tigge)
        )

        # 创建图形
        plt.figure(figsize=(10, 10))

        # 设置点的透明度和大小
        point_alpha = 0.35
        point_size = 12

        # 设置线的粗细 - 两种线使用相同的虚线和粗度
        line_width = 2.5

        plt.scatter(hour_targets_tigge, hour_tigge_sampled, alpha=point_alpha, s=point_size, c='blue', label='ECMWF')

        # 绘制模型预测散点图 - 红色
        plt.scatter(hour_targets_pred, hour_preds_sampled, alpha=point_alpha, s=point_size, c='red', label='Proposed')

        # 添加对角线 - 使用实线
        max_val = max(np.max(hour_targets), np.max(hour_preds), np.max(hour_tigge))
        min_val = min(np.min(hour_targets), np.min(hour_preds), np.min(hour_tigge))
        plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5)

        # 通过原点的线性拟合: y = kx (无截距)
        z_pred = np.sum(hour_targets * hour_preds) / np.sum(hour_targets * hour_targets)
        p_pred = lambda x: z_pred * x

        z_tigge = np.sum(hour_targets * hour_tigge) / np.sum(hour_targets * hour_targets)
        p_tigge = lambda x: z_tigge * x

        plt.plot(np.sort(hour_targets_pred), p_pred(np.sort(hour_targets_pred)), "r--", linewidth=line_width)
        plt.plot(np.sort(hour_targets_tigge), p_tigge(np.sort(hour_targets_tigge)), "b--", linewidth=line_width)

        metrics_text_pred = f"Proposed: y={z_pred:.2f}x"
        metrics_text_pred_2 = f"N={len(hour_targets)}"
        metrics_text_pred_3 = f"RMSE={metrics['RMSE_pred']:.3f}"
        metrics_text_pred_4 = f"MAE={metrics['MAE_pred']:.3f}"
        metrics_text_pred_5 = f"FA={metrics['FA_pred']:.2f}%"
        metrics_text_pred_6 = f"R={metrics['R_pred']:.3f}"

        # 放置文本 - 左上角，红色，字体更大
        plt.text(0.05, 0.95, metrics_text_pred, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='left', color='red')
        plt.text(0.05, 0.90, metrics_text_pred_2, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='left', color='red')
        plt.text(0.05, 0.85, metrics_text_pred_3, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='left', color='red')
        plt.text(0.05, 0.80, metrics_text_pred_4, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='left', color='red')
        plt.text(0.05, 0.75, metrics_text_pred_5, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='left', color='red')
        plt.text(0.05, 0.70, metrics_text_pred_6, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='left', color='red')

        metrics_text_tigge = f"ECMWF: y={z_tigge:.2f}x"
        metrics_text_tigge_2 = f"RMSE={metrics['RMSE_tigge']:.3f}"
        metrics_text_tigge_3 = f"MAE={metrics['MAE_tigge']:.3f}"
        metrics_text_tigge_4 = f"FA={metrics['FA_tigge']:.2f}%"
        metrics_text_tigge_5 = f"R={metrics['R_tigge']:.3f}"

        # 放置文本 - 右下角，蓝色，字体更大
        plt.text(0.95, 0.30, metrics_text_tigge, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='right', color='blue')
        plt.text(0.95, 0.25, metrics_text_tigge_2, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='right', color='blue')
        plt.text(0.95, 0.20, metrics_text_tigge_3, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='right', color='blue')
        plt.text(0.95, 0.15, metrics_text_tigge_4, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='right', color='blue')
        plt.text(0.95, 0.10, metrics_text_tigge_5, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='right', color='blue')

        # 设置轴标签
        plt.xlabel('Observed wind speed (m/s)', fontsize=14)
        plt.ylabel('Estimated wind speed (m/s)', fontsize=14)
        plt.title(f'Wind Speed Scatter Plot at {hour_names[hour]}', fontsize=16)
        # 去除网格线
        plt.grid(False)
        plt.legend(loc='lower right', fontsize=12)

        # 设置轴范围
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        # 调整布局
        plt.tight_layout()

        # 保存图形
        plt.savefig(os.path.join(save_dir, f'scatter_plot_{hour:02d}00.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存 {hour_names[hour]} 时间点的风速散点图")

    # 创建组合图 (所有时间点在一个大图中)
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    for i, hour in enumerate([0, 6, 12, 18]):
        hour_mask = (hours == hour)

        # 跳过没有数据的时间点
        if not np.any(hour_mask):
            axes[i].text(0.5, 0.5, f"No data for {hour_names[hour]}",
                         ha='center', va='center', fontsize=14)
            continue

        # 获取该时间点的数据
        hour_preds = all_preds[hour_mask].flatten()
        hour_targets = all_targets[hour_mask].flatten()
        hour_tigge = all_tigge_wind[hour_mask].flatten()

        # 数据采样 - 减少点的数量以提高可视性
        np.random.seed(42 + i)  # 为每个子图使用不同的随机种子

        sample_size_tigge = min(1200, len(hour_targets))
        sample_size_pred = min(2000, len(hour_targets))

        # 分别采样ECMWF和模型预测点
        sample_indices_tigge = np.random.choice(len(hour_targets), sample_size_tigge, replace=False)
        sample_indices_pred = np.random.choice(len(hour_targets), sample_size_pred, replace=False)

        # 获取采样后的数据
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

        # 绘制散点图 - 同时显示模型和ECMWF
        ax = axes[i]

        # 设置点的透明度和大小
        point_alpha = 0.35
        point_size = 10

        # 设置线的粗细 - 两种线使用相同的虚线和粗度
        line_width = 2.5

        # 绘制ECMWF散点图 - 蓝色
        ax.scatter(hour_targets_tigge, hour_tigge_sampled, alpha=point_alpha, s=point_size, c='blue', label='ECMWF')

        # 绘制模型预测散点图 - 红色
        ax.scatter(hour_targets_pred, hour_preds_sampled, alpha=point_alpha, s=point_size, c='red', label='Proposed')

        # 添加对角线 - 使用实线
        max_val = max(np.max(hour_targets), np.max(hour_preds), np.max(hour_tigge))
        min_val = min(np.min(hour_targets), np.min(hour_preds), np.min(hour_tigge))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5)

        # 添加线性拟合 - 强制通过原点
        z_pred = np.sum(hour_targets * hour_preds) / np.sum(hour_targets * hour_targets)
        p_pred = lambda x: z_pred * x

        z_tigge = np.sum(hour_targets * hour_tigge) / np.sum(hour_targets * hour_targets)
        p_tigge = lambda x: z_tigge * x

        # 修改：两种拟合线都使用相同的虚线和粗度
        ax.plot(np.sort(hour_targets_pred), p_pred(np.sort(hour_targets_pred)), "r--", linewidth=line_width)
        ax.plot(np.sort(hour_targets_tigge), p_tigge(np.sort(hour_targets_tigge)), "b--", linewidth=line_width)

        # 模型预测指标 - 左上角
        metrics_text_pred = f"Proposed: y={z_pred:.2f}x"
        metrics_text_pred_2 = f"RMSE={metrics['RMSE_pred']:.3f}"
        metrics_text_pred_3 = f"R={metrics['R_pred']:.3f}"

        # ECMWF指标 - 右下角
        metrics_text_tigge = f"ECMWF: y={z_tigge:.2f}x"
        metrics_text_tigge_2 = f"RMSE={metrics['RMSE_tigge']:.3f}"
        metrics_text_tigge_3 = f"R={metrics['R_tigge']:.3f}"

        # 放置文本 - 左上角，红色
        ax.text(0.05, 0.95, metrics_text_pred, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='left', color='red')
        ax.text(0.05, 0.90, metrics_text_pred_2, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='left', color='red')
        ax.text(0.05, 0.85, metrics_text_pred_3, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='left', color='red')

        # 放置文本 - 右下角，蓝色
        ax.text(0.95, 0.20, metrics_text_tigge, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right', color='blue')
        ax.text(0.95, 0.15, metrics_text_tigge_2, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right', color='blue')
        ax.text(0.95, 0.10, metrics_text_tigge_3, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right', color='blue')

        # 设置轴标签
        ax.set_xlabel('Observed wind speed (m/s)', fontsize=12)
        ax.set_ylabel('Estimated wind speed (m/s)', fontsize=12)
        ax.set_title(f'Wind Speed at {hour_names[hour]}', fontsize=14)
        # 去除网格线
        ax.grid(False)
        ax.legend(loc='lower right', fontsize=10)

        # 设置轴范围
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle('Wind Speed Scatter Plots by Time of Day', fontsize=20)

    # 保存组合图
    plt.savefig(os.path.join(save_dir, 'scatter_plots_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("已保存组合风速散点图")


# === 新增: 绘制中国区域风速分布对比图 ===
def plot_china_wind_distribution(all_preds, all_targets, all_tigge_wind, save_dir='china_wind_distribution'):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 计算平均风速分布
    mean_pred = np.mean(all_preds, axis=0)
    mean_tigge = np.mean(all_tigge_wind, axis=0)

    # 使用研究区域的经纬度范围
    lat_min, lat_max = 35.13, 47.0
    lon_min, lon_max = 103.0, 126.88

    # 创建华北地区掩膜
    H, W = mean_pred.shape
    north_china_mask = create_north_china_mask(H, W, lat_min, lat_max, lon_min, lon_max)

    # 应用掩膜
    mean_pred = np.where(north_china_mask, mean_pred, np.nan)
    mean_tigge = np.where(north_china_mask, mean_tigge, np.nan)

    # 创建经纬度网格 - 确保精确匹配指定的范围
    lats = np.linspace(lat_min, lat_max, H)
    lons = np.linspace(lon_min, lon_max, W)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 确定颜色范围
    vmin = 0
    vmax = 6  # 根据图片调整最大值

    # 创建自定义颜色映射 - 从蓝色到红色的渐变
    cmap = plt.colormaps.get_cmap('coolwarm')  # 使用coolwarm代替Blues_r

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 添加子图标签 - 明确标注哪个是Proposed，哪个是ECMWF
    axes[0].text(0.05, 0.95, '(a) Proposed', transform=axes[0].transAxes,
                 fontsize=16, fontweight='bold', va='top')
    axes[1].text(0.05, 0.95, '(b) ECMWF', transform=axes[1].transAxes,
                 fontsize=16, fontweight='bold', va='top')

    # 获取掩膜边界
    contours = measure.find_contours(north_china_mask.reshape(H, W), 0.5)

    im1 = axes[0].pcolormesh(lon_grid, lat_grid, mean_pred,
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             shading='auto', alpha=1.0)

    # 添加区域边界线
    for contour in contours:
        # 将轮廓点转换为经纬度坐标
        contour_lons = np.interp(contour[:, 1], np.arange(W), lons)
        contour_lats = np.interp(contour[:, 0], np.arange(H), lats)
        axes[0].plot(contour_lons, contour_lats, 'k-', linewidth=0.8, alpha=0.7)

    # 添加主图边界 - 绘制一个黑色矩形框
    axes[0].plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                 [lat_min, lat_min, lat_max, lat_max, lat_min],
                 'k-', linewidth=1.5)

    axes[0].set_xlabel('Longitude (°E)', fontsize=12)
    axes[0].set_ylabel('Latitude (°N)', fontsize=12)
    # 确保精确设置经纬度范围
    axes[0].set_xlim(lon_min, lon_max)
    axes[0].set_ylim(lat_min, lat_max)
    # 设置刻度以确保显示精确的起点和终点
    axes[0].set_xticks([lon_min, 110, 120, lon_max])
    axes[0].set_yticks([lat_min, 40, 45, lat_max])
    # 去除网格线
    axes[0].grid(False)

    # 在图内添加直方图 - 放在左上角但避开标签区域，并向下移动
    inset_ax1 = axes[0].inset_axes([0.1, 0.60, 0.3, 0.2])  # 修改位置，向下移动到0.60
    hist_data1 = mean_pred.flatten()
    hist_data1 = hist_data1[~np.isnan(hist_data1)]
    if len(hist_data1) > 0:
        # 计算直方图 - 修改为蓝色
        counts, bins, _ = inset_ax1.hist(hist_data1, bins=20, range=(vmin, vmax),
                                         color='blue', alpha=0.7)  # 修改为蓝色

        # 设置直方图格式
        inset_ax1.set_xlim(vmin, vmax)
        inset_ax1.set_xlabel('Wind speed (m/s)', fontsize=8)
        inset_ax1.set_ylabel('Cell counts', fontsize=8)  # 使用"Cell counts"
        inset_ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        inset_ax1.tick_params(axis='both', which='major', labelsize=8)
        # 去除直方图网格线
        inset_ax1.grid(False)

    # 修改：改回使用pcolormesh代替散点图，增加颜色饱和度
    # 绘制ECMWF模型的风速分布
    im2 = axes[1].pcolormesh(lon_grid, lat_grid, mean_tigge,
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             shading='auto', alpha=1.0)  # 增加alpha值到1.0

    # 添加区域边界线
    for contour in contours:
        # 将轮廓点转换为经纬度坐标
        contour_lons = np.interp(contour[:, 1], np.arange(W), lons)
        contour_lats = np.interp(contour[:, 0], np.arange(H), lats)
        axes[1].plot(contour_lons, contour_lats, 'k-', linewidth=0.8, alpha=0.7)

    # 添加主图边界 - 绘制一个黑色矩形框
    axes[1].plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                 [lat_min, lat_min, lat_max, lat_max, lat_min],
                 'k-', linewidth=1.5)

    axes[1].set_xlabel('Longitude (°E)', fontsize=12)
    axes[1].set_ylabel('Latitude (°N)', fontsize=12)
    # 确保精确设置经纬度范围
    axes[1].set_xlim(lon_min, lon_max)
    axes[1].set_ylim(lat_min, lat_max)
    # 设置刻度以确保显示精确的起点和终点
    axes[1].set_xticks([lon_min, 110, 120, lon_max])
    axes[1].set_yticks([lat_min, 40, 45, lat_max])
    # 去除网格线
    axes[1].grid(False)

    # 在图内添加直方图 - 放在左上角但避开标签区域
    inset_ax2 = axes[1].inset_axes([0.1, 0.60, 0.3, 0.2])  # 修改位置，向下移动到0.60
    hist_data2 = mean_tigge.flatten()
    hist_data2 = hist_data2[~np.isnan(hist_data2)]
    if len(hist_data2) > 0:
        # 计算直方图 - 修改为蓝色
        counts, bins, _ = inset_ax2.hist(hist_data2, bins=20, range=(vmin, vmax),
                                         color='blue', alpha=0.7)  # 修改为蓝色

        # 设置直方图格式
        inset_ax2.set_xlim(vmin, vmax)
        inset_ax2.set_xlabel('Wind speed (m/s)', fontsize=8)
        inset_ax2.set_ylabel('Number of samples', fontsize=8)
        inset_ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        inset_ax2.tick_params(axis='both', which='major', labelsize=8)
        # 去除直方图网格线
        inset_ax2.grid(False)

    # 修改：添加垂直颜色条在最右侧
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 位置：[左, 下, 宽, 高]
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Wind Speed (m/s)', fontsize=12)

    # 保存图形
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.1, top=0.95)
    plt.savefig(os.path.join(save_dir, 'china_wind_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("已保存中国区域风速分布对比图")

def validate_model(model, val_loader, device):
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
        val_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(val_loader):
            try:
                tigge_spatial = batch['tigge_spatial'].to(device)
                dem_spatial = batch['dem_spatial'].to(device)
                tigge_seq = batch['tigge_seq'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                time_features_seq = batch['time_features_seq'].to(device)
                target = batch['target'].to(device)

                output = model(tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq)
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
        np.save('val_preds.npy', all_preds_orig)
        np.save('val_targets.npy', all_targets_orig)
        np.save('val_tigge_wind.npy', all_tigge_wind_orig)

        np.save('val_dates.npy', all_dates)
        print("Validation predictions, targets, TIGGE wind, dates saved as .npy files.")

        print("绘制时间点风速散点图...")
        plot_hourly_scatter_comparison(all_preds_orig, all_targets_orig, all_tigge_wind_orig, all_dates,
                                       save_dir='hourly_metrics')

        plot_china_wind_distribution(all_preds_orig, all_targets_orig, all_tigge_wind_orig,
                                     save_dir='china_wind_distribution')

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
        with open('hourly_metrics.json', 'w') as f:
            json.dump({hour: {k: float(v) for k, v in metrics.items()}
                       for hour, metrics in hourly_metrics.items()}, f, indent=2)
        print("时间点指标已保存至 hourly_metrics.json")

    

        print("加载所有baseline模型的指标数据...")
        baseline_metrics, baseline_monthly_metrics = load_all_baseline_metrics()

        metrics_all = calculate_metrics(torch.from_numpy(all_preds_orig), torch.from_numpy(all_targets_orig),
                                        torch.from_numpy(all_tigge_wind_orig))
        print("Yearly Validation Metrics:", metrics_all)

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
            print(f"{season} Validation Metrics:", seasonal_metrics[season])

        all_metrics = {'Yearly': metrics_all, **seasonal_metrics}
        with open('val_metrics.json', 'w') as f:
            json.dump(all_metrics, f)
        print("Validation metrics saved as val_metrics.json")

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
                print(f"Month {months_names[month - 1]} Validation Metrics:", monthly_metrics[month])

        # 保存月度指标到单独的文件
        monthly_metrics_dict = {str(k): v for k, v in monthly_metrics.items()}
        with open('val_monthly_metrics.json', 'w') as f:
            json.dump(monthly_metrics_dict, f)
        print("Monthly validation metrics saved as val_monthly_metrics.json")

        print("\n===== 全年指标对比 =====")
        model_names = ['Proposed'] + list(baseline_metrics.keys()) + ['ECMWF']

        # 为了表格美观，计算每个指标名称的最大长度
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

        # 为每个指标创建一个季节性对比图
        for metric in metrics_names:
            plt.figure(figsize=(14, 8))

            # 设置季节标签和位置
            seasons_list = list(seasons.keys())
            x = np.arange(len(seasons_list))
            width = 0.1  # 柱状图宽度

            # 计算每个模型的柱状图位置偏移
            offsets = np.linspace(-0.35, 0.35, len(model_names))

            # 定义自定义颜色方案 - 可以根据需要调整
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                      '#bcbd22', '#17becf']
            # 为每个模型绘制柱状图
            for i, model_name in enumerate(model_names):
                values = []

                for season in seasons_list:
                    if model_name == 'Proposed':
                        values.append(seasonal_metrics[season][f'{metric}_pred'])
                    elif model_name == 'ECMWF':
                        values.append(seasonal_metrics[season][f'{metric}_tigge'])
                    else:
                        try:
                            values.append(baseline_metrics[model_name][season][f'{metric}_pred'])
                        except (KeyError, TypeError):
                            values.append(np.nan)  # 使用NaN表示缺失数据

                color_idx = i % len(colors)
                plt.bar(x + offsets[i], values, width, label=model_name, color=colors[color_idx])

            # 设置图表标题和标签
            plt.ylabel(metrics_display[metric], fontsize=14)
            plt.xticks(x, seasons_list, fontsize=12)
            plt.legend(loc='best', fontsize=9)

            # 保存图表
            plt.tight_layout()
            plt.savefig(f'seasonal_comparison_{metric}.png', dpi=300)
            plt.close()
            print(f"季节性{metric}对比图已保存为 seasonal_comparison_{metric}.png")

        print("\n创建月度指标对比可视化...")

        # 为每个指标创建一个月度对比折线图
        for metric in metrics_names:
            plt.figure(figsize=(14, 8))

            # 为每个模型绘制折线图
            for model_name in model_names:
                values = []

                for month in months:
                    if model_name == 'Proposed':
                        if month in monthly_metrics:
                            values.append(monthly_metrics[month][f'{metric}_pred'])
                        else:
                            values.append(np.nan)
                    elif model_name == 'ECMWF':
                        if month in monthly_metrics:
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

                # 设置不同的线型和标记
                if model_name == 'Proposed':
                    plt.plot(months, values, '-o', linewidth=2, markersize=8, label=model_name)
                elif model_name == 'ECMWF':
                    plt.plot(months, values, '--s', linewidth=2, markersize=8, label=model_name, color='red')
                else:
                    # 为baseline模型选择不同的标记
                    markers = ['^', 'v', '<', '>', 'x', 'd', 'p']
                    marker_idx = list(baseline_metrics.keys()).index(model_name) % len(markers)
                    plt.plot(months, values, '--', marker=markers[marker_idx], linewidth=1.5, markersize=7,
                             label=model_name)

            # 设置图表标题和标签
            plt.ylabel(metrics_display[metric], fontsize=14)
            plt.xticks(months, months_names, fontsize=12)
            plt.legend(loc='best', fontsize=9)

            # 保存图表
            plt.tight_layout()
            plt.savefig(f'monthly_comparison_{metric}.png', dpi=300)
            plt.close()
            print(f"月度{metric}对比图已保存为 monthly_comparison_{metric}.png")

        hours = ['00', '06', '12', '18']
        hour_indices = {hour: i for i, hour in enumerate(hours)}
        for hour in hours:
            hour_mask = (pd.to_datetime(all_dates).hour == int(hour))
            hour_dates = pd.to_datetime(all_dates[hour_mask])
            daily_preds = all_preds_orig[hour_mask].mean(axis=(1, 2))
            daily_targets = all_targets_orig[hour_mask].mean(axis=(1, 2))
            daily_tigge = all_tigge_wind_orig[hour_mask].mean(axis=(1, 2))

            days_of_year = hour_dates.dayofyear.values

            plt.figure(figsize=(12, 6))
            plt.plot(days_of_year, daily_targets, label='ERA5', color='orange')
            plt.plot(days_of_year, daily_preds, label='Predicted', color='blue')
            plt.plot(days_of_year, daily_tigge, label='ECMWF', color='green')
            plt.xlabel('Day of Year (2023)')
            plt.ylabel('Wind Speed (m/s)')
            plt.title(f'Wind Speed Comparison at {hour}:00 (2023)')
            plt.xticks(np.arange(0, 366, 50))
            plt.legend()
            plt.savefig(f'val_wind_speed_2023_{hour}.png')
            plt.close()
            print(f'Wind speed plot saved as val_wind_speed_{hour}.png')

        # === 修改点 2.4: 更新 metrics_names，添加 R2 可视化 ===
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
            plt.title(f'Validation {name} Comparison (Yearly and Seasonal)')
            plt.xticks(index, ['Yearly', 'Spring', 'Summer', 'Autumn', 'Winter'])
            plt.legend()
            plt.savefig(f'val_metrics_{name}.png')
            plt.close()
            print(f'Metrics plot saved as val_metrics_{name}.png')

        # === 新增: 计算并绘制空间指标分布 ===
        print("\n计算并绘制空间指标分布...")

        # 确保 all_preds_orig 和 all_targets_orig 的形状正确
        if len(all_preds_orig.shape) == 3:  # [N, H, W]
            spatial_metrics = calculate_spatial_metrics(
                all_preds_orig,
                all_targets_orig,
                all_tigge_wind_orig
            )

            # 华北地区的经纬度范围
            lat_range = (35.13, 47.0)
            lon_range = (103.0, 126.88)

            # 绘制空间指标分布图
            try:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                plot_spatial_metrics(spatial_metrics, lat_range, lon_range, save_dir='spatial_metrics')
                print("空间指标分布图已保存到 spatial_metrics 目录")
            except ImportError:
                print("警告: 缺少绘制空间分布图所需的库，请安装 mpl_toolkits")

            try:
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
                print(f"绘制论文风格空间分布图时出错: {str(e)}")
        else:
            # 如果形状不是[N, H, W]，尝试重塑数据
            try:
                # 假设数据形状为(1457, 48, 96)，我们需要确保它是[N, H, W]格式
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
                plot_spatial_metrics(spatial_metrics, lat_range, lon_range, save_dir='spatial_metrics')
                print("空间指标分布图已保存到 spatial_metrics 目录")

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

# 主函数（完成，与训练一致）
if __name__ == "__main__":
    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    batch_size = 16
    val_ds = WindDataset("./show_relevance_visualization/val.nc", H, W)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("Datasets loaded successfully")

    print("Initializing model...")
    model = WindSpeedPredictor(
        H, W,
        tigge_features=8,
        dropout_rate=0.24733185479083603,
        ltc_hidden_dim=216,
        cbam_reduction=16
    ).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    print("Model loaded successfully")

    print("Starting Validation...")
    validate_model(model, val_loader, device)
    print("Validation completed!")
