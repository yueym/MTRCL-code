import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import gc
import time


# 模型定义部分（与训练代码相同）
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


class WindSpeedPredictor(nn.Module):
    def __init__(self, H, W, tigge_features=8, dropout_rate=0.24733185479083603, ltc_hidden_dim=216, cbam_reduction=16):
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

    def forward(self, tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq):
        b = tigge_spatial.size(0)
        spatial_input = torch.cat([tigge_spatial, dem_spatial], dim=1)
        time_emb = self.time_embed(time_features_t).view(b, 56, 1, 1)
        resnet_out = self.resnet(spatial_input, time_emb)
        ltc_out = self.ltc(tigge_seq, time_features_seq)
        fused = self.gated_fusion(resnet_out, ltc_out)
        pred = self.mlp(fused)
        return pred.squeeze(1)

# 🔥 修正版数据集类 - 添加缺失的属性
class WindDatasetEmbedded(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading EMBEDDED dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset variables: {list(self.ds.data_vars.keys())}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset dimensions: {dict(self.ds.dims)}")

        # 🔥 关键添加：获取特征数量
        self.tigge_features_count = len(self.ds.coords['tigge_feature'])
        self.dem_features_count = len(self.ds.coords['dem_feature'])

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Selected TIGGE features (RandomForest): {list(self.ds.coords['tigge_feature'].values)}")
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Selected DEM features (RandomForest): {list(self.ds.coords['dem_feature'].values)}")

        # 🔥 关键修改：检查数据的实际形状
        sample_size = len(self.ds.coords['sample'])
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total samples in dataset: {sample_size}")

        # 检查是否是网格数据还是采样数据
        expected_grid_size = H * W  # 4608

        if sample_size % expected_grid_size == 0:
            # 数据是网格格式
            self.is_grid_data = True
            self.time_steps = sample_size // expected_grid_size
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset is in GRID format: {self.time_steps} timesteps × {expected_grid_size} spatial points")
        else:
            # 数据是采样格式，需要重新组织
            self.is_grid_data = False
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset is in SAMPLED format, will reshape to grid")

            # 🔥 关键：计算能够构成完整网格的时间步数
            self.time_steps = sample_size // expected_grid_size
            if self.time_steps == 0:
                # 如果样本太少，直接使用现有样本数作为"空间维度"
                self.H = int(np.sqrt(sample_size))
                self.W = sample_size // self.H
                self.time_steps = 1
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Too few samples, using {self.H}×{self.W} spatial grid")
            else:
                # 只使用能够构成完整网格的样本
                self.valid_samples = self.time_steps * expected_grid_size
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using {self.valid_samples}/{sample_size} samples for complete grids")

        # 处理时间特征
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normalizing time features")
        time_data = self.ds['time_features'].values
        if self.is_grid_data:
            # 网格数据：每个时间步重复H*W次
            unique_time_data = time_data[::expected_grid_size]  # 每隔4608个取一个
        else:
            if self.time_steps > 0:
                unique_time_data = time_data[:self.valid_samples:expected_grid_size]
            else:
                unique_time_data = time_data[:1]  # 只有一个时间步

        self.time_scaler = StandardScaler()
        normalized_unique_time = self.time_scaler.fit_transform(unique_time_data)

        # 创建时间索引
        if self.is_grid_data or self.time_steps > 0:
            times = pd.to_datetime({
                'year': unique_time_data[:, 0],
                'month': unique_time_data[:, 1],
                'day': unique_time_data[:, 2],
                'hour': unique_time_data[:, 3]
            })
        else:
            # 单个时间步的情况
            times = pd.to_datetime({
                'year': [unique_time_data[0, 0]],
                'month': [unique_time_data[0, 1]],
                'day': [unique_time_data[0, 2]],
                'hour': [unique_time_data[0, 3]]
            })

        self.time_points = times.values
        self.T = len(self.time_points)

        # 生成标准化的时间特征
        self.normalized_time_features = normalized_unique_time

        # 计算可用的样本索引
        self.sample_indices = np.arange(max(0, self.T - self.seq_len + 1))

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED dataset initialized:")
        print(f"  - Spatial size: {self.H}×{self.W}")
        print(f"  - Time steps: {self.T}")
        print(f"  - Valid sample indices: {len(self.sample_indices)}")
        print(f"  - Is grid data: {self.is_grid_data}")
        print(f"  - TIGGE features count: {self.tigge_features_count}")  # 🔥 添加调试信息
        print(f"  - DEM features count: {self.dem_features_count}")      # 🔥 添加调试信息

    def __len__(self):
        return max(1, len(self.sample_indices))  # 至少返回1

    def __getitem__(self, idx):
        if len(self.sample_indices) == 0:
            # 处理边界情况
            actual_idx = 0
            t = self.seq_len - 1
        else:
            actual_idx = self.sample_indices[idx % len(self.sample_indices)]
            t = actual_idx + self.seq_len - 1

        # 确保时间索引不超出范围
        t = min(t, self.T - 1)
        seq_start = max(0, t - self.seq_len + 1)

        seq_data = []
        time_features_seq = []

        # 构建序列数据
        for i in range(self.seq_len):
            time_idx = min(seq_start + i, self.T - 1)

            if self.is_grid_data:
                # 网格数据：直接按时间步和空间位置索引
                start_sample = time_idx * (self.H * self.W)
                end_sample = start_sample + (self.H * self.W)
                tigge_data = self.ds['X_tigge'].isel(sample=slice(start_sample, end_sample)).values
                tigge_data = tigge_data.reshape(self.H, self.W, self.tigge_features_count)  # 🔥 使用正确的特征数
            else:
                if self.time_steps > 0:
                    # 采样数据但有多个时间步
                    start_sample = time_idx * (self.H * self.W)
                    end_sample = start_sample + (self.H * self.W)
                    tigge_data = self.ds['X_tigge'].isel(sample=slice(start_sample, end_sample)).values
                    tigge_data = tigge_data.reshape(self.H, self.W, self.tigge_features_count)  # 🔥 使用正确的特征数
                else:
                    # 只有一个时间步，重复使用
                    tigge_data = self.ds['X_tigge'].values.reshape(self.H, self.W, self.tigge_features_count)  # 🔥 使用正确的特征数

            seq_data.append(tigge_data)
            time_features_seq.append(self.normalized_time_features[time_idx])

        tigge_seq = np.stack(seq_data)
        time_features_seq = np.stack(time_features_seq)

        # 当前时间步数据
        if self.is_grid_data:
            start_sample = t * (self.H * self.W)
            end_sample = start_sample + (self.H * self.W)
            tigge_spatial = self.ds['X_tigge'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W, self.tigge_features_count)  # 🔥 使用正确的特征数
            dem_spatial = self.ds['X_dem'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W, self.dem_features_count)    # 🔥 使用正确的特征数
            target = self.ds['y'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W)
        else:
            if self.time_steps > 0:
                start_sample = t * (self.H * self.W)
                end_sample = start_sample + (self.H * self.W)
                tigge_spatial = self.ds['X_tigge'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W, self.tigge_features_count)  # 🔥 使用正确的特征数
                dem_spatial = self.ds['X_dem'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W, self.dem_features_count)    # 🔥 使用正确的特征数
                target = self.ds['y'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W)
            else:
                tigge_spatial = self.ds['X_tigge'].values.reshape(self.H, self.W, self.tigge_features_count)  # 🔥 使用正确的特征数
                dem_spatial = self.ds['X_dem'].values.reshape(self.H, self.W, self.dem_features_count)        # 🔥 使用正确的特征数
                target = self.ds['y'].values.reshape(self.H, self.W)

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
    """计算所有指标，包括MAPE"""
    pred = pred.flatten()
    target = target.flatten()
    tigge_wind = tigge_wind.flatten()

    # 现有6个指标
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

    # R²计算
    ss_tot = torch.sum((target - mean_target) ** 2).item()
    ss_res_pred = torch.sum((target - pred) ** 2).item()
    ss_res_tigge = torch.sum((target - tigge_wind) ** 2).item()
    R2_pred = 1 - (ss_res_pred / ss_tot) if ss_tot > 0 else 0
    R2_tigge = 1 - (ss_res_tigge / ss_tot) if ss_tot > 0 else 0

    # 新增MAPE计算
    epsilon = 1e-8
    target_safe = torch.where(torch.abs(target) < epsilon, epsilon, target)
    MAPE_pred = torch.mean(torch.abs((pred - target) / target_safe)).item() * 100
    MAPE_tigge = torch.mean(torch.abs((tigge_wind - target) / target_safe)).item() * 100

    return {
        'FA_pred': FA_pred, 'RMSE_pred': RMSE_pred, 'MAE_pred': MAE_pred,
        'rRMSE_pred': rRMSE_pred, 'rMAE_pred': rMAE_pred, 'R_pred': R_pred,
        'R2_pred': R2_pred, 'MAPE_pred': MAPE_pred,
        'FA_tigge': FA_tigge, 'RMSE_tigge': RMSE_tigge, 'MAE_tigge': MAE_tigge,
        'rRMSE_tigge': rRMSE_tigge, 'rMAE_tigge': rMAE_tigge, 'R_tigge': R_tigge,
        'R2_tigge': R2_tigge, 'MAPE_tigge': MAPE_tigge
    }


def test_model(model, test_loader, device):
    """测试embedded模型"""
    model.eval()
    all_preds = []
    all_targets = []
    all_tigge_wind = []

    # 🔥 关键修改：加载embedded标准化器
    scaler_target = joblib.load('./show_relevance_visualization_embedded/target_scaler_embedded.pkl')
    scaler_tigge = joblib.load('./show_relevance_visualization_embedded/tigge_feature_scaler_embedded.pkl')

    target_data_min = scaler_target.data_min_[0]
    target_range = 1 / scaler_target.scale_[0]

    # 🔥 关键修改：tigge_wind_speed在原始27个特征中始终是索引26
    tigge_data_min = scaler_tigge.data_min_[26]
    tigge_range = 1 / scaler_tigge.scale_[26]

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            tigge_spatial = batch['tigge_spatial'].to(device)
            dem_spatial = batch['dem_spatial'].to(device)
            tigge_seq = batch['tigge_seq'].to(device)
            time_features_t = batch['time_features_t'].to(device)
            time_features_seq = batch['time_features_seq'].to(device)
            target = batch['target'].to(device)

            output = model(tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq)

            all_preds.append(output.cpu())
            all_targets.append(target.cpu())
            # 🔥 关键修改：tigge_wind_speed在wrapper选择的8个特征中是索引7（最后一个）
            all_tigge_wind.append(tigge_spatial[:, 0, :, :].cpu())

    # 合并结果
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_tigge_wind = torch.cat(all_tigge_wind, dim=0).numpy()

    # 反标准化
    all_preds_orig = (all_preds * target_range) + target_data_min
    all_targets_orig = (all_targets * target_range) + target_data_min
    all_tigge_wind_orig = (all_tigge_wind * tigge_range) + tigge_data_min

    all_preds_orig = np.clip(all_preds_orig, 0, 100)
    all_targets_orig = np.clip(all_targets_orig, 0, 100)
    all_tigge_wind_orig = np.clip(all_tigge_wind_orig, 0, 100)

    # 保存结果文件，添加_embedded后缀
    np.save('test_preds_embedded.npy', all_preds_orig)
    np.save('test_targets_embedded.npy', all_targets_orig)
    np.save('test_tigge_wind_embedded.npy', all_tigge_wind_orig)

    # 计算指标
    all_preds_tensor = torch.from_numpy(all_preds_orig).float()
    all_targets_tensor = torch.from_numpy(all_targets_orig).float()
    all_tigge_wind_tensor = torch.from_numpy(all_tigge_wind_orig).float()

    metrics = calculate_metrics(all_preds_tensor, all_targets_tensor, all_tigge_wind_tensor)

    # 控制台输出指标
    print("\n===== Embedded Method Test Results =====")
    print(f"FA (%): {metrics['FA_pred']:.4f}")
    print(f"RMSE (m/s): {metrics['RMSE_pred']:.4f}")
    print(f"MAE (m/s): {metrics['MAE_pred']:.4f}")
    print(f"rRMSE (%): {metrics['rRMSE_pred']:.4f}")
    print(f"rMAE (%): {metrics['rMAE_pred']:.4f}")
    print(f"R: {metrics['R_pred']:.4f}")
    print(f"R²: {metrics['R2_pred']:.4f}")
    print(f"MAPE (%): {metrics['MAPE_pred']:.4f}")


if __name__ == "__main__":
    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 🔥 关键修改：加载embedded测试数据集
    batch_size = 16
    test_ds = WindDatasetEmbedded("./show_relevance_visualization_embedded/test_embedded.nc", H, W)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 初始化模型
    model = WindSpeedPredictor(
        H, W,
        tigge_features=test_ds.tigge_features_count,
        dropout_rate=0.24733185479083603,
        ltc_hidden_dim=216,
        cbam_reduction=16
    ).to(device)

    # 🔥 关键修改：加载embedded训练好的模型
    model.load_state_dict(torch.load('checkpoints_embedded/best_model_embedded.pth', map_location=device))
    print("Embedded model loaded successfully")

    # 开始测试
    test_model(model, test_loader, device)
    print("Embedded method testing completed!")