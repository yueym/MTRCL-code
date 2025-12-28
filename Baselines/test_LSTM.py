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


# 添加在导入语句后
def count_conv2d_flops(input_shape, weight_shape, stride=1, padding=0):
    """计算Conv2d层的FLOPs"""
    batch_size, in_channels, input_height, input_width = input_shape
    out_channels, in_channels, kernel_height, kernel_width = weight_shape

    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    flops_per_output = kernel_height * kernel_width * in_channels * 2
    total_flops = batch_size * out_channels * output_height * output_width * flops_per_output

    return total_flops


def count_linear_flops(input_shape, weight_shape):
    """计算Linear层的FLOPs"""
    batch_size = input_shape[0]
    in_features, out_features = weight_shape
    flops = batch_size * out_features * in_features * 2
    return flops


def estimate_lstm_flops(model, input_sample):
    """估算LSTM模型的FLOPs"""
    tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq = input_sample
    batch_size, seq_len, tigge_channels, H, W = tigge_seq.shape
    time_features = time_features_seq.shape[2]

    total_flops = 0

    print(f"Calculating FLOPs for LSTM model...")
    print(f"Input shape - Batch: {batch_size}, Seq: {seq_len}, Channels: {tigge_channels}, H: {H}, W: {W}")

    # LSTM部分的FLOPs计算
    print(f"Calculating LSTM FLOPs...")

    # LSTM输入维度计算
    lstm_input_dim = tigge_channels * H * W + time_features * H * W
    lstm_hidden_dim = 128
    lstm_layers = 2

    # LSTM的FLOPs计算 (每个时间步)
    # 对于每个LSTM层: 4个门 × (input×hidden + hidden×hidden + bias)
    for layer in range(lstm_layers):
        input_dim = lstm_input_dim if layer == 0 else lstm_hidden_dim

        # 输入到隐藏状态的变换: 4个门 × (input_dim × hidden_dim)
        input_to_hidden_flops = 4 * input_dim * lstm_hidden_dim * 2  # *2 for multiply-add

        # 隐藏状态到隐藏状态的变换: 4个门 × (hidden_dim × hidden_dim)
        hidden_to_hidden_flops = 4 * lstm_hidden_dim * lstm_hidden_dim * 2

        # 每个时间步的FLOPs
        timestep_flops = input_to_hidden_flops + hidden_to_hidden_flops

        # 乘以序列长度和batch大小
        layer_flops = timestep_flops * seq_len * batch_size
        total_flops += layer_flops

        print(f"  LSTM Layer {layer + 1}: {layer_flops:,} FLOPs")
        print(f"    - Input dim: {input_dim}, Hidden dim: {lstm_hidden_dim}")
        print(f"    - Per timestep: {timestep_flops:,}, Total: {layer_flops:,}")

    # 空间特征处理器的FLOPs
    print(f"Calculating Spatial Processor FLOPs...")

    # TIGGE处理器: 8 -> 32 (hidden_dim//2), kernel=3x3
    flops = count_conv2d_flops((batch_size, 8, H, W), (32, 8, 3, 3), padding=1)
    total_flops += flops
    print(f"  TIGGE processor: {flops:,} FLOPs")

    # DEM处理器: 3 -> 16 (hidden_dim//4), kernel=3x3
    flops = count_conv2d_flops((batch_size, 3, H, W), (16, 3, 3, 3), padding=1)
    total_flops += flops
    print(f"  DEM processor: {flops:,} FLOPs")

    # 融合层: (32+16) -> 64, kernel=1x1
    flops = count_conv2d_flops((batch_size, 48, H, W), (64, 48, 1, 1))
    total_flops += flops
    print(f"  Fusion layer: {flops:,} FLOPs")

    # 门控融合的FLOPs
    print(f"Calculating Gated Fusion FLOPs...")

    # LSTM投影: 128 -> 64, kernel=1x1
    flops = count_conv2d_flops((batch_size, 128, H, W), (64, 128, 1, 1))
    total_flops += flops

    # 空间投影: 64 -> 64, kernel=1x1
    flops = count_conv2d_flops((batch_size, 64, H, W), (64, 64, 1, 1))
    total_flops += flops

    # 门控计算: 128 -> 64 -> 64
    flops = count_conv2d_flops((batch_size, 128, H, W), (64, 128, 1, 1))
    total_flops += flops
    flops = count_conv2d_flops((batch_size, 64, H, W), (64, 64, 1, 1))
    total_flops += flops

    # 最终融合: 128 -> 64, kernel=3x3
    flops = count_conv2d_flops((batch_size, 128, H, W), (64, 128, 3, 3), padding=1)
    total_flops += flops

    print(f"  Gated Fusion total: {flops * 5:,} FLOPs")

    # MLP部分
    print(f"Calculating MLP FLOPs...")
    mlp_layers = [(64, 512), (512, 384), (384, 192), (192, 1)]
    current_shape = (batch_size, 64, H, W)

    for i, (in_ch, out_ch) in enumerate(mlp_layers):
        flops = count_conv2d_flops(current_shape, (out_ch, in_ch, 1, 1))
        total_flops += flops
        print(f"  MLP layer {i + 1} ({in_ch}->{out_ch}): {flops:,} FLOPs")
        current_shape = (batch_size, out_ch, H, W)

    return total_flops


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params



class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.4):
        super(LSTMFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # 初始化LSTM权重
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x_seq, time_features_seq):
        """
        x_seq: (batch_size, seq_len, channels, H, W)
        time_features_seq: (batch_size, seq_len, time_features)
        """
        batch_size, seq_len, channels, H, W = x_seq.size()

        # 确保张量是连续的，然后进行reshape
        x_seq = x_seq.contiguous()
        time_features_seq = time_features_seq.contiguous()

        # 将空间维度展平并与时间特征结合
        x_flat = x_seq.reshape(batch_size, seq_len, -1)  # 使用reshape替代view

        # 扩展时间特征到空间维度
        time_expanded = time_features_seq.unsqueeze(2).expand(-1, -1, H * W,
                                                              -1)  # (batch_size, seq_len, H*W, time_features)
        time_flat = time_expanded.reshape(batch_size, seq_len, -1)  # 使用reshape替代view

        # 组合空间特征和时间特征
        combined_features = torch.cat([x_flat, time_flat], dim=-1)  # (batch_size, seq_len, total_features)

        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(combined_features)

        # 使用最后一个时间步的输出
        final_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # 将输出重新reshape为空间形式
        output = final_output.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (batch_size, hidden_dim, H, W)

        return output


class SpatialFeatureProcessor(nn.Module):
    def __init__(self, tigge_features, dem_features, hidden_dim, dropout_rate=0.4):
        super(SpatialFeatureProcessor, self).__init__()

        # 处理TIGGE空间特征
        self.tigge_processor = nn.Sequential(
            nn.Conv2d(tigge_features, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # 处理DEM特征
        self.dem_processor = nn.Sequential(
            nn.Conv2d(dem_features, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim // 2 + hidden_dim // 4, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, tigge_spatial, dem_spatial):
        tigge_features = self.tigge_processor(tigge_spatial)
        dem_features = self.dem_processor(dem_spatial)

        combined = torch.cat([tigge_features, dem_features], dim=1)
        output = self.fusion(combined)

        return output


class GatedFusion(nn.Module):
    def __init__(self, lstm_dim, spatial_dim, output_dim):
        super(GatedFusion, self).__init__()

        # 确保维度匹配
        self.lstm_proj = nn.Conv2d(lstm_dim, output_dim, kernel_size=1)
        self.spatial_proj = nn.Conv2d(spatial_dim, output_dim, kernel_size=1)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(output_dim * 2, output_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(output_dim * 2, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, lstm_out, spatial_out):
        # 投影到相同维度
        lstm_proj = self.lstm_proj(lstm_out)
        spatial_proj = self.spatial_proj(spatial_out)

        # 计算门控权重
        combined = torch.cat([lstm_proj, spatial_proj], dim=1)
        gate_weights = self.gate(combined)

        # 门控融合
        gated_lstm = gate_weights * lstm_proj
        gated_spatial = (1 - gate_weights) * spatial_proj

        # 最终融合
        final_combined = torch.cat([gated_lstm, gated_spatial], dim=1)
        output = self.final_fusion(final_combined)

        return output


class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.4):
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


class LSTMWindSpeedPredictor(nn.Module):
    def __init__(self, H, W, tigge_features=8, dem_features=3, time_features=5,
                 lstm_hidden_dim=128, spatial_hidden_dim=64, fusion_dim=64,
                 lstm_layers=2, dropout_rate=0.4):
        super(LSTMWindSpeedPredictor, self).__init__()
        self.H = H
        self.W = W

        # 计算LSTM输入维度
        lstm_input_dim = tigge_features * H * W + time_features * H * W

        # LSTM特征提取器
        self.lstm_extractor = LSTMFeatureExtractor(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout_rate=dropout_rate
        )

        # 空间特征处理器
        self.spatial_processor = SpatialFeatureProcessor(
            tigge_features=tigge_features,
            dem_features=dem_features,
            hidden_dim=spatial_hidden_dim,
            dropout_rate=dropout_rate
        )

        # 门控融合
        self.gated_fusion = GatedFusion(
            lstm_dim=lstm_hidden_dim,
            spatial_dim=spatial_hidden_dim,
            output_dim=fusion_dim
        )

        # 最终预测层
        self.mlp = MLP(fusion_dim, dropout_rate=dropout_rate)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self, tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq):
        # LSTM处理时序特征
        lstm_features = self.lstm_extractor(tigge_seq, time_features_seq)

        # 空间特征处理
        spatial_features = self.spatial_processor(tigge_spatial, dem_spatial)

        # 特征融合
        fused_features = self.gated_fusion(lstm_features, spatial_features)

        # 最终预测
        pred = self.mlp(fused_features)

        return pred.squeeze(1)


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

    # MAPE计算
    epsilon = 1e-8
    target_safe = torch.where(torch.abs(target) < epsilon, epsilon, target)
    MAPE_pred = torch.mean(torch.abs((pred - target) / target_safe)).item() * 100
    MAPE_tigge = torch.mean(torch.abs((tigge_wind - target) / target_safe)).item() * 100

    return {
        'FA_pred': FA_pred, 'RMSE_pred': RMSE_pred, 'MAE_pred': MAE_pred,
        'rRMSE_pred': rRMSE_pred, 'rMAE_pred': rMAE_pred, 'R_pred': R_pred, 'R2_pred': R2_pred,
        'FA_tigge': FA_tigge, 'RMSE_tigge': RMSE_tigge, 'MAE_tigge': MAE_tigge,
        'rRMSE_tigge': rRMSE_tigge, 'rMAE_tigge': rMAE_tigge, 'R_tigge': R_tigge, 'R2_tigge': R2_tigge,
        'MAPE_pred': MAPE_pred, 'MAPE_tigge': MAPE_tigge
    }


def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_tigge_wind = []
    all_dates = []
    criterion = nn.SmoothL1Loss()

    # 加载标准化器
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
                actual_indices = test_loader.dataset.sample_indices[batch_idx * test_loader.batch_size:
                                                                    (batch_idx + 1) * test_loader.batch_size]
                batch_dates = [test_loader.dataset.time_points[idx + test_loader.dataset.seq_len - 1] for idx in
                               actual_indices
                               if idx + test_loader.dataset.seq_len - 1 < len(test_loader.dataset.time_points)]
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

        # 反标准化
        all_preds_orig = (all_preds * target_range) + target_data_min
        all_targets_orig = (all_targets * target_range) + target_data_min
        all_tigge_wind_orig = (all_tigge_wind * tigge_range) + tigge_data_min
        all_preds_orig = np.clip(all_preds_orig, 0, 100)
        all_targets_orig = np.clip(all_targets_orig, 0, 100)
        all_tigge_wind_orig = np.clip(all_tigge_wind_orig, 0, 100)

        # 保存LSTM专用文件
        np.save('LSTM_test_preds.npy', all_preds_orig)
        np.save('LSTM_test_targets.npy', all_targets_orig)
        np.save('LSTM_test_tigge_wind.npy', all_tigge_wind_orig)
        np.save('LSTM_test_dates.npy', all_dates)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test predictions, targets, TIGGE wind, and dates saved as LSTM_test_*.npy files.")

        # 计算年度指标
        metrics_all = calculate_metrics(
            torch.from_numpy(all_preds_orig),
            torch.from_numpy(all_targets_orig),
            torch.from_numpy(all_tigge_wind_orig)
        )

        # ===== 新增：清晰输出7项测试指标 =====
        print(f"\n{'=' * 80}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LSTM MODEL - FINAL TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"FA (%)     : {metrics_all['FA_pred']:.2f}")
        print(f"RMSE (m/s) : {metrics_all['RMSE_pred']:.4f}")
        print(f"MAE (m/s)  : {metrics_all['MAE_pred']:.4f}")
        print(f"rRMSE (%)  : {metrics_all['rRMSE_pred']:.2f}")
        print(f"rMAE (%)   : {metrics_all['rMAE_pred']:.2f}")
        print(f"R          : {metrics_all['R_pred']:.4f}")
        print(f"MAPE (%)   : {metrics_all['MAPE_pred']:.2f}")
        print(f"{'=' * 80}\n")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Yearly Test Metrics (LSTM):", metrics_all)

        # 计算季度指标
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
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {season} Test Metrics (LSTM):",
                  seasonal_metrics[season])

        # 保存年度和季度指标
        all_metrics = {'Yearly': metrics_all, **seasonal_metrics}
        with open('LSTM_test_metrics.json', 'w') as f:
            json.dump(all_metrics, f)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test metrics saved as LSTM_test_metrics.json")

        # 计算月度指标
        monthly_metrics = {}
        months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
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
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Month {months_names[month - 1]} Test Metrics (LSTM):",
                    monthly_metrics[month])

        # 保存月度指标
        monthly_metrics_dict = {str(k): v for k, v in monthly_metrics.items()}
        with open('LSTM_test_monthly_metrics.json', 'w') as f:
            json.dump(monthly_metrics_dict, f)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monthly test metrics saved as LSTM_test_monthly_metrics.json")

if __name__ == "__main__":
    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets...")
    batch_size = 8  # 与训练时保持一致的batch_size
    test_ds = WindDataset("./show_relevance_visualization/test.nc", H, W)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Datasets loaded successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing LSTM model...")
    model = LSTMWindSpeedPredictor(
        H=H, W=W,
        tigge_features=8,
        dem_features=3,
        time_features=5,
        lstm_hidden_dim=128,
        spatial_hidden_dim=64,
        fusion_dim=64,
        lstm_layers=2,
        dropout_rate=0.4
    ).to(device)

    # 加载LSTM训练的最佳模型
    model.load_state_dict(torch.load('checkpoints/best_model_lstm.pth'))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LSTM model loaded successfully")

    # 获取一个样本用于计算
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Preparing sample for GFLOPs and forecast time calculation...")
    sample = next(iter(test_loader))
    tigge_spatial_sample = sample['tigge_spatial'][0:1].to(device)
    dem_spatial_sample = sample['dem_spatial'][0:1].to(device)
    tigge_seq_sample = sample['tigge_seq'][0:1].to(device)
    time_features_t_sample = sample['time_features_t'][0:1].to(device)
    time_features_seq_sample = sample['time_features_seq'][0:1].to(device)

    # 计算FLOPs
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== CALCULATING MODEL COMPLEXITY =====")
    model.eval()

    input_sample = (
    tigge_spatial_sample, dem_spatial_sample, tigge_seq_sample, time_features_t_sample, time_features_seq_sample)
    total_flops = estimate_lstm_flops(model, input_sample)
    gflops = total_flops / 1e9

    # 计算参数量
    total_params, trainable_params = count_parameters(model)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MODEL COMPLEXITY ANALYSIS =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Estimated GFLOPs: {gflops:.6f} GFLOPs")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total FLOPs: {total_flops:,} FLOPs")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total parameters: {total_params:,} parameters")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trainable parameters: {trainable_params:,} parameters")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Non-trainable parameters: {total_params - trainable_params:,} parameters")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model size: {total_params * 4 / (1024 ** 2):.2f} MB (float32)")

    # 计算预测时间
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MEASURING FORECAST TIME =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warming up model...")

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(tigge_spatial_sample, dem_spatial_sample, tigge_seq_sample, time_features_t_sample,
                      time_features_seq_sample)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting inference time measurement...")

    # 测量推理时间
    num_runs = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(tigge_spatial_sample, dem_spatial_sample, tigge_seq_sample, time_features_t_sample,
                      time_features_seq_sample)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # 计算预测时间
    forecast_time_ms = (end_time - start_time) * 1000 / num_runs
    forecast_time_us = forecast_time_ms * 1000
    forecast_time_s = forecast_time_ms / 1000

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== FORECAST TIME ANALYSIS =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Average forecast time: {forecast_time_ms:.4f} ms per sample")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Average forecast time: {forecast_time_us:.2f} μs per sample")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Average forecast time: {forecast_time_s:.6f} s per sample")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Throughput: {1000 / forecast_time_ms:.2f} samples/second")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Throughput: {3600000 / forecast_time_ms:.0f} samples/hour")

    # 计算单个网格点的预测时间
    total_grid_points = H * W
    grid_forecast_time_ms = forecast_time_ms / total_grid_points
    grid_forecast_time_us = forecast_time_us / total_grid_points
    grid_forecast_time_ns = grid_forecast_time_us * 1000

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== PER GRID POINT ANALYSIS =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Forecast time per grid point: {grid_forecast_time_ms:.6f} ms")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Forecast time per grid point: {grid_forecast_time_us:.3f} μs")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Forecast time per grid point: {grid_forecast_time_ns:.1f} ns")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total grid points: {total_grid_points} points")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Grid resolution: {H} × {W}")

    # 计算内存使用情况
    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        memory_used_gb = memory_used / 1024  # GB
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MEMORY USAGE =====")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Peak GPU Memory used: {memory_used:.2f} MB")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Peak GPU Memory used: {memory_used_gb:.3f} GB")

    # 计算计算强度
    compute_intensity = gflops / forecast_time_s  # GFLOPs per second

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== COMPUTATIONAL EFFICIENCY =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computational intensity: {compute_intensity:.2f} GFLOPS/s")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FLOPs per parameter: {total_flops / total_params:.2f}")

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== SUMMARY FOR RESEARCH PAPER =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model: ResNet")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GFLOPs: {gflops:.3f}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Forecast time: {forecast_time_ms:.2f} ms")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Parameters: {total_params / 1e6:.2f}M")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model size: {total_params * 4 / (1024 ** 2):.1f} MB")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Testing...")
    test_model(model, test_loader, device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing completed!")