import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import time
import gc
from torch.cuda.amp import autocast, GradScaler


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# CBAM模块
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


# ResNetCBAM模块
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


# ODEFunc模块
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


# LTC模块
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


# GatedFusion模块
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


# MLP模块
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


# 完整模型
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


# 🔥 修正版数据集类 - 解决reshape错误
class WindDatasetEmbedded(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading EMBEDDED dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset variables: {list(self.ds.data_vars.keys())}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset dimensions: {dict(self.ds.dims)}")

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
                tigge_data = tigge_data.reshape(self.H, self.W, -1)
            else:
                if self.time_steps > 0:
                    # 采样数据但有多个时间步
                    start_sample = time_idx * (self.H * self.W)
                    end_sample = start_sample + (self.H * self.W)
                    tigge_data = self.ds['X_tigge'].isel(sample=slice(start_sample, end_sample)).values
                    tigge_data = tigge_data.reshape(self.H, self.W, -1)
                else:
                    # 只有一个时间步，重复使用
                    tigge_data = self.ds['X_tigge'].values.reshape(self.H, self.W, -1)

            seq_data.append(tigge_data)
            time_features_seq.append(self.normalized_time_features[time_idx])

        tigge_seq = np.stack(seq_data)
        time_features_seq = np.stack(time_features_seq)

        # 当前时间步数据
        if self.is_grid_data:
            start_sample = t * (self.H * self.W)
            end_sample = start_sample + (self.H * self.W)
            tigge_spatial = self.ds['X_tigge'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H,
                                                                                                           self.W, -1)
            dem_spatial = self.ds['X_dem'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W,
                                                                                                       -1)
            target = self.ds['y'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W)
        else:
            if self.time_steps > 0:
                start_sample = t * (self.H * self.W)
                end_sample = start_sample + (self.H * self.W)
                tigge_spatial = self.ds['X_tigge'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H,
                                                                                                               self.W,
                                                                                                               -1)
                dem_spatial = self.ds['X_dem'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H,
                                                                                                           self.W, -1)
                target = self.ds['y'].isel(sample=slice(start_sample, end_sample)).values.reshape(self.H, self.W)
            else:
                tigge_spatial = self.ds['X_tigge'].values.reshape(self.H, self.W, -1)
                dem_spatial = self.ds['X_dem'].values.reshape(self.H, self.W, -1)
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


# 训练函数 - 修改保存文件名，添加_embedded后缀
def train_model_embedded(model, train_loader, val_loader, device,
                         epochs=12, learning_rate=0.0003931296040961602, weight_decay=0.0001025927948297095,
                         patience=5, accumulation_steps=1):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started EMBEDDED method training process")
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_mae = []
    train_rmse = []
    train_r2 = []
    val_mae = []
    val_rmse = []
    val_r2 = []
    patience_counter = 0

    checkpoint_dir = 'checkpoints_embedded'
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Embedded checkpoint directory created: {checkpoint_dir}")

    try:
        first_batch = next(iter(train_loader))
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - tigge_spatial shape: {first_batch['tigge_spatial'].shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - dem_spatial shape: {first_batch['dem_spatial'].shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - tigge_seq shape: {first_batch['tigge_seq'].shape}")
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_t shape: {first_batch['time_features_t'].shape}")
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_seq shape: {first_batch['time_features_seq'].shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - target shape: {first_batch['target'].shape}")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning - Could not load first batch for debugging: {e}")

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch + 1}/{epochs} (EMBEDDED method)")
        model.train()
        train_loss = 0.0
        batch_count = 0
        train_preds = []
        train_targets = []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            try:
                tigge_spatial = batch['tigge_spatial'].to(device)
                dem_spatial = batch['dem_spatial'].to(device)
                tigge_seq = batch['tigge_seq'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                time_features_seq = batch['time_features_seq'].to(device)
                target = batch['target'].to(device)

                with autocast():
                    output = model(tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq)
                    loss = criterion(output, target)

                train_preds.append(output.detach().cpu())
                train_targets.append(target.detach().cpu())

                scaler.scale(loss / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if batch_idx % 100 == 0:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gradient norm: {grad_norm:.4f}")

                train_loss += loss.item() * tigge_spatial.size(0)
                batch_count += 1
                if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
                    batch_time = time.time() - batch_start
                    progress = (batch_idx + 1) / len(train_loader) * 100
                    eta = (len(train_loader) - batch_idx - 1) * batch_time / 60
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1}/{epochs}, "
                      f"Batch {batch_idx + 1}/{len(train_loader)} ({progress:.1f}%), "
                      f"Loss: {loss.item():.6f}, Batch time: {batch_time:.2f}s, ETA: {eta:.2f} min")
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing batch {batch_idx}: {str(e)}")
                continue

        train_loss = train_loss / len(train_loader.dataset) if batch_count > 0 else float('inf')
        train_losses.append(train_loss)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing training metrics")
        train_preds = torch.cat(train_preds).numpy().flatten()
        train_targets = torch.cat(train_targets).numpy().flatten()
        train_mae_val = np.mean(np.abs(train_preds - train_targets))
        train_rmse_val = np.sqrt(np.mean((train_preds - train_targets) ** 2))
        train_mean = np.mean(train_targets)
        ss_tot = np.sum((train_targets - train_mean) ** 2)
        ss_res = np.sum((train_targets - train_preds) ** 2)
        train_r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        train_mae.append(train_mae_val)
        train_rmse.append(train_rmse_val)
        train_r2.append(train_r2_val)
        print(
            f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, MAE: {train_mae_val:.4f}, RMSE: {train_rmse_val:.4f}, R²: {train_r2_val:.4f}')

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting validation")
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    tigge_spatial = batch['tigge_spatial'].to(device)
                    dem_spatial = batch['dem_spatial'].to(device)
                    tigge_seq = batch['tigge_seq'].to(device)
                    time_features_t = batch['time_features_t'].to(device)
                    time_features_seq = batch['time_features_seq'].to(device)
                    target = batch['target'].to(device)
                    with autocast():
                        output = model(tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq)
                        loss = criterion(output, target)
                    val_preds.append(output.detach().cpu())
                    val_targets.append(target.detach().cpu())
                    val_loss += loss.item() * tigge_spatial.size(0)
                    val_batch_count += 1
                    if batch_idx % 100 == 0 or batch_idx == len(val_loader) - 1:
                        progress = (batch_idx + 1) / len(val_loader) * 100
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation progress: {progress:.1f}%")

                except Exception as e:
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing validation batch {batch_idx}: {str(e)}")
                    continue

        val_loss = val_loss / len(val_loader.dataset) if val_batch_count > 0 else float('inf')
        val_losses.append(val_loss)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing validation metrics")
        val_preds = torch.cat(val_preds).numpy().flatten()
        val_targets = torch.cat(val_targets).numpy().flatten()
        val_mae_val = np.mean(np.abs(val_preds - val_targets))
        val_rmse_val = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        val_mean = np.mean(val_targets)
        ss_tot = np.sum((val_targets - val_mean) ** 2)
        ss_res = np.sum((val_targets - val_preds) ** 2)
        val_r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        val_mae.append(val_mae_val)
        val_rmse.append(val_rmse_val)
        val_r2.append(val_r2_val)
        print(
            f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Validation Loss: {val_loss:.6f}, MAE: {val_mae_val:.4f}, RMSE: {val_rmse_val:.4f}, R²: {val_r2_val:.4f}')

        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model_embedded.pth'))
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] EMBEDDED model saved with val loss: {best_val_loss:.6f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Early stopping at epoch {epoch + 1}')
                break

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving EMBEDDED checkpoint for epoch {epoch + 1}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}_embedded.pth'))

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        remaining_epochs = epochs - (epoch + 1)
        est_remaining_time = remaining_epochs * epoch_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1} completed in {epoch_time / 60:.2f} minutes")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Estimated time remaining: {est_remaining_time / 3600:.2f} hours")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED method training completed in {total_time / 3600:.2f} hours")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating EMBEDDED visualization plots")

    # 所有图表文件名都添加 _embedded 后缀
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Embedded Method)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_embedded.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED loss curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_mae) + 1), val_mae, label='Validation MAE', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error (Embedded Method)')
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_curve_embedded.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED MAE curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label='Validation RMSE', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error (Embedded Method)')
    plt.legend()
    plt.grid(True)
    plt.savefig('rmse_curve_embedded.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED RMSE curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_r2) + 1), train_r2, label='Train R²', color='blue', linewidth=2)
    plt.plot(range(1, len(val_r2) + 1), val_r2, label='Validation R²', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R² Score (Embedded Method)')
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_curve_embedded.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED R² curve saved")

    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] EMBEDDED training metrics visualizations saved as:')
    print('  - loss_curve_embedded.png')
    print('  - mae_curve_embedded.png')
    print('  - rmse_curve_embedded.png')
    print('  - r2_curve_embedded.png')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'best_val_loss': best_val_loss
    }


if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED Method Training Program started")

    set_seed(42)

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Setting up EMBEDDED training parameters")
    batch_size = 16
    epochs = 12
    learning_rate = 0.0003931296040961602
    weight_decay = 0.0001025927948297095
    patience = 5
    accumulation_steps = 1

    # 使用embedded数据集路径
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading EMBEDDED training dataset")
    try:
        train_ds = WindDatasetEmbedded("./show_relevance_visualization_embedded/train_embedded.nc", H, W)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading EMBEDDED validation dataset")
        val_ds = WindDatasetEmbedded("./show_relevance_visualization_embedded/val_embedded.nc", H, W)
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error loading datasets: {e}")
        print("Please ensure that the embedded feature selection has been completed successfully.")
        exit(1)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating EMBEDDED data loaders")
    # 使用 num_workers=0 避免多进程问题
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED data loaders created successfully")

    # 获取RandomForest选择后的特征数量
    try:
        tigge_features_count = len(train_ds.ds.coords['tigge_feature'])
        dem_features_count = len(train_ds.ds.coords['dem_feature'])
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED method using {tigge_features_count} TIGGE features and {dem_features_count} DEM features (selected by RandomForest)")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error getting feature counts: {e}")
        # 设置默认值
        tigge_features_count = 8
        dem_features_count = 3
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using default feature counts: {tigge_features_count} TIGGE, {dem_features_count} DEM")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing EMBEDDED model")
    model = WindSpeedPredictor(
        H, W,
        tigge_features=tigge_features_count,  # 使用RandomForest选择后的特征数量
        dropout_rate=0.24733185479083603,
        ltc_hidden_dim=216,
        cbam_reduction=16
    ).to(device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED model initialized successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting EMBEDDED Method Training")
    training_results = train_model_embedded(
        model, train_loader, val_loader, device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        accumulation_steps=accumulation_steps
    )

    # 保存训练结果到JSON文件，添加_embedded后缀
    import json

    with open('training_results_embedded.json', 'w') as f:
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj


        json.dump(training_results, f, indent=2, default=convert_numpy)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED training results saved to training_results_embedded.json")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EMBEDDED Method Training completed!")

    # 打印最终结果摘要
    print(f"\n{'=' * 60}")
    print(f"EMBEDDED METHOD TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Feature Selection Method: RandomForest Feature Importance")
    print(f"Embedded Algorithm: RandomForestRegressor")
    print(f"Best Validation Loss: {training_results['best_val_loss']:.6f}")
    print(f"Final Train MAE: {training_results['train_mae'][-1]:.4f}")
    print(f"Final Train RMSE: {training_results['train_rmse'][-1]:.4f}")
    print(f"Final Train R²: {training_results['train_r2'][-1]:.4f}")
    print(f"Final Val MAE: {training_results['val_mae'][-1]:.4f}")
    print(f"Final Val RMSE: {training_results['val_rmse'][-1]:.4f}")
    print(f"Final Val R²: {training_results['val_r2'][-1]:.4f}")
    print(f"TIGGE Features Used (RandomForest): {tigge_features_count}")
    print(f"DEM Features Used (RandomForest): {dem_features_count}")

    try:
        print(f"Selected TIGGE Features: {list(train_ds.ds.coords['tigge_feature'].values)}")
        print(f"Selected DEM Features: {list(train_ds.ds.coords['dem_feature'].values)}")
    except:
        print("Selected features info not available")

    print(f"{'=' * 60}")

    print(f"\nGenerated EMBEDDED files:")
    print(f"Models:")
    print(f"  - checkpoints_embedded/best_model_embedded.pth")
    print(f"  - checkpoints_embedded/checkpoint_epoch_*_embedded.pth")
    print(f"Visualizations:")
    print(f"  - loss_curve_embedded.png")
    print(f"  - mae_curve_embedded.png")
    print(f"  - rmse_curve_embedded.png")
    print(f"  - r2_curve_embedded.png")
    print(f"Results:")
    print(f"  - training_results_embedded.json")

    print(f"\nEmbedded Method Details:")
    print(f"  ✅ RandomForest Feature Importance")
    print(f"  🌲 Based on tree-based feature selection")
    print(f"  🔄 Training and feature selection integrated")
    print(f"  ⚡ Efficient computation with bootstrap sampling")
    print(f"  📊 Feature importance from Gini impurity reduction")
    print(f"  🎯 Out-of-bag scoring for model validation")
    print(f"  🔧 Robust handling of sampled data structure")

