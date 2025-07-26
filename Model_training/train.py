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
    torch.backends.cudnn.benchmark = False  # 与优化阶段一致，禁用 benchmark

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
    def __init__(self, input_dim=8, hidden_dim=216, output_dim=216, seq_len=4, dt=6.0, dropout_rate=0.24733185479083603):
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
        time_features_seq = time_features_seq.unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1, 1).reshape(b * H * W, self.seq_len, 5)
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
        self.ltc = LTC(input_dim=tigge_features, hidden_dim=ltc_hidden_dim, output_dim=ltc_hidden_dim, dropout_rate=dropout_rate)
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

# 数据集类（无变化）
class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)

        # 检查数据集结构
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset variables: {list(self.ds.data_vars.keys())}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset dimensions: {dict(self.ds.dims)}")

        # 修改：使用正确的变量名
        tigge_min = float(self.ds['X_tigge'].min().values)
        tigge_max = float(self.ds['X_tigge'].max().values)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] X_tigge range: [{tigge_min}, {tigge_max}]")

        dem_min = float(self.ds['X_dem'].min().values)
        dem_max = float(self.ds['X_dem'].max().values)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] X_dem range: [{dem_min}, {dem_max}]")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normalizing time features")
        # 修改：使用正确的变量名
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
            # 修改：使用正确的变量名和形状处理
            tigge_data = self.ds['X_tigge'].sel(sample=mask).values.reshape(self.H, self.W, -1)
            seq_data.append(tigge_data)
            time_features = self.ds['time_features_normalized'].sel(sample=mask).values[0]
            time_features_seq.append(time_features)

        tigge_seq = np.stack(seq_data)
        time_features_seq = np.stack(time_features_seq)

        time_t = self.time_points[t]
        mask_t = self.ds.time == time_t

        # 修改：使用正确的变量名
        tigge_spatial = self.ds['X_tigge'].sel(sample=mask_t).values.reshape(self.H, self.W, -1)
        dem_spatial = self.ds['X_dem'].sel(sample=mask_t).values.reshape(self.H, self.W, -1)
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

# 训练函数
def train_model(model, train_loader, val_loader, device,
                epochs=12, learning_rate=0.0003931296040961602, weight_decay=0.0001025927948297095,
                patience=5, accumulation_steps=1):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started training process")
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
    os.makedirs('checkpoints', exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint directory created")

    first_batch = next(iter(train_loader))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - tigge_spatial shape: {first_batch['tigge_spatial'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - dem_spatial shape: {first_batch['dem_spatial'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - tigge_seq shape: {first_batch['tigge_seq'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_t shape: {first_batch['time_features_t'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_seq shape: {first_batch['time_features_seq'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - target shape: {first_batch['target'].shape}")

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch + 1}/{epochs}")
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
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Model saved with val loss: {best_val_loss:.6f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Early stopping at epoch {epoch + 1}')
                break

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving checkpoint for epoch {epoch + 1}")
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
        }, f'checkpoints/checkpoint_epoch_{epoch + 1}.pth')
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        remaining_epochs = epochs - (epoch + 1)
        est_remaining_time = remaining_epochs * epoch_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1} completed in {epoch_time / 60:.2f} minutes")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Estimated time remaining: {est_remaining_time / 3600:.2f} hours")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed in {total_time / 3600:.2f} hours")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating visualization plots")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loss curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_mae) + 1), val_mae, label='Validation MAE', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_curve.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MAE curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label='Validation RMSE', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('rmse_curve.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RMSE curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_r2) + 1), train_r2, label='Train R²', color='blue', linewidth=2)
    plt.plot(range(1, len(val_r2) + 1), val_r2, label='Validation R²', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R² Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_curve.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] R² curve saved")

    print(
        f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Training metrics visualizations saved as: loss_curve.png, mae_curve.png, rmse_curve.png, r2_curve.png')


if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Program started")

    set_seed(42)

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Setting up training parameters")
    batch_size = 16
    epochs = 12
    learning_rate = 0.0003931296040961602
    weight_decay = 0.0001025927948297095
    patience = 5
    accumulation_steps = 1

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading training dataset")
    train_ds = WindDataset("./show_relevance_visualization/train.nc", H, W)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading validation dataset")
    val_ds = WindDataset("./show_relevance_visualization/val.nc", H, W)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating data loaders")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Data loaders created successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model")
    model = WindSpeedPredictor(
        H, W,
        tigge_features=8,
        dropout_rate=0.24733185479083603,
        ltc_hidden_dim=216,
        cbam_reduction=16
    ).to(device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model initialized successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Training")
    train_model(
        model, train_loader, val_loader, device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        accumulation_steps=accumulation_steps
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed!")