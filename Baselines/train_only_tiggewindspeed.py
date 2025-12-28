import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
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


class ResNetCBAM_WindOnly(nn.Module):
    def __init__(self, in_channels=1, dropout_rate=0.25):
        super(ResNetCBAM_WindOnly, self).__init__()
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


class ODEFunc_WindOnly(nn.Module):
    def __init__(self, hidden_dim=216, input_dim=1, dropout_rate=0.25):
        super(ODEFunc_WindOnly, self).__init__()
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


class LTC_WindOnly(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=216, output_dim=216, seq_len=4, dt=6.0, dropout_rate=0.25):
        super(LTC_WindOnly, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.dt = dt
        self.ode_func = ODEFunc_WindOnly(hidden_dim, input_dim, dropout_rate)
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
    def __init__(self, input_dim, dropout_rate=0.25):
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


class WindSpeedPredictor_WindOnly(nn.Module):
    def __init__(self, H, W, dropout_rate=0.25, ltc_hidden_dim=216, cbam_reduction=16):
        super(WindSpeedPredictor_WindOnly, self).__init__()
        self.H = H
        self.W = W
        self.resnet = ResNetCBAM_WindOnly(in_channels=1, dropout_rate=dropout_rate)
        self.ltc = LTC_WindOnly(input_dim=1, hidden_dim=ltc_hidden_dim, output_dim=ltc_hidden_dim,
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
        print(f"Total parameters (Wind Only): {total_params:,}")

    def forward(self, wind_spatial, wind_seq, time_features_t, time_features_seq):
        b = wind_spatial.size(0)
        time_emb = self.time_embed(time_features_t).view(b, 56, 1, 1)
        resnet_out = self.resnet(wind_spatial, time_emb)
        ltc_out = self.ltc(wind_seq, time_features_seq)
        fused = self.gated_fusion(resnet_out, ltc_out)
        pred = self.mlp(fused)
        return pred.squeeze(1)


class WindDataset_WindOnly(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        self.ds = xr.open_dataset(ds_path, cache=False)

        # 直接搜索并提取tigge_wind_speed参数
        self.wind_speed_index = self._find_wind_speed_parameter()

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

    def _find_wind_speed_parameter(self):
        """直接在数据集中搜索tigge_wind_speed参数"""
        print("\n" + "=" * 60)
        print("🔍 在数据集中搜索tigge_wind_speed参数")
        print("=" * 60)

        # 检查数据集中是否直接有tigge_wind_speed变量
        if 'tigge_wind_speed' in self.ds.data_vars:
            print("✅ 找到直接的tigge_wind_speed变量！")
            wind_speed_data = self.ds['tigge_wind_speed'].values
            print(f"tigge_wind_speed数据形状: {wind_speed_data.shape}")
            print(f"数值范围: [{wind_speed_data.min():.6f}, {wind_speed_data.max():.6f}]")
            print(f"平均值: {wind_speed_data.mean():.6f}")
            print("使用直接的tigge_wind_speed变量")
            return 'direct'  # 标记为直接使用

        # 如果没有直接变量，检查数据集的属性和坐标
        print("数据集变量列表:")
        for var in self.ds.data_vars:
            print(f"  - {var}: {self.ds[var].shape}")

        print("\n数据集坐标:")
        for coord in self.ds.coords:
            print(
                f"  - {coord}: {self.ds.coords[coord].shape if hasattr(self.ds.coords[coord], 'shape') else 'scalar'}")

        # 检查X_tigge的维度信息
        if 'X_tigge' in self.ds.data_vars:
            print(f"\nX_tigge变量信息:")
            print(f"  形状: {self.ds['X_tigge'].shape}")
            print(f"  维度: {self.ds['X_tigge'].dims}")

            # 检查是否有特征名称维度
            if len(self.ds['X_tigge'].dims) > 1:
                feature_dim = self.ds['X_tigge'].dims[-1]  # 通常特征维度是最后一个
                print(f"  特征维度名称: {feature_dim}")

                # 检查特征维度的坐标
                if feature_dim in self.ds.coords:
                    feature_names = self.ds.coords[feature_dim].values
                    print(f"  特征名称列表:")
                    for i, name in enumerate(feature_names):
                        print(f"    {i:2d}: {name}")

                    # 搜索tigge_wind_speed
                    if 'tigge_wind_speed' in feature_names:
                        wind_index = list(feature_names).index('tigge_wind_speed')
                        print(f"\n🎯 找到tigge_wind_speed！索引位置: {wind_index}")

                        # 验证数据
                        tigge_data = self.ds['X_tigge'].values
                        wind_speed_data = tigge_data[:, wind_index]
                        print(f"验证数据:")
                        print(f"  数据形状: {wind_speed_data.shape}")
                        print(f"  数值范围: [{wind_speed_data.min():.6f}, {wind_speed_data.max():.6f}]")
                        print(f"  平均值: {wind_speed_data.mean():.6f}")
                        print(f"  标准差: {wind_speed_data.std():.6f}")

                        # 合理性检查
                        if wind_speed_data.min() >= 0 and 0 < wind_speed_data.mean() < 20:
                            print("✅ 数据合理性检查通过")
                        else:
                            print("⚠️  数据可能存在异常，但继续使用")

                        return wind_index
                    else:
                        print("❌ 在特征名称中未找到tigge_wind_speed")
                        print("可用的特征名称:", list(feature_names))
                else:
                    print(f"⚠️  特征维度{feature_dim}没有坐标信息")
            else:
                print("⚠️  X_tigge只有一个维度")

        # 如果都没找到，尝试其他可能的搜索方式
        print("\n🔍 尝试其他搜索方式...")

        # 检查是否有包含wind_speed的变量名
        wind_related_vars = [var for var in self.ds.data_vars if 'wind' in var.lower()]
        if wind_related_vars:
            print(f"找到包含'wind'的变量: {wind_related_vars}")
            for var in wind_related_vars:
                if 'speed' in var.lower():
                    print(f"✅ 可能的风速变量: {var}")
                    return var

        # 最后的fallback：如果实在找不到，询问用户
        print("\n❌ 无法自动找到tigge_wind_speed参数")
        print("请检查数据集结构，或手动指定参数位置")

        # 返回一个默认值，但会在后续处理中报错
        raise ValueError("未找到tigge_wind_speed参数！请检查数据集结构。")

    def _extract_wind_speed_data(self, sample_mask):
        """根据找到的索引提取wind_speed数据"""
        if self.wind_speed_index == 'direct':
            # 直接使用tigge_wind_speed变量
            return self.ds['tigge_wind_speed'].sel(sample=sample_mask).values.reshape(self.H, self.W, 1)
        elif isinstance(self.wind_speed_index, str):
            # 使用找到的变量名
            return self.ds[self.wind_speed_index].sel(sample=sample_mask).values.reshape(self.H, self.W, 1)
        elif isinstance(self.wind_speed_index, int):
            # 使用索引位置
            tigge_full = self.ds['X_tigge'].sel(sample=sample_mask).values.reshape(self.H, self.W, -1)
            return tigge_full[:, :, self.wind_speed_index:self.wind_speed_index + 1]
        else:
            raise ValueError(f"无效的wind_speed_index: {self.wind_speed_index}")

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
            # 使用新的提取方法
            wind_data = self._extract_wind_speed_data(mask)
            seq_data.append(wind_data)
            time_features = self.ds['time_features_normalized'].sel(sample=mask).values[0]
            time_features_seq.append(time_features)

        wind_seq = np.stack(seq_data)  # shape: (seq_len, H, W, 1)
        time_features_seq = np.stack(time_features_seq)

        time_t = self.time_points[t]
        mask_t = self.ds.time == time_t
        wind_spatial = self._extract_wind_speed_data(mask_t)
        target = self.ds['y'].sel(sample=mask_t).values.reshape(self.H, self.W)
        time_features_t = time_features_seq[-1]

        return {
            'wind_spatial': torch.from_numpy(wind_spatial).float().permute(2, 0, 1),  # (1, H, W)
            'wind_seq': torch.from_numpy(wind_seq).float().permute(0, 3, 1, 2),  # (seq_len, 1, H, W)
            'time_features_t': torch.from_numpy(time_features_t).float(),
            'time_features_seq': torch.from_numpy(time_features_seq).float(),
            'target': torch.from_numpy(target).float()
        }


def calculate_metrics_with_mape(pred, target):
    """计算包含MAPE的8项指标"""
    pred = pred.flatten()
    target = target.flatten()

    FA = ((pred - target).abs() < 1).float().mean().item() * 100
    RMSE = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    MAE = torch.mean((pred - target).abs()).item()
    mean_target = torch.mean(target).item()
    rRMSE = (RMSE / mean_target) * 100 if mean_target > 0 else 0
    rMAE = (MAE / mean_target) * 100 if mean_target > 0 else 0
    R = torch.corrcoef(torch.stack([pred, target]))[0, 1].item()

    ss_tot = torch.sum((target - mean_target) ** 2).item()
    ss_res = torch.sum((target - pred) ** 2).item()
    R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    mask = target > 1e-6
    if torch.sum(mask) > 0:
        MAPE = torch.mean(torch.abs((target[mask] - pred[mask]) / target[mask])).item() * 100
    else:
        MAPE = 0.0

    return {
        'FA': FA, 'RMSE': RMSE, 'MAE': MAE, 'rRMSE': rRMSE,
        'rMAE': rMAE, 'R': R, 'R2': R2, 'MAPE': MAPE
    }


def train_model(model, train_loader, val_loader, device, num_epochs=80):
    """训练模型"""
    print(f"开始训练模型 (只使用tigge_wind_speed历史数据)...")

    # 创建保存目录
    os.makedirs('checkpoints_wind', exist_ok=True)

    # 优化器和损失函数
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=3.9e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    # 加载标准化器
    scaler_target = joblib.load('./show_relevance_visualization/target_scaler.pkl')
    target_data_min = scaler_target.data_min_[0]
    target_range = 1 / scaler_target.scale_[0]

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                wind_spatial = batch['wind_spatial'].to(device)
                wind_seq = batch['wind_seq'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                time_features_seq = batch['time_features_seq'].to(device)
                target = batch['target'].to(device)

                optimizer.zero_grad()
                output = model(wind_spatial, wind_seq, time_features_t, time_features_seq)
                loss = criterion(output, target)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                if (batch_idx + 1) % 50 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.6f}')

            except Exception as e:
                print(f"训练批次错误 {batch_idx}: {str(e)}")
                continue

        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    wind_spatial = batch['wind_spatial'].to(device)
                    wind_seq = batch['wind_seq'].to(device)
                    time_features_t = batch['time_features_t'].to(device)
                    time_features_seq = batch['time_features_seq'].to(device)
                    target = batch['target'].to(device)

                    output = model(wind_spatial, wind_seq, time_features_t, time_features_seq)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    val_batches += 1

                    all_preds.append(output.cpu())
                    all_targets.append(target.cpu())

                except Exception as e:
                    print(f"验证批次错误 {batch_idx}: {str(e)}")
                    continue

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)

        # 计算验证集指标
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # 反标准化
            all_preds_orig = (all_preds.numpy() * target_range) + target_data_min
            all_targets_orig = (all_targets.numpy() * target_range) + target_data_min
            all_preds_orig = np.clip(all_preds_orig, 0, 100)
            all_targets_orig = np.clip(all_targets_orig, 0, 100)

            val_metrics = calculate_metrics_with_mape(
                torch.from_numpy(all_preds_orig),
                torch.from_numpy(all_targets_orig)
            )

            print(f'Epoch [{epoch + 1}/{num_epochs}]:')
            print(f'  Train Loss: {avg_train_loss:.6f}')
            print(f'  Val Loss: {avg_val_loss:.6f}')
            print(f'  Val FA: {val_metrics["FA"]:.2f}%, RMSE: {val_metrics["RMSE"]:.4f}, '
                  f'MAE: {val_metrics["MAE"]:.4f}, R: {val_metrics["R"]:.4f}')
            print('-' * 60)

            # 学习率调整
        scheduler.step(avg_val_loss)

        # 早停和模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints_wind/best_model_wind_only.pth')
            print(f'保存最佳模型，验证损失: {best_val_loss:.6f}')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'早停触发，在第 {epoch + 1} 轮停止训练')
            break

        # 每5轮保存一次模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints_wind/model_epoch_{epoch + 1}_wind_only.pth')

        # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

    with open('checkpoints_wind/training_history_wind_only.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("模型已保存至 checkpoints_wind/best_model_wind_only.pth")

    return model


def test_model(model, test_loader, device):
    """测试模型"""
    model.eval()
    all_preds = []
    all_targets = []

    scaler_target = joblib.load('./show_relevance_visualization/target_scaler.pkl')
    target_data_min = scaler_target.data_min_[0]
    target_range = 1 / scaler_target.scale_[0]

    print("开始测试模型...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                wind_spatial = batch['wind_spatial'].to(device)
                wind_seq = batch['wind_seq'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                time_features_seq = batch['time_features_seq'].to(device)
                target = batch['target'].to(device)

                output = model(wind_spatial, wind_seq, time_features_t, time_features_seq)

                all_preds.append(output.cpu())
                all_targets.append(target.cpu())

                if (batch_idx + 1) % 50 == 0:
                    print(f"已处理测试批次: {batch_idx + 1}/{len(test_loader)}")

            except Exception as e:
                print(f"测试批次错误 {batch_idx}: {str(e)}")
                continue

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # 反标准化
    all_preds_orig = (all_preds * target_range) + target_data_min
    all_targets_orig = (all_targets * target_range) + target_data_min
    all_preds_orig = np.clip(all_preds_orig, 0, 100)
    all_targets_orig = np.clip(all_targets_orig, 0, 100)

    # 计算测试指标
    test_metrics = calculate_metrics_with_mape(
        torch.from_numpy(all_preds_orig),
        torch.from_numpy(all_targets_orig)
    )

    print("\n" + "=" * 60)
    print("MTRCL模型测试结果 - 仅使用tigge_wind_speed历史数据")
    print("=" * 60)
    print(f"准确率 (FA):           {test_metrics['FA']:.2f}%")
    print(f"均方根误差 (RMSE):     {test_metrics['RMSE']:.4f} m/s")
    print(f"平均绝对误差 (MAE):    {test_metrics['MAE']:.4f} m/s")
    print(f"相对RMSE (rRMSE):     {test_metrics['rRMSE']:.2f}%")
    print(f"相对MAE (rMAE):       {test_metrics['rMAE']:.2f}%")
    print(f"相关系数 (R):         {test_metrics['R']:.4f}")
    print(f"决定系数 (R²):        {test_metrics['R2']:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {test_metrics['MAPE']:.2f}%")
    print("=" * 60)

    # 保存测试结果
    with open('checkpoints_wind/test_results_wind_only.json', 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    return test_metrics


# 主程序
if __name__ == "__main__":
    # Windows多进程保护
    import multiprocessing

    multiprocessing.freeze_support()

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 首先检查一个数据集文件的结构
    print("🔍 检查数据集结构...")
    try:
        sample_ds = xr.open_dataset("./show_relevance_visualization/train.nc", cache=False)
        print("✅ 成功打开数据集，开始搜索tigge_wind_speed参数...")
        sample_ds.close()
    except Exception as e:
        print(f"❌ 无法打开数据集: {e}")
        exit(1)

    # 加载数据集（会自动搜索tigge_wind_speed）
    print("加载数据集...")
    batch_size = 16

    try:
        print("🔍 搜索训练集中的tigge_wind_speed...")
        train_ds = WindDataset_WindOnly("./show_relevance_visualization/train.nc", H, W)

        print("🔍 搜索验证集中的tigge_wind_speed...")
        val_ds = WindDataset_WindOnly("./show_relevance_visualization/val.nc", H, W)

        print("🔍 搜索测试集中的tigge_wind_speed...")
        test_ds = WindDataset_WindOnly("./show_relevance_visualization/test.nc", H, W)

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("请检查数据集中是否包含tigge_wind_speed参数")
        exit(1)

    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print("✅ 数据集加载完成")
    print(f"训练集样本数: {len(train_ds)}")
    print(f"验证集样本数: {len(val_ds)}")
    print(f"测试集样本数: {len(test_ds)}")

    # 验证数据提取是否正确
    print("\n🧪 验证数据提取...")
    sample_batch = train_ds[0]
    print(f"单个样本数据形状验证:")
    print(f"  wind_spatial shape: {sample_batch['wind_spatial'].shape} (期望: [1, 48, 96])")
    print(f"  wind_seq shape: {sample_batch['wind_seq'].shape} (期望: [4, 1, 48, 96])")
    print(f"  target shape: {sample_batch['target'].shape} (期望: [48, 96])")
    print(
        f"  wind_spatial数值范围: [{sample_batch['wind_spatial'].min():.4f}, {sample_batch['wind_spatial'].max():.4f}]")
    print("✅ 数据提取验证完成！\n")

    # 初始化模型
    print("初始化模型...")
    model = WindSpeedPredictor_WindOnly(
        H, W,
        dropout_rate=0.25,
        ltc_hidden_dim=216,
        cbam_reduction=16
    ).to(device)

    # 训练模型
    print("开始训练...")
    start_time = time.time()
    model = train_model(model, train_loader, val_loader, device, num_epochs=1)
    end_time = time.time()
    print(f"训练耗时: {(end_time - start_time) / 3600:.2f} 小时")

    # 测试模型
    print("开始测试...")
    model.load_state_dict(torch.load('checkpoints_wind/best_model_wind_only.pth', map_location=device))
    test_metrics = test_model(model, test_loader, device)

    print("\n🎉 训练和测试完成！")
    print(f"模型文件保存在: checkpoints_wind/best_model_wind_only.pth")
    print(f"测试结果保存在: checkpoints_wind/test_results_wind_only.json")

    # 输出最终结果摘要
    print(f"\n📊 最终测试结果摘要:")
    print(f"FA: {test_metrics['FA']:.2f}%")
    print(f"RMSE: {test_metrics['RMSE']:.4f} m/s")
    print(f"MAE: {test_metrics['MAE']:.4f} m/s")
    print(f"R: {test_metrics['R']:.4f}")
    print(f"MAPE: {test_metrics['MAPE']:.2f}%")