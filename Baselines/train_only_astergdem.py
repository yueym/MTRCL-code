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


class ResNetCBAM_TopoOnly(nn.Module):
    def __init__(self, in_channels=3, dropout_rate=0.25):
        super(ResNetCBAM_TopoOnly, self).__init__()
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


class SimpleLTC_TopoOnly(nn.Module):
    def __init__(self, hidden_dim=216, output_dim=216, dropout_rate=0.25):
        super(SimpleLTC_TopoOnly, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, hidden_dim)
        )
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

    def forward(self, time_features_t, H, W):
        b = time_features_t.shape[0]
        time_emb = self.time_net(time_features_t)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        out = self.output_layer(time_emb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
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


class WindSpeedPredictor_TopoOnly(nn.Module):
    def __init__(self, H, W, dropout_rate=0.25, ltc_hidden_dim=216, cbam_reduction=16):
        super(WindSpeedPredictor_TopoOnly, self).__init__()
        self.H = H
        self.W = W
        self.resnet = ResNetCBAM_TopoOnly(in_channels=3, dropout_rate=dropout_rate)
        self.ltc = SimpleLTC_TopoOnly(hidden_dim=ltc_hidden_dim, output_dim=ltc_hidden_dim,
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
        print(f"Total parameters (Topo Only): {total_params:,}")

    def forward(self, dem_spatial, time_features_t):
        b = dem_spatial.size(0)
        time_emb = self.time_embed(time_features_t).view(b, 56, 1, 1)
        resnet_out = self.resnet(dem_spatial, time_emb)
        ltc_out = self.ltc(time_features_t, self.H, self.W)
        fused = self.gated_fusion(resnet_out, ltc_out)
        pred = self.mlp(fused)
        return pred.squeeze(1)


class WindDataset_TopoOnly(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        self.ds = xr.open_dataset(ds_path, cache=False)

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
        dem_spatial = self.ds['X_dem'].sel(sample=mask_t).values.reshape(self.H, self.W, 3)
        target = self.ds['y'].sel(sample=mask_t).values.reshape(self.H, self.W)
        time_features_t = self.ds['time_features_normalized'].sel(sample=mask_t).values[0]

        return {
            'dem_spatial': torch.from_numpy(dem_spatial).float().permute(2, 0, 1),
            'time_features_t': torch.from_numpy(time_features_t).float(),
            'target': torch.from_numpy(target).float()
        }


def calculate_metrics_with_mape(pred, target):
    """计算包含MAPE的7项指标"""
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


def train_model(model, train_loader, val_loader, device, num_epochs=150):
    """训练模型"""
    print(f"开始训练模型 (只使用3个地形参数)...")

    os.makedirs('checkpoints_topo', exist_ok=True)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=3.9e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    scaler_target = joblib.load('./show_relevance_visualization/target_scaler.pkl')
    target_data_min = scaler_target.data_min_[0]
    target_range = 1 / scaler_target.scale_[0]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                dem_spatial = batch['dem_spatial'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                target = batch['target'].to(device)

                optimizer.zero_grad()
                output = model(dem_spatial, time_features_t)
                loss = criterion(output, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.6f}')

            except Exception as e:
                print(f"训练批次错误 {batch_idx}: {str(e)}")
                continue

        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    dem_spatial = batch['dem_spatial'].to(device)
                    time_features_t = batch['time_features_t'].to(device)
                    target = batch['target'].to(device)

                    output = model(dem_spatial, time_features_t)
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

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

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

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints_topo/best_model_topo_only.pth')
            print(f'保存最佳模型，验证损失: {best_val_loss:.6f}')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'早停触发，在第 {epoch + 1} 轮停止训练')
            break

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints_topo/model_epoch_{epoch + 1}_topo_only.pth')

    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

    with open('checkpoints_topo/training_history_topo_only.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("模型已保存至 checkpoints_topo/best_model_topo_only.pth")

    return model


def test_model(model, test_loader, device):
    """测试模型"""
    model.eval()
    all_preds = []
    all_targets = []

    scaler_target = joblib.load('./show_relevance_visualization/target_scaler.pkl')
    target_data_min = scaler_target.data_min_[0]
    target_range = 1 / scaler_target.scale_[0]

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                dem_spatial = batch['dem_spatial'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                target = batch['target'].to(device)

                output = model(dem_spatial, time_features_t)

                all_preds.append(output.cpu())
                all_targets.append(target.cpu())

                if (batch_idx + 1) % 100 == 0:
                    print(f"已处理测试批次: {batch_idx + 1}/{len(test_loader)}")

            except Exception as e:
                print(f"测试批次错误 {batch_idx}: {str(e)}")
                continue

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    all_preds_orig = (all_preds * target_range) + target_data_min
    all_targets_orig = (all_targets * target_range) + target_data_min
    all_preds_orig = np.clip(all_preds_orig, 0, 100)
    all_targets_orig = np.clip(all_targets_orig, 0, 100)

    test_metrics = calculate_metrics_with_mape(
        torch.from_numpy(all_preds_orig),
        torch.from_numpy(all_targets_orig)
    )

    print("\n" + "=" * 60)
    print("MTRCL模型测试结果 - 仅使用3个地形参数")
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

    with open('checkpoints_topo/test_results_topo_only.json', 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    return test_metrics


if __name__ == "__main__":
    # Windows多进程保护
    import multiprocessing

    multiprocessing.freeze_support()

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载数据集...")
    batch_size = 16

    train_ds = WindDataset_TopoOnly("./show_relevance_visualization/train.nc", H, W)
    val_ds = WindDataset_TopoOnly("./show_relevance_visualization/val.nc", H, W)
    test_ds = WindDataset_TopoOnly("./show_relevance_visualization/test.nc", H, W)

    # 修改：设置num_workers=0来解决Windows多进程问题
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print("数据集加载完成")
    print(f"训练集样本数: {len(train_ds)}")
    print(f"验证集样本数: {len(val_ds)}")
    print(f"测试集样本数: {len(test_ds)}")

    print("初始化模型...")
    model = WindSpeedPredictor_TopoOnly(
        H, W,
        dropout_rate=0.25,
        ltc_hidden_dim=216,
        cbam_reduction=16
    ).to(device)

    print("开始训练...")
    start_time = time.time()
    model = train_model(model, train_loader, val_loader, device, num_epochs=8)
    end_time = time.time()
    print(f"训练耗时: {(end_time - start_time) / 3600:.2f} 小时")

    print("开始测试...")
    model.load_state_dict(torch.load('checkpoints_topo/best_model_topo_only.pth'))
    test_metrics = test_model(model, test_loader, device)

    print("训练和测试完成！")
