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
import time


def count_linear_flops(input_shape, weight_shape):
    """计算Linear层的FLOPs"""
    batch_size = input_shape[0]
    in_features, out_features = weight_shape
    flops = batch_size * out_features * in_features * 2
    return flops


def estimate_model_flops(model, input_sample):
    """估算BPNN模型的FLOPs"""
    input_features = input_sample
    batch_size, input_dim = input_features.shape

    total_flops = 0
    current_shape = (batch_size, input_dim)

    # 遍历模型中的每个线性层来计算FLOPs
    # BPNN模型是一个nn.Sequential
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
            flops = count_linear_flops(current_shape, (in_features, out_features))
            total_flops += flops
            print(f"  Linear layer {i + 1} ({in_features}->{out_features}): {flops:,} FLOPs")
            current_shape = (batch_size, out_features)

    return total_flops


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# 这个BPNN模型定义与您的训练代码完全一致
class BPNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(BPNN, self).__init__()
        self.net = nn.Sequential(
            # 第一层: 降维
            nn.Linear(input_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # 第二层
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # 第三层
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # 第四层
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # 第五层
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # 输出层
            nn.Linear(4096, output_dim),
            nn.Sigmoid()  # 将输出映射到[0, 1]之间，匹配归一化后的目标值
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        self.input_dim = H * W * (8 + 3)
        self.output_dim = H * W
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)
        tigge_min = float(self.ds['X_tigge'].min().values)
        tigge_max = float(self.ds['X_tigge'].max().values)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] tigge_features range: [{tigge_min}, {tigge_max}]")
        dem_min = float(self.ds['X_dem'].min().values)
        dem_max = float(self.ds['X_dem'].max().values)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] dem_features range: [{dem_min}, {dem_max}]")
        time_data = self.ds['time_features'].values
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
        time_t = self.time_points[t]
        mask_t = self.ds.time == time_t

        tigge_spatial = self.ds['X_tigge'].sel(sample=mask_t).values.reshape(self.H, self.W, 8)
        dem_spatial = self.ds['X_dem'].sel(sample=mask_t).values.reshape(self.H, self.W, 3)
        target = self.ds['y'].sel(sample=mask_t).values.reshape(self.H, self.W)

        spatial_features = np.concatenate([tigge_spatial, dem_spatial], axis=-1)
        spatial_features_flat = spatial_features.reshape(-1)
        target_flat = target.reshape(-1)

        return {
            'input_features': torch.from_numpy(spatial_features_flat).float(),
            'target': torch.from_numpy(target_flat).float(),
            'tigge_wind_flat': torch.from_numpy(tigge_spatial.reshape(-1)).float()
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


def test_model(model, test_loader, device, H, W):
    model.eval()
    all_preds = []
    all_targets = []
    all_tigge_wind = []
    all_dates = []
    criterion = nn.SmoothL1Loss()

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
                input_features = batch['input_features'].to(device)
                target = batch['target'].to(device)
                tigge_wind_flat = batch['tigge_wind_flat'].to(device)

                output = model(input_features)
                loss = criterion(output, target)
                test_loss += loss.item() * input_features.size(0)
                batch_count += 1

                # 获取当前批次大小
                current_batch_size = output.size(0)

                # 反归一化
                pred_orig = (output.cpu().numpy() * target_range) + target_data_min
                target_orig = (target.cpu().numpy() * target_range) + target_data_min

                # 修正：tigge_wind_flat 的形状是 (batch_size, H*W*8)，我们只需要第一个通道
                # 首先 reshape 为 (batch_size, H*W, 8)，然后取第一个通道 (wind speed)
                tigge_wind_reshaped = tigge_wind_flat.cpu().numpy().reshape(current_batch_size, H * W, 8)
                tigge_wind_speed = tigge_wind_reshaped[:, :, 0]  # 取第一个通道 (wind speed)
                tigge_wind_orig = (tigge_wind_speed * tigge_range) + tigge_data_min

                pred_orig = np.clip(pred_orig, 0, 100)
                target_orig = np.clip(target_orig, 0, 100)
                tigge_wind_orig = np.clip(tigge_wind_orig, 0, 100)

                # 按照批量大小逐个reshape
                for i in range(current_batch_size):
                    all_preds.append(pred_orig[i].reshape(H, W))
                    all_targets.append(target_orig[i].reshape(H, W))
                    all_tigge_wind.append(tigge_wind_orig[i].reshape(H, W))

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

        # 检查是否有足够的数据进行堆叠
        if len(all_preds) == 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: No valid predictions collected!")
            return

        # 堆叠数据
        all_preds = np.stack(all_preds)
        all_targets = np.stack(all_targets)
        all_tigge_wind = np.stack(all_tigge_wind)
        all_dates = np.array(all_dates[:len(all_preds)])

        # 保存文件
        np.save('BPNN_test_preds.npy', all_preds)
        np.save('BPNN_test_targets.npy', all_targets)
        np.save('BPNN_test_tigge_wind.npy', all_tigge_wind)
        np.save('BPNN_test_dates.npy', all_dates)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test predictions, targets, TIGGE wind, and dates saved as BPNN_test_*.npy files.")

        metrics_all = calculate_metrics(
            torch.from_numpy(all_preds),
            torch.from_numpy(all_targets),
            torch.from_numpy(all_tigge_wind)
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Yearly Test Metrics (BPNN):", metrics_all)

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
            season_preds = all_preds[mask]
            season_targets = all_targets[mask]
            season_tigge = all_tigge_wind[mask]
            seasonal_metrics[season] = calculate_metrics(
                torch.from_numpy(season_preds),
                torch.from_numpy(season_targets),
                torch.from_numpy(season_tigge)
            )
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {season} Test Metrics (BPNN):", seasonal_metrics[season])

        all_metrics = {'Yearly': metrics_all, **seasonal_metrics}
        with open('BPNN_test_metrics.json', 'w') as f:
            json.dump(all_metrics, f)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test metrics saved as BPNN_test_metrics.json")

        monthly_metrics = {}
        months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in range(1, 13):
            month_mask = (pd.to_datetime(all_dates).month == month)
            if np.sum(month_mask) > 0:
                month_preds = all_preds[month_mask]
                month_targets = all_targets[month_mask]
                month_tigge = all_tigge_wind[month_mask]
                monthly_metrics[month] = calculate_metrics(
                    torch.from_numpy(month_preds),
                    torch.from_numpy(month_targets),
                    torch.from_numpy(month_tigge)
                )
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Month {months_names[month - 1]} Test Metrics (BPNN):",
                      monthly_metrics[month])

        monthly_metrics_dict = {str(k): v for k, v in monthly_metrics.items()}
        with open('BPNN_test_monthly_metrics.json', 'w') as f:
            json.dump(monthly_metrics_dict, f)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monthly test metrics saved as BPNN_test_monthly_metrics.json")


if __name__ == "__main__":
    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets...")
    batch_size = 16
    test_ds = WindDataset("./show_relevance_visualization/test.nc", H, W)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Datasets loaded successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
    input_dim = H * W * (8 + 3)
    output_dim = H * W
    model = BPNN(input_dim=input_dim, output_dim=output_dim, dropout_rate=0.3).to(device)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model weights...")
    model.load_state_dict(torch.load('checkpoints/best_model_BPNN.pth'))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model weights loaded successfully.")

    # ==============================================================================
    # 调试步骤：检查模型输出维度
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== DEBUGGING: CHECKING MODEL OUTPUT DIMENSION =====")
    model.eval()
    with torch.no_grad():
        # 从测试集获取一个样本
        sample = next(iter(test_loader))
        input_features_sample = sample['input_features'][0:1].to(device)

        # 运行模型并获取输出
        output_debug = model(input_features_sample)

        # 打印输出形状
        print(f"Model input shape: {input_features_sample.shape}")
        print(f"Model output shape: {output_debug.shape}")

        # 检查输出维度是否正确
        expected_output_dim = H * W
        actual_output_dim = output_debug.shape[1]

        if actual_output_dim == expected_output_dim:
            print(f"SUCCESS: Model output dimension ({actual_output_dim}) is correct.")
        else:
            print(
                f"!!! ERROR: Model output dimension ({actual_output_dim}) is INCORRECT. Expected: {expected_output_dim} !!!")
            # 如果维度不正确，直接退出程序，因为这表明保存的模型权重有根本性问题。
            # 您需要用正确的模型结构重新训练。
            exit()
            # ==============================================================================

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Preparing sample for GFLOPs and forecast time calculation...")
    sample = next(iter(test_loader))
    input_features_sample = sample['input_features'][0:1].to(device)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== CALCULATING MODEL COMPLEXITY =====")
    model.eval()

    # 修改：更新input_sample
    input_sample = input_features_sample
    total_flops = estimate_model_flops(model, input_sample)
    gflops = total_flops / 1e9

    total_params, trainable_params = count_parameters(model)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MODEL COMPLEXITY ANALYSIS =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Estimated GFLOPs: {gflops:.6f} GFLOPs")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total FLOPs: {total_flops:,} FLOPs")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total parameters: {total_params:,} parameters")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trainable parameters: {trainable_params:,} parameters")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Non-trainable parameters: {total_params - trainable_params:,} parameters")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model size: {total_params * 4 / (1024 ** 2):.2f} MB (float32)")

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MEASURING FORECAST TIME =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warming up model...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_features_sample)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting inference time measurement...")
    num_runs = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_features_sample)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    forecast_time_ms = (end_time - start_time) * 1000 / num_runs
    forecast_time_us = forecast_time_ms * 1000
    forecast_time_s = forecast_time_ms / 1000

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== FORECAST TIME ANALYSIS =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Average forecast time: {forecast_time_ms:.4f} ms per sample")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Average forecast time: {forecast_time_us:.2f} μs per sample")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Average forecast time: {forecast_time_s:.6f} s per sample")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Throughput: {1000 / forecast_time_ms:.2f} samples/second")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Throughput: {3600000 / forecast_time_ms:.0f} samples/hour")

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

    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
        memory_used_gb = memory_used / 1024
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MEMORY USAGE =====")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Peak GPU Memory used: {memory_used:.2f} MB")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Peak GPU Memory used: {memory_used_gb:.3f} GB")

    compute_intensity = gflops / forecast_time_s
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== COMPUTATIONAL EFFICIENCY =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computational intensity: {compute_intensity:.2f} GFLOPS/s")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FLOPs per parameter: {total_flops / total_params:.2f}")

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== SUMMARY FOR RESEARCH PAPER =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model: BPNN")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GFLOPs: {gflops:.3f}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Forecast time: {forecast_time_ms:.2f} ms")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Parameters: {total_params / 1e6:.2f}M")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model size: {total_params * 4 / (1024 ** 2):.1f} MB")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Testing...")
    test_model(model, test_loader, device, H, W)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing completed!")