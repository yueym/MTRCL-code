import torch
import torch.nn as nn
import torch.nn.functional as F
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
import math
from sklearn.metrics import r2_score


# ==============================================================================
# 1. 模型定义 (与训练代码完全一致)
# ==============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        if x.dim() == 3:
            batch_size = x.size(0)
            outputs = []
            for b in range(batch_size):
                support = torch.mm(adj, x[b])
                output = self.linear(support)
                outputs.append(output)
            return torch.stack(outputs, dim=0)
        else:
            support = torch.mm(adj, x)
            return self.linear(support)


class STDGN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_nodes, k_neighbors, dropout_rate=0.3):
        super(STDGN, self).__init__()
        self.num_nodes = num_nodes
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim

        self.gcn1 = GCNLayer(in_channels, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pred_layer = nn.Linear(hidden_dim, out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)

    def _create_spatial_graph(self, H, W, device):
        num_nodes = H * W
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        node_coords = torch.tensor([(i // W, i % W) for i in range(num_nodes)], device=device, dtype=torch.float)
        dist_matrix = torch.cdist(node_coords, node_coords, p=1)
        _, knn_indices = torch.topk(dist_matrix, self.k_neighbors + 1, largest=False)
        for node_idx in range(num_nodes):
            neighbors = knn_indices[node_idx, 1:]
            adj[node_idx, neighbors] = 1.0
        adj = adj + adj.T
        adj[adj > 0] = 1.0
        adj = adj + torch.eye(num_nodes, device=device)
        return adj

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        num_nodes = H * W
        x = x.reshape(batch_size, seq_len, num_nodes, C)

        if not hasattr(self, 'adj_matrix') or self.adj_matrix.device != x.device:
            self.adj_matrix = self._create_spatial_graph(H, W, x.device)

        h_0 = torch.zeros(1, batch_size, num_nodes, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            current_x_t = x[:, t, :, :]
            out = F.relu(self.gcn1(current_x_t, self.adj_matrix))
            out = self.dropout(out)
            out = self.gcn2(out, self.adj_matrix)

            out_flat = out.reshape(batch_size * num_nodes, 1, self.hidden_dim)
            h_0_flat = h_0.reshape(1, batch_size * num_nodes, self.hidden_dim)
            _, h_t_flat = self.gru(out_flat, h_0_flat)
            h_0 = h_t_flat.reshape(1, batch_size, num_nodes, self.hidden_dim)

        final_spatial_features = h_0.squeeze(0)
        output = self.pred_layer(final_spatial_features)

        output = output.reshape(batch_size, H, W, out_channels).permute(0, 3, 1, 2)
        return output


# ==============================================================================
# 2. 测试数据集定义 (与训练代码完全一致)
# ==============================================================================
class WindDatasetSTDGN(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)
        time_data = self.ds['time_features'].values
        times = pd.to_datetime(
            {'year': time_data[:, 0], 'month': time_data[:, 1], 'day': time_data[:, 2], 'hour': time_data[:, 3]})
        self.ds = self.ds.assign_coords(time=("sample", times)).sortby('time')
        self.time_points = np.unique(self.ds.time.values)
        self.T = len(self.time_points)
        self.sample_indices = np.arange(self.T - self.seq_len + 1)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset initialized with {len(self.sample_indices)} sequences")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        actual_idx = self.sample_indices[idx]

        tigge_list = []
        dem_list = []

        for t_offset in range(self.seq_len):
            time_point = self.time_points[actual_idx + t_offset]
            mask = self.ds.time == time_point
            tigge_spatial = self.ds['X_tigge'].sel(sample=mask).values.reshape(self.H, self.W, 8)
            dem_spatial = self.ds['X_dem'].sel(sample=mask).values.reshape(self.H, self.W, 3)
            tigge_list.append(tigge_spatial)
            dem_list.append(dem_spatial)

        target_time_point = self.time_points[actual_idx + self.seq_len - 1]
        target_mask = self.ds.time == target_time_point
        target = self.ds['y'].sel(sample=target_mask).values.reshape(self.H, self.W)

        tigge_sequence = np.stack(tigge_list, axis=0)
        dem_sequence = np.stack(dem_list, axis=0)
        spatial_sequence = np.concatenate([tigge_sequence, dem_sequence], axis=-1)

        spatial_sequence = torch.from_numpy(spatial_sequence).float().permute(3, 0, 1, 2)
        target = torch.from_numpy(target).float()

        return {'spatial_sequence': spatial_sequence, 'target': target}


# ==============================================================================
# 3. 核心功能：指标计算和测试函数
# ==============================================================================
def calculate_metrics(pred, target, tigge_wind):
    pred = pred.flatten()
    target = target.flatten()
    tigge_wind = tigge_wind.flatten()
    epsilon = 1e-8
    FA_pred = ((pred - target).abs() < 1).float().mean().item() * 100
    FA_tigge = ((tigge_wind - target).abs() < 1).float().mean().item() * 100
    RMSE_pred = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    RMSE_tigge = torch.sqrt(torch.mean((tigge_wind - target) ** 2)).item()
    MAE_pred = torch.mean((pred - target).abs()).item()
    MAE_tigge = torch.mean((tigge_wind - target).abs()).item()
    mean_target = torch.mean(target).item()
    rRMSE_pred = (RMSE_pred / mean_target) * 100 if mean_target != 0 else 0
    rRMSE_tigge = (RMSE_tigge / mean_target) * 100 if mean_target != 0 else 0
    rMAE_pred = (MAE_pred / mean_target) * 100 if mean_target != 0 else 0
    rMAE_tigge = (MAE_tigge / mean_target) * 100 if mean_target != 0 else 0
    R_pred = torch.corrcoef(torch.stack([pred, target]))[0, 1].item()
    R_tigge = torch.corrcoef(torch.stack([tigge_wind, target]))[0, 1].item()
    R2_pred = r2_score(target.cpu().numpy(), pred.cpu().numpy())
    R2_tigge = r2_score(target.cpu().numpy(), tigge_wind.cpu().numpy())

    target_safe = torch.where(torch.abs(target) < epsilon, epsilon, target)
    MAPE_pred = torch.mean(torch.abs((pred - target) / target_safe)).item() * 100
    MAPE_tigge = torch.mean(torch.abs((tigge_wind - target) / target_safe)).item() * 100

    return {
        'FA_pred': FA_pred, 'RMSE_pred': RMSE_pred, 'MAE_pred': MAE_pred,
        'rRMSE_pred': rRMSE_pred, 'rMAE_pred': rMAE_pred, 'R_pred': R_pred, 'R2_pred': R2_pred, 'MAPE_pred': MAPE_pred,
        'FA_tigge': FA_tigge, 'RMSE_tigge': RMSE_tigge, 'MAE_tigge': MAE_tigge,
        'rRMSE_tigge': rRMSE_tigge, 'rMAE_tigge': rMAE_tigge, 'R_tigge': R_tigge, 'R2_tigge': R2_tigge,
        'MAPE_tigge': MAPE_tigge,
    }


def test_model(model, test_loader, device, H, W, seq_len, batch_size):
    model.eval()
    all_preds, all_targets, all_tigge_wind, all_dates = [], [], [], []
    criterion = nn.SmoothL1Loss()

    scaler_target = joblib.load('./show_relevance_visualization/target_scaler.pkl')
    scaler_tigge = joblib.load('./show_relevance_visualization/tigge_feature_scaler.pkl')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Scaler files loaded successfully")
    target_data_min = scaler_target.data_min_[0]
    target_range = 1 / scaler_target.scale_[0]
    tigge_data_min = scaler_tigge.data_min_[26]
    tigge_range = 1 / scaler_tigge.scale_[26]

    with torch.no_grad():
        test_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(test_loader):
            try:
                spatial_sequence = batch['spatial_sequence'].to(device)
                target = batch['target'].to(device)

                # STDGN的输入需要 permute
                spatial_sequence = spatial_sequence.permute(0, 2, 1, 3, 4)

                # 获取TIGGE基准风：序列中的最后一个时间步的第一通道
                # spatial_sequence shape: (batch, seq_len, C, H, W)
                tigge_wind_flat = spatial_sequence[:, -1, 0, :, :]

                output = model(spatial_sequence)
                loss = criterion(output.squeeze(1), target)
                test_loss += loss.item() * target.size(0)
                batch_count += 1

                # 反归一化
                pred_orig = (output.squeeze(1).cpu().numpy() * target_range) + target_data_min
                target_orig = (target.cpu().numpy() * target_range) + target_data_min
                tigge_wind_orig = (tigge_wind_flat.cpu().numpy() * tigge_range) + tigge_data_min

                pred_orig = np.clip(pred_orig, 0, 100)
                target_orig = np.clip(target_orig, 0, 100)
                tigge_wind_orig = np.clip(tigge_wind_orig, 0, 100)

                all_preds.append(pred_orig)
                all_targets.append(target_orig)
                all_tigge_wind.append(tigge_wind_orig)

                # 日期处理
                start_idx = batch_idx * batch_size
                end_idx = start_idx + target.shape[0]
                # 获取每个样本对应的目标时间点
                batch_indices = test_loader.dataset.sample_indices[start_idx:end_idx]
                batch_dates = test_loader.dataset.time_points[batch_indices + seq_len - 1]
                all_dates.extend(batch_dates)

                if batch_idx % 50 == 0 or batch_idx == len(test_loader) - 1:
                    progress = (batch_idx + 1) / len(test_loader) * 100
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test progress: {progress:.1f}%")

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing test batch {batch_idx}: {str(e)}")
                continue

        test_loss = test_loss / len(test_loader.dataset) if batch_count > 0 else float('inf')
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Test Loss: {test_loss:.4f}')

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_tigge_wind = np.concatenate(all_tigge_wind)
        all_dates = np.array(all_dates)

        # 保存文件
        np.save('STDGN_test_preds.npy', all_preds)
        np.save('STDGN_test_targets.npy', all_targets)
        np.save('STDGN_test_tigge_wind.npy', all_tigge_wind)
        np.save('STDGN_test_dates.npy', all_dates)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test predictions and targets saved as STDGN_test_*.npy files.")

        # 计算并保存指标
        metrics_all = calculate_metrics(
            torch.from_numpy(all_preds), torch.from_numpy(all_targets), torch.from_numpy(all_tigge_wind)
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Yearly Test Metrics (STDGN):", metrics_all)

        seasons = {'Spring': (3, 5), 'Summer': (6, 8), 'Autumn': (9, 11), 'Winter': (12, 2)}
        seasonal_metrics = {}
        for season, (start_month, end_month) in seasons.items():
            if start_month < end_month:
                mask = (pd.to_datetime(all_dates).month >= start_month) & (pd.to_datetime(all_dates).month <= end_month)
            else:
                mask = (pd.to_datetime(all_dates).month >= start_month) | (pd.to_datetime(all_dates).month <= end_month)
            if np.sum(mask) > 0:
                season_preds = all_preds[mask]
                season_targets = all_targets[mask]
                season_tigge = all_tigge_wind[mask]
                seasonal_metrics[season] = calculate_metrics(
                    torch.from_numpy(season_preds), torch.from_numpy(season_targets),
                    torch.from_numpy(season_tigge)
                )
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {season} Test Metrics (STDGN):",
                      seasonal_metrics[season])

        all_metrics = {'Yearly': metrics_all, **seasonal_metrics}
        with open('STDGN_test_metrics.json', 'w') as f:
            json.dump(all_metrics, f)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test metrics saved as STDGN_test_metrics.json")

        monthly_metrics = {}
        months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in range(1, 13):
            month_mask = (pd.to_datetime(all_dates).month == month)
            if np.sum(month_mask) > 0:
                month_preds = all_preds[month_mask]
                month_targets = all_targets[month_mask]
                month_tigge = all_tigge_wind[month_mask]
                monthly_metrics[month] = calculate_metrics(
                    torch.from_numpy(month_preds), torch.from_numpy(month_targets),
                    torch.from_numpy(month_tigge)
                )
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Month {months_names[month - 1]} Test Metrics (STDGN):",
                      monthly_metrics[month])

        monthly_metrics_dict = {str(k): v for k, v in monthly_metrics.items()}
        with open('STDGN_test_monthly_metrics.json', 'w') as f:
            json.dump(monthly_metrics_dict, f)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monthly test metrics saved as STDGN_test_monthly_metrics.json")


# ==============================================================================
# 4. 主函数
# ==============================================================================
if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Program started")
    set_seed(42)

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    # 模型参数，与训练时保持一致
    batch_size = 8
    seq_len = 4
    in_channels = 8 + 3
    hidden_dim = 32
    out_channels = 1
    num_nodes = H * W
    k_neighbors = 8

    # 加载数据
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets...")
    test_ds = WindDatasetSTDGN("./show_relevance_visualization/test.nc", H, W, seq_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Datasets loaded successfully")

    # 加载模型
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
    model = STDGN(
        in_channels=in_channels, hidden_dim=hidden_dim, out_channels=out_channels,
        num_nodes=num_nodes, k_neighbors=k_neighbors, dropout_rate=0.3
    ).to(device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model initialized successfully")
    model.load_state_dict(torch.load('checkpoints/best_model_STDGN.pth'))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model weights loaded successfully.")

    # 准备样本用于性能测试
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Preparing sample for GFLOPs and forecast time calculation...")
    sample = next(iter(test_loader))
    spatial_sequence_sample = sample['spatial_sequence'][0:1].to(device)
    spatial_sequence_sample = spatial_sequence_sample.permute(0, 2, 1, 3, 4)  # 调整维度顺序

    # ==============================================================================
    # 5. 模型复杂度分析
    # ==============================================================================
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== CALCULATING MODEL COMPLEXITY =====")


    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params


    def estimate_stdgn_flops(model, input_sample):
        """估算STDGN模型的FLOPs"""
        batch_size, seq_len, C, H, W = input_sample.shape
        num_nodes = H * W

        total_flops = 0

        print(f"Calculating FLOPs for STDGN layers...")

        # GCN layers FLOPs
        # GCN1: input_dim -> hidden_dim
        gcn1_flops_per_node = C * model.hidden_dim * 2  # mult + add
        gcn1_flops = gcn1_flops_per_node * num_nodes * batch_size * seq_len
        total_flops += gcn1_flops
        print(f"  GCN1: {gcn1_flops:,} FLOPs")

        # GCN2: hidden_dim -> hidden_dim
        gcn2_flops_per_node = model.hidden_dim * model.hidden_dim * 2
        gcn2_flops = gcn2_flops_per_node * num_nodes * batch_size * seq_len
        total_flops += gcn2_flops
        print(f"  GCN2: {gcn2_flops:,} FLOPs")

        # GRU FLOPs (简化估算)
        # GRU has 3 gates: reset, update, new
        # Each gate: input_size * hidden_size + hidden_size * hidden_size
        input_size = model.hidden_dim
        hidden_size = model.hidden_dim
        gru_flops_per_step = 3 * (input_size * hidden_size + hidden_size * hidden_size) * 2
        gru_total_flops = gru_flops_per_step * num_nodes * batch_size * seq_len
        total_flops += gru_total_flops
        print(f"  GRU: {gru_total_flops:,} FLOPs")

        # Prediction layer FLOPs
        pred_flops = model.hidden_dim * out_channels * 2 * num_nodes * batch_size
        total_flops += pred_flops
        print(f"  Prediction layer: {pred_flops:,} FLOPs")

        # Graph construction FLOPs (简化估算)
        # K-NN search and adjacency matrix construction
        graph_flops = num_nodes * num_nodes * 2  # distance calculation
        graph_flops += num_nodes * k_neighbors * 2  # K-NN selection
        total_flops += graph_flops
        print(f"  Graph construction: {graph_flops:,} FLOPs")

        return total_flops


    model.eval()
    total_flops = estimate_stdgn_flops(model, spatial_sequence_sample)
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

    # ==============================================================================
    # 6. 推理性能测量
    # ==============================================================================
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MEASURING FORECAST TIME =====")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warming up model...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(spatial_sequence_sample)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting inference time measurement...")
    num_runs = 50  # STDGN比较慢，减少测试次数
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(spatial_sequence_sample)

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
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model: STDGN")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GFLOPs: {gflops:.3f}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Forecast time: {forecast_time_ms:.2f} ms")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Parameters: {total_params / 1e6:.2f}M")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model size: {total_params * 4 / (1024 ** 2):.1f} MB")

    # ==============================================================================
    # 7. 开始测试
    # ==============================================================================
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Testing...")
    test_model(model, test_loader, device, H, W, seq_len, batch_size)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing completed!")