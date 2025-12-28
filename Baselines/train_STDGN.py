import torch
import torch.nn as nn
import torch.nn.functional as F
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
import math


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 手动实现的GCN层
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: (batch_size, num_nodes, features) 或 (num_nodes, features)
        adj: (num_nodes, num_nodes)
        """
        if x.dim() == 3:  # batch情况: (batch_size, num_nodes, features)
            # 对每个batch分别计算
            batch_size = x.size(0)
            outputs = []
            for b in range(batch_size):
                support = torch.mm(adj, x[b])  # (num_nodes, features)
                output = self.linear(support)
                outputs.append(output)
            return torch.stack(outputs, dim=0)  # (batch_size, num_nodes, out_features)
        else:  # 单个样本: (num_nodes, features)
            support = torch.mm(adj, x)
            return self.linear(support)


# STDGN模型定义
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
        """基于固定的空间位置创建静态邻接矩阵"""
        num_nodes = H * W
        adj = torch.zeros(num_nodes, num_nodes, device=device)

        # 预计算所有节点的H, W坐标
        node_coords = torch.tensor([(i // W, i % W) for i in range(num_nodes)], device=device, dtype=torch.float)

        # 计算所有节点对之间的曼哈顿距离
        dist_matrix = torch.cdist(node_coords, node_coords, p=1)

        # 创建邻接矩阵，连接k个最近的邻居
        _, knn_indices = torch.topk(dist_matrix, self.k_neighbors + 1, largest=False)

        for node_idx in range(num_nodes):
            neighbors = knn_indices[node_idx, 1:]  # 排除自己
            adj[node_idx, neighbors] = 1.0

        # 对称化并添加自环
        adj = adj + adj.T
        adj[adj > 0] = 1.0
        adj = adj + torch.eye(num_nodes, device=device)

        return adj

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        num_nodes = H * W
        # 使用 reshape 代替 view
        x = x.reshape(batch_size, seq_len, num_nodes, C)

        # 创建静态图
        if not hasattr(self, 'adj_matrix') or self.adj_matrix.device != x.device:
            self.adj_matrix = self._create_spatial_graph(H, W, x.device)

        # 初始化GRU隐藏状态
        h_0 = torch.zeros(1, batch_size, num_nodes, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            current_x_t = x[:, t, :, :]  # (batch_size, num_nodes, C)

            # 图卷积，正确传递参数
            out = F.relu(self.gcn1(current_x_t, self.adj_matrix))  # (batch_size, num_nodes, hidden_dim)
            out = self.dropout(out)
            out = self.gcn2(out, self.adj_matrix)  # (batch_size, num_nodes, hidden_dim)

            # GRU处理
            # 使用 reshape 代替 view
            out_flat = out.reshape(batch_size * num_nodes, 1, self.hidden_dim)
            h_0_flat = h_0.reshape(1, batch_size * num_nodes, self.hidden_dim)

            _, h_t_flat = self.gru(out_flat, h_0_flat)

            # 使用 reshape 代替 view
            h_0 = h_t_flat.reshape(1, batch_size, num_nodes, self.hidden_dim)

        final_spatial_features = h_0.squeeze(0)  # (batch_size, num_nodes, hidden_dim)
        output = self.pred_layer(final_spatial_features)  # (batch_size, num_nodes, out_channels)

        # 使用 reshape 代替 view
        output = output.reshape(batch_size, H, W, out_channels).permute(0, 3, 1, 2)

        return output


class WindDatasetSTDGN(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)

        time_data = self.ds['time_features'].values
        times = pd.to_datetime({
            'year': time_data[:, 0], 'month': time_data[:, 1],
            'day': time_data[:, 2], 'hour': time_data[:, 3]
        })
        self.ds = self.ds.assign_coords(time=("sample", times)).sortby('time')
        self.time_points = np.unique(self.ds.time.values)
        self.T = len(self.time_points)

        self.sample_indices = np.arange(self.T - self.seq_len + 1)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset initialized with {len(self.sample_indices)} sequences")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        actual_idx = self.sample_indices[idx]

        # 获取目标时间点
        target_time_point = self.time_points[actual_idx + self.seq_len - 1]

        # 创建一个列表来存储序列中的每个时间步数据
        tigge_list = []
        dem_list = []

        # 循环获取序列中的每个时间步
        for t_offset in range(self.seq_len):
            time_point = self.time_points[actual_idx + t_offset]

            # 使用时间点作为 mask 来筛选数据
            mask = self.ds.time == time_point

            # 筛选数据并重塑为空间形状
            tigge_spatial = self.ds['X_tigge'].sel(sample=mask).values.reshape(self.H, self.W, 8)
            dem_spatial = self.ds['X_dem'].sel(sample=mask).values.reshape(self.H, self.W, 3)

            tigge_list.append(tigge_spatial)
            dem_list.append(dem_spatial)

        # 获取目标数据
        target_mask = self.ds.time == target_time_point
        target = self.ds['y'].sel(sample=target_mask).values.reshape(self.H, self.W)

        # 将列表中的数据堆叠成 numpy 数组
        tigge_sequence = np.stack(tigge_list, axis=0)  # (seq_len, H, W, 8)
        dem_sequence = np.stack(dem_list, axis=0)  # (seq_len, H, W, 3)

        # 拼接特征
        spatial_sequence = np.concatenate([tigge_sequence, dem_sequence], axis=-1)  # (seq_len, H, W, 11)

        # 转换为 PyTorch 张量
        spatial_sequence = torch.from_numpy(spatial_sequence).float().permute(3, 0, 1, 2)  # (C, seq_len, H, W)
        target = torch.from_numpy(target).float()

        return {'spatial_sequence': spatial_sequence, 'target': target}


def train_model(model, train_loader, val_loader, device, epochs=20, learning_rate=0.0005, weight_decay=0.001,
                patience=7):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started training process")
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-7,
                                                           verbose=True)
    scaler = GradScaler()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_mae, train_rmse, train_r2 = [], [], []
    val_mae, val_rmse, val_r2 = [], [], []
    patience_counter = 0

    os.makedirs('checkpoints', exist_ok=True)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch + 1}/{epochs}")
        model.train()

        train_loss = 0.0
        batch_count = 0
        train_preds, train_targets = [], []

        for batch_idx, batch in enumerate(train_loader):
            try:
                spatial_sequence = batch['spatial_sequence'].to(device)
                target = batch['target'].to(device)
                spatial_sequence = spatial_sequence.permute(0, 2, 1, 3, 4)

                with autocast():
                    output = model(spatial_sequence)
                    loss = criterion(output.squeeze(1), target)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                train_loss += loss.item() * spatial_sequence.size(0)
                batch_count += 1
                train_preds.append(output.detach().cpu())
                train_targets.append(target.detach().cpu())

                if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                    progress = (batch_idx + 1) / len(train_loader) * 100
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1}/{epochs}, "
                          f"Batch {batch_idx + 1}/{len(train_loader)} ({progress:.1f}%), "
                          f"Loss: {loss.item():.6f}, Grad norm: {grad_norm:.4f}")
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing batch {batch_idx}: {str(e)}")
                continue

        train_loss = train_loss / len(train_loader.dataset) if batch_count > 0 else float('inf')
        train_losses.append(train_loss)

        # 计算训练指标
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing training metrics")
        train_preds_cat = torch.cat(train_preds).numpy().flatten()
        train_targets_cat = torch.cat(train_targets).numpy().flatten()
        train_mae_val = np.mean(np.abs(train_preds_cat - train_targets_cat))
        train_rmse_val = np.sqrt(np.mean((train_preds_cat - train_targets_cat) ** 2))
        train_mean = np.mean(train_targets_cat)
        ss_tot = np.sum((train_targets_cat - train_mean) ** 2)
        ss_res = np.sum((train_targets_cat - train_preds_cat) ** 2)
        train_r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        train_mae.append(train_mae_val)
        train_rmse.append(train_rmse_val)
        train_r2.append(train_r2_val)
        print(
            f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, MAE: {train_mae_val:.4f}, RMSE: {train_rmse_val:.4f}, R²: {train_r2_val:.4f}')

        # Validation
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting validation")
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    spatial_sequence = batch['spatial_sequence'].to(device)
                    target = batch['target'].to(device)
                    spatial_sequence = spatial_sequence.permute(0, 2, 1, 3, 4)

                    with autocast():
                        output = model(spatial_sequence)
                        loss = criterion(output.squeeze(1), target)

                    val_loss += loss.item() * spatial_sequence.size(0)
                    val_batch_count += 1
                    val_preds.append(output.detach().cpu())
                    val_targets.append(target.detach().cpu())

                    if batch_idx % 50 == 0 or batch_idx == len(val_loader) - 1:
                        progress = (batch_idx + 1) / len(val_loader) * 100
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation progress: {progress:.1f}%")

                except Exception as e:
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing validation batch {batch_idx}: {str(e)}")
                    continue

        val_loss = val_loss / len(val_loader.dataset) if val_batch_count > 0 else float('inf')
        val_losses.append(val_loss)

        # 计算验证指标
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing validation metrics")
        val_preds_cat = torch.cat(val_preds).numpy().flatten()
        val_targets_cat = torch.cat(val_targets).numpy().flatten()
        val_mae_val = np.mean(np.abs(val_preds_cat - val_targets_cat))
        val_rmse_val = np.sqrt(np.mean((val_preds_cat - val_targets_cat) ** 2))
        val_mean = np.mean(val_targets_cat)
        ss_tot = np.sum((val_targets_cat - val_mean) ** 2)
        ss_res = np.sum((val_targets_cat - val_preds_cat) ** 2)
        val_r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        val_mae.append(val_mae_val)
        val_rmse.append(val_rmse_val)
        val_r2.append(val_r2_val)
        print(
            f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Validation Loss: {val_loss:.6f}, MAE: {val_mae_val:.4f}, RMSE: {val_rmse_val:.4f}, R²: {val_r2_val:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model_STDGN.pth')
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Model saved with val loss: {best_val_loss:.6f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Early stopping at epoch {epoch + 1}')
                break

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        remaining_epochs = epochs - (epoch + 1)
        est_remaining_time = remaining_epochs * epoch_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1} completed in {epoch_time / 60:.2f} minutes")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Estimated time remaining: {est_remaining_time / 3600:.2f} hours")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed in {total_time / 3600:.2f} hours")

    # 生成可视化图表
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating visualization plots")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - STDGN')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_STDGN.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loss curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_mae) + 1), val_mae, label='Validation MAE', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error - STDGN')
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_curve_STDGN.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MAE curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label='Validation RMSE', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error - STDGN')
    plt.legend()
    plt.grid(True)
    plt.savefig('rmse_curve_STDGN.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RMSE curve saved")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_r2) + 1), train_r2, label='Train R²', color='blue', linewidth=2)
    plt.plot(range(1, len(val_r2) + 1), val_r2, label='Validation R²', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R² Score - STDGN')
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_curve_STDGN.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] R² curve saved")

    print(
        f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Training metrics visualizations saved as: loss_curve_STDGN.png, mae_curve_STDGN.png, rmse_curve_STDGN.png, r2_curve_STDGN.png')


if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Program started")
    set_seed(42)

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    # 优化后的参数配置
    batch_size = 8  # 适中的batch size
    seq_len = 4
    epochs = 20
    learning_rate = 0.0005
    weight_decay = 0.001
    patience = 7

    in_channels = 8 + 3
    hidden_dim = 32
    out_channels = 1
    num_nodes = H * W
    k_neighbors = 8

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets...")
    train_ds = WindDatasetSTDGN("./show_relevance_visualization/train.nc", H, W, seq_len)
    val_ds = WindDatasetSTDGN("./show_relevance_visualization/val.nc", H, W, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Data loaders created successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model")
    model = STDGN(
        in_channels=in_channels, hidden_dim=hidden_dim, out_channels=out_channels,
        num_nodes=num_nodes, k_neighbors=k_neighbors, dropout_rate=0.3
    ).to(device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model initialized successfully")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Training")
    train_model(model, train_loader, val_loader, device, epochs=epochs, learning_rate=learning_rate,
                weight_decay=weight_decay, patience=patience)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed!")