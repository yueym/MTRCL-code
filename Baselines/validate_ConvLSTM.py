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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dropout_rate=0.4, output_dim=None):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim if output_dim is not None else hidden_dim

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dim, self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(self.hidden_dim // 2, self.output_dim, kernel_size=1)
        )

        for m in self.output_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_seq, time_features_seq):
        b, seq_len, c, height, width = x_seq.size()  # 重命名 h, w 为 height, width
        h_states = []  # 使用 h_states 而不是 h
        c_states = []  # 使用 c_states 而不是 c

        for i in range(self.num_layers):
            h_states.append(torch.zeros(b, self.hidden_dim, height, width).to(x_seq.device))
            c_states.append(torch.zeros(b, self.hidden_dim, height, width).to(x_seq.device))

        for t in range(seq_len):
            x_t = x_seq[:, t]
            time_t = time_features_seq[:, t]
            time_emb = time_t.view(b, -1, 1, 1).expand(-1, -1, height, width)
            x_t = torch.cat([x_t, time_emb], dim=1)

            for i in range(self.num_layers):
                layer_input = x_t if i == 0 else h_states[i - 1]
                h_states[i], c_states[i] = self.cell_list[i](layer_input, (h_states[i], c_states[i]))
                if i < self.num_layers - 1 and self.dropout_rate > 0:
                    h_states[i] = nn.functional.dropout(h_states[i], p=self.dropout_rate, training=self.training)

        output = self.output_layer(h_states[-1])
        return output


class GatedFusion(nn.Module):
    def __init__(self, C1):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(C1, C1 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(C1 // 2, C1, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, convlstm_out):
        gate = self.gate(convlstm_out)
        output = gate * convlstm_out
        return output


# MLP模块（无变化）
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


class WindSpeedPredictor(nn.Module):
    def __init__(self, H, W, tigge_features=8, dropout_rate=0.4, convlstm_hidden_dim=64, cbam_reduction=16):
        super(WindSpeedPredictor, self).__init__()
        self.H = H
        self.W = W

        self.convlstm = ConvLSTM(
            input_dim=tigge_features + 5,
            hidden_dim=convlstm_hidden_dim,
            kernel_size=3,
            num_layers=2,
            dropout_rate=dropout_rate,
            output_dim=convlstm_hidden_dim
        )
        self.gated_fusion = GatedFusion(convlstm_hidden_dim)
        self.mlp = MLP(convlstm_hidden_dim, dropout_rate=dropout_rate)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self, tigge_spatial, dem_spatial, tigge_seq, time_features_t, time_features_seq):

        convlstm_out = self.convlstm(tigge_seq, time_features_seq)
        fused = self.gated_fusion(convlstm_out)
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
    return {
        'FA_pred': FA_pred, 'RMSE_pred': RMSE_pred, 'MAE_pred': MAE_pred,
        'rRMSE_pred': rRMSE_pred, 'rMAE_pred': rMAE_pred, 'R_pred': R_pred, 'R2_pred': R2_pred,
        'FA_tigge': FA_tigge, 'RMSE_tigge': RMSE_tigge, 'MAE_tigge': MAE_tigge,
        'rRMSE_tigge': rRMSE_tigge, 'rMAE_tigge': rMAE_tigge, 'R_tigge': R_tigge, 'R2_tigge': R2_tigge
    }


# 验证函数（修改）
def validate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_tigge_wind = []
    all_dates = []
    criterion = nn.SmoothL1Loss()

    # 加载标准化器（无变化）
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
                val_loss += loss.item() * tigge_seq.size(0)
                batch_count += 1
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
                all_tigge_wind.append(batch['tigge_spatial'][:, 0, :, :].cpu())

                actual_indices = val_loader.dataset.sample_indices[batch_idx * val_loader.batch_size:
                                                                   (batch_idx + 1) * val_loader.batch_size]
                batch_dates = [val_loader.dataset.time_points[idx + val_loader.dataset.seq_len - 1] for idx in
                               actual_indices
                               if idx + val_loader.dataset.seq_len - 1 < len(val_loader.dataset.time_points)]
                all_dates.extend(batch_dates)

                if batch_idx % 100 == 0 or batch_idx == len(val_loader) - 1:
                    progress = (batch_idx + 1) / len(val_loader) * 100
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation progress: {progress:.1f}%")

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing validation batch {batch_idx}: {str(e)}")
                continue

        val_loss = val_loss / len(val_loader.dataset) if batch_count > 0 else float('inf')
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Validation Loss: {val_loss:.4f}')

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

        np.save('ConvLSTM_val_preds.npy', all_preds_orig)
        np.save('ConvLSTM_val_targets.npy', all_targets_orig)
        np.save('ConvLSTM_val_tigge_wind.npy', all_tigge_wind_orig)
        np.save('ConvLSTM_val_dates.npy', all_dates)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation predictions, targets, TIGGE wind, and dates saved as ConvLSTM_*.npy files.")

        metrics_all = calculate_metrics(
            torch.from_numpy(all_preds_orig),
            torch.from_numpy(all_targets_orig),
            torch.from_numpy(all_tigge_wind_orig)
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Yearly Validation Metrics (ConvLSTM):", metrics_all)

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
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {season} Validation Metrics (ConvLSTM):",
                  seasonal_metrics[season])

        all_metrics = {'Yearly': metrics_all, **seasonal_metrics}
        with open('ConvLSTM_val_metrics.json', 'w') as f:
            json.dump(all_metrics, f)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation metrics saved as ConvLSTM_val_metrics.json")

        monthly_metrics = {}
        months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = list(range(1, 13))
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
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Month {months_names[month - 1]} Validation Metrics (ConvLSTM):",
                    monthly_metrics[month])

        monthly_metrics_dict = {str(k): v for k, v in monthly_metrics.items()}
        with open('ConvLSTM_val_monthly_metrics.json', 'w') as f:
            json.dump(monthly_metrics_dict, f)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monthly validation metrics saved as ConvLSTM_val_monthly_metrics.json")

# 主函数（修改）
if __name__ == "__main__":
        H, W = 48, 96
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets...")
        batch_size = 16
        val_ds = WindDataset("./show_relevance_visualization/val.nc", H, W)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Datasets loaded successfully")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
        model = WindSpeedPredictor(
            H, W,
            tigge_features=8,
            dropout_rate=0.4,
            convlstm_hidden_dim=64,
            cbam_reduction=16
        ).to(device)
        model.load_state_dict(torch.load('checkpoints/best_model_convlstm.pth'))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Validation...")
        validate_model(model, val_loader, device)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation completed!")


