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


class ResNet(nn.Module):
    def __init__(self, in_channels=11, dropout_rate=0.3):
        super(ResNet, self).__init__()
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
        return out


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

    def forward(self, resnet_out):
        gate = self.gate(resnet_out)
        output = gate * resnet_out
        return output


class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
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
    def __init__(self, H, W, tigge_features=8, dropout_rate=0.3,
                 cbam_reduction=16):
        super(WindSpeedPredictor, self).__init__()
        self.H = H
        self.W = W
        self.resnet = ResNet(in_channels=tigge_features + 3, dropout_rate=dropout_rate)
        self.gated_fusion = GatedFusion(56)
        self.mlp = MLP(56, dropout_rate=dropout_rate)
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

    def forward(self, tigge_spatial, dem_spatial, time_features_t):
        b = tigge_spatial.size(0)
        spatial_input = torch.cat([tigge_spatial, dem_spatial], dim=1)
        time_emb = self.time_embed(time_features_t).view(b, 56, 1, 1)
        resnet_out = self.resnet(spatial_input, time_emb)
        fused = self.gated_fusion(resnet_out)
        pred = self.mlp(fused)
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
        time_t = self.time_points[t]
        mask_t = self.ds.time == time_t
        tigge_spatial = self.ds['X_tigge'].sel(sample=mask_t).values.reshape(self.H, self.W, 8)
        dem_spatial = self.ds['X_dem'].sel(sample=mask_t).values.reshape(self.H, self.W, 3)
        target = self.ds['y'].sel(sample=mask_t).values.reshape(self.H, self.W)
        time_features_t = self.ds['time_features_normalized'].sel(sample=mask_t).values[0]
        return {
            'tigge_spatial': torch.from_numpy(tigge_spatial).float().permute(2, 0, 1),
            'dem_spatial': torch.from_numpy(dem_spatial).float().permute(2, 0, 1),
            'time_features_t': torch.from_numpy(time_features_t).float(),
            'target': torch.from_numpy(target).float()
        }

def train_model(model, train_loader, val_loader, device,
                epochs=10, learning_rate=0.0002, weight_decay=0.008,
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
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_t shape: {first_batch['time_features_t'].shape}")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - target shape: {first_batch['target'].shape}")  # 修改点 14：移除 tigge_seq 和 time_features_seq 的调试输出

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
                time_features_t = batch['time_features_t'].to(device)
                target = batch['target'].to(device)

                with autocast():
                    output = model(tigge_spatial, dem_spatial,
                                   time_features_t)
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

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting除了: validation")
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
                    time_features_t = batch['time_features_t'].to(device)
                    target = batch['target'].to(device)
                    with autocast():
                        output = model(tigge_spatial, dem_spatial,
                                       time_features_t)
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
            torch.save(model.state_dict(), 'checkpoints/best_model_ResNet_TE.pth')
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
    plt.savefig('loss_curve_ResNet_TE.png')
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
    plt.savefig('mae_curve_ResNet_TE.png')
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
    plt.savefig('rmse_curve_ResNet_TE.png')
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
    plt.savefig('r2_curve_ResNet_TE.png')
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] R² curve saved")

    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Training metrics visualizations saved as: loss_curve.png, mae_curve.png, rmse_curve.png, r2_curve.png')


if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Program started")

    set_seed(42)

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Setting up training parameters")
    batch_size = 28
    epochs = 10
    learning_rate = 0.0002
    weight_decay = 0.008
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
        dropout_rate=0.3,
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
