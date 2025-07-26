# ==================== 环境准备 ====================
import os
import numpy as np
import xarray as xr
import pandas as pd
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import joblib
import gc  # 添加垃圾回收模块，处理大数据集
import time  # 添加时间模块，用于跟踪进度
from datetime import datetime  # 添加日期时间模块，用于记录时间戳

# ==================== 路径设置 ====================
raw_data_path = {
    "ERA5": "./final_processed/ERA5/",
    "TIGGE": "./final_processed/TIGGE/",
    "DEM": "./final_processed/DEM/",
}
processed_path = "./show_relevance_visualization/"


# ==================== 辅助函数 ====================
def create_directories():
    """创建必要的目录结构"""
    os.makedirs(processed_path, exist_ok=True)
    print(f"已创建输出目录: {processed_path}")


def plot_feature_importance(importance_df, filename, title):
    """绘制特征重要性柱状图，改进可视化以处理极端值"""
    try:
        print(f"正在绘制特征重要性图: {filename}...")

        # 数据验证
        if importance_df is None or importance_df.empty:
            print(f"警告: {filename} 的重要性数据为空，跳过绘图")
            return

        if 'importance' not in importance_df.columns:
            print(f"警告: {filename} 缺少 'importance' 列，跳过绘图")
            return

        # 计算相对重要性（百分比）而不是简单归一化到0-100
        total_importance = importance_df['importance'].sum()
        if total_importance == 0:
            print(f"警告: {filename} 的总重要性为0，跳过绘图")
            return

        importance_df['percentage'] = (importance_df['importance'] / total_importance) * 100

        # 根据特征数量动态调整图表高度
        fig_height = max(10, len(importance_df) * 0.5)  # 确保足够的高度显示所有特征

        fig_width = 10  # 原来是20，现在改为12，横向压缩

        # 图1：相对重要性（百分比）- 修改标题和样式
        plt.figure(figsize=(fig_width, fig_height))

        # 设置全局字体为Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'

        # 修改：使用统一的蓝色而非渐变色
        # 压缩横向比例，但保持排名不变
        scaled_percentages = np.sqrt(importance_df['percentage']) * (60 / np.sqrt(100))  # 将最大值从100压缩到60
        bars = plt.barh(importance_df['feature'], scaled_percentages, color='blue', height=0.8)

        # 修改：更新标题和轴标签
        if "TIGGE" in title:
            plt.title("Feature Importance (ECMWF-TIGGE)", fontsize=22, fontweight='bold')
        elif "DEM" in title:
            plt.title("Feature Importance (ASTER GDEM V3)", fontsize=22, fontweight='bold')
        else:
            plt.title(title, fontsize=22, fontweight='bold')

        plt.xlabel("Importance (%)", fontsize=22, fontweight='bold')
        plt.ylabel("Feature", fontsize=22, fontweight='bold')
        plt.gca().invert_yaxis()  # 最重要的特征在顶部

        # 修改：去除网格线
        plt.grid(False)

        # 修改：加粗边框
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)

        # 设置刻度标签字体
        plt.xticks(fontsize=18, fontweight='bold')
        plt.yticks(fontsize=18, fontweight='bold')

        # 修改：设置x轴范围，确保图表比例合适
        plt.xlim(0, 65)  # 设置一个合适的上限，给最大值留一些空间

        plt.tight_layout()
        plt.savefig(os.path.join(processed_path, f"{filename}_relative.png"), dpi=1000)
        plt.close()

        # 不再生成对数和平方根变换的图
        print(f"特征重要性图已保存: {filename}")

    except Exception as e:
        print(f"绘制特征重要性图时出错 ({filename}): {e}")
        plt.close('all')  # 确保关闭所有图形


def plot_feature_count_metrics(metrics_dict, filename, title):
    """绘制特征数量与MAE/RMSE的关系图，动态刻度并返回数据"""
    try:
        print(f"正在绘制特征数量评估图: {filename}...")

        # 数据验证
        if not metrics_dict:
            print(f"警告: {filename} 的metrics数据为空，跳过绘图")
            return [], [], []

        feature_counts = sorted(metrics_dict.keys())
        mae_values = [metrics_dict[n]['MAE'] for n in feature_counts]
        rmse_values = [metrics_dict[n]['RMSE'] for n in feature_counts]

        plt.figure(figsize=(12, 8))

        # 设置全局字体为Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'

        plt.plot(feature_counts, mae_values, marker='o', label='MAE', color='red', linewidth=2)
        plt.plot(feature_counts, rmse_values, marker='o', label='RMSE', color='blue', linewidth=2)

        # 修改：更新标题和轴标签
        if "TIGGE" in title:
            plt.title("Feature Selection (ECMWF-TIGGE)", fontsize=22, fontweight='bold')
        elif "DEM" in title:
            plt.title("Feature Selection (ASTER GDEM V3)", fontsize=22, fontweight='bold')
        else:
            plt.title(title, fontsize=22, fontweight='bold')

        plt.xlabel("Feature Count", fontsize=22, fontweight='bold')
        plt.ylabel("Value (m/s)", fontsize=22, fontweight='bold')

        # 修改：设置x轴范围，确保右侧有额外空间且保留所有刻度
        max_feature = max(feature_counts)
        extra_space = 1 if max_feature <= 4 else 2  # 为小数据集添加1个单位，大数据集添加2个单位

        # 设置x轴范围，确保0点在原点且保留所有刻度
        plt.xlim(-0.5, max_feature + extra_space)

        if "TIGGE" in title:
            plt.xticks(np.arange(0, max_feature + extra_space + 1, 2), fontsize=18, fontweight='bold')
        else:
            plt.xticks(np.arange(0, max_feature + extra_space + 1, 1), fontsize=18, fontweight='bold')

        plt.yticks(fontsize=18, fontweight='bold')

        # 修改：去除网格线
        plt.grid(False)

        # 修改：加粗边框
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)

        # 设置图例字体
        legend = plt.legend(fontsize=14)
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # 不再使用margins，因为它可能导致刻度问题
        # plt.margins(x=0)

        plt.tight_layout()
        plt.savefig(os.path.join(processed_path, filename), dpi=1000)
        plt.close()
        print(f"特征数量评估图已保存: {filename}")
        return feature_counts, mae_values, rmse_values

    except Exception as e:
        print(f"绘制特征数量评估图时出错 ({filename}): {e}")
        plt.close('all')
        return [], [], []


def plot_combined_tigge_analysis(importance_df, metrics_dict, filename):
    """绘制TIGGE组合分析图：左侧特征重要性，右侧特征数量与MAE/RMSE关系"""
    try:
        print(f"正在绘制TIGGE组合分析图: {filename}...")

        # 数据验证
        if importance_df is None or importance_df.empty:
            print(f"警告: TIGGE重要性数据为空，跳过组合分析图")
            return

        if not metrics_dict:
            print(f"警告: TIGGE metrics数据为空，跳过组合分析图")
            return

        # 设置全局字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'

        # 创建图形和子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # =============== 左侧：特征重要性图 ===============
        # 计算相对重要性（百分比）
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['percentage'] = (importance_df['importance'] / total_importance) * 100

            # 压缩横向比例
            scaled_percentages = np.sqrt(importance_df['percentage']) * (60 / np.sqrt(100))

            # 绘制横向柱状图
            bars1 = ax1.barh(importance_df['feature'], scaled_percentages, color='blue', height=0.8)

            # 设置左侧子图
            ax1.set_title("(a) Feature Importance", fontsize=24, fontweight='bold')
            ax1.set_xlabel("Importance (%)", fontsize=24, fontweight='bold')
            ax1.set_ylabel("Feature", fontsize=24, fontweight='bold')
            ax1.invert_yaxis()  # 最重要的特征在顶部
            ax1.grid(False)
            ax1.set_xlim(0, 65)

            # 设置刻度字体大小
            ax1.tick_params(axis='x', labelsize=20, which='major')
            ax1.tick_params(axis='y', labelsize=20, which='major')

            # 加粗边框
            for spine in ax1.spines.values():
                spine.set_linewidth(1.5)

        # =============== 右侧：特征数量与MAE/RMSE关系图 ===============
        feature_counts = sorted(metrics_dict.keys())
        mae_values = [metrics_dict[n]['MAE'] for n in feature_counts]
        rmse_values = [metrics_dict[n]['RMSE'] for n in feature_counts]

        # 绘制线图
        ax2.plot(feature_counts, mae_values, marker='o', label='MAE', color='red', linewidth=2, markersize=6)
        ax2.plot(feature_counts, rmse_values, marker='o', label='RMSE', color='blue', linewidth=2, markersize=6)

        # 设置右侧子图
        ax2.set_title("(b) Feature Selection", fontsize=24, fontweight='bold')
        ax2.set_xlabel("Feature Count", fontsize=24, fontweight='bold')
        ax2.set_ylabel("Value (m/s)", fontsize=24, fontweight='bold')

        # 设置x轴范围和刻度
        max_feature = max(feature_counts)
        extra_space = 2
        ax2.set_xlim(-0.5, max_feature + extra_space)
        ax2.set_xticks(np.arange(0, max_feature + extra_space + 1, 2))

        # 设置刻度字体大小
        ax2.tick_params(axis='x', labelsize=20, which='major')
        ax2.tick_params(axis='y', labelsize=20, which='major')

        ax2.grid(False)

        # 加粗边框
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)

        # 设置图例
        legend = ax2.legend(fontsize=20)
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # 调整子图间距
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(processed_path, filename), dpi=1000, bbox_inches='tight')
        plt.close()
        print(f"TIGGE组合分析图已保存: {filename}")

    except Exception as e:
        print(f"绘制TIGGE组合分析图时出错: {e}")
        plt.close('all')


def plot_combined_dem_analysis(importance_df, metrics_dict, filename):
    """绘制DEM组合分析图：左侧特征重要性，右侧特征数量与MAE/RMSE关系"""
    try:
        print(f"正在绘制DEM组合分析图: {filename}...")

        # 数据验证
        if importance_df is None or importance_df.empty:
            print(f"警告: DEM重要性数据为空，跳过组合分析图")
            return

        if not metrics_dict:
            print(f"警告: DEM metrics数据为空，跳过组合分析图")
            return

        # 设置全局字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'

        # 创建图形和子图，减小整体尺寸
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))  # 调整为 (15, 8)，缩小图形大小

        # =============== 左侧：特征重要性图 ===============
        # 计算相对重要性（百分比）
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['percentage'] = (importance_df['importance'] / total_importance) * 100

            # 压缩横向比例
            scaled_percentages = np.sqrt(importance_df['percentage']) * (60 / np.sqrt(100))

            # 绘制横向柱状图，减小柱子宽度
            bars1 = ax1.barh(importance_df['feature'], scaled_percentages, color='blue', height=0.4)  # 减小 height=0.6

            # 设置左侧子图
            ax1.set_title("(a) Feature Importance", fontsize=24, fontweight='bold')  # 减小字体大小
            ax1.set_xlabel("Importance (%)", fontsize=24, fontweight='bold')
            ax1.set_ylabel("Feature", fontsize=24, fontweight='bold')
            ax1.invert_yaxis()  # 最重要的特征在顶部
            ax1.grid(False)
            ax1.set_xlim(0, 65)

            # 设置刻度字体大小
            ax1.tick_params(axis='x', labelsize=20, which='major')  # 减小刻度字体
            ax1.tick_params(axis='y', labelsize=20, which='major')

            # 加粗边框
            for spine in ax1.spines.values():
                spine.set_linewidth(1.5)

        # =============== 右侧：特征数量与MAE/RMSE关系图 ===============
        feature_counts = sorted(metrics_dict.keys())
        mae_values = [metrics_dict[n]['MAE'] for n in feature_counts]
        rmse_values = [metrics_dict[n]['RMSE'] for n in feature_counts]

        # 绘制线图
        ax2.plot(feature_counts, mae_values, marker='o', label='MAE', color='red', linewidth=2, markersize=6)
        ax2.plot(feature_counts, rmse_values, marker='o', label='RMSE', color='blue', linewidth=2, markersize=6)

        # 设置右侧子图
        ax2.set_title("(b) Feature Selection", fontsize=24, fontweight='bold')  # 减小字体大小
        ax2.set_xlabel("Feature Count", fontsize=24, fontweight='bold')
        ax2.set_ylabel("Value (m/s)", fontsize=24, fontweight='bold')

        # 设置x轴范围和刻度，减小间距
        max_feature = max(feature_counts)
        extra_space = 0.5  # 减小额外空间
        ax2.set_xlim(-0.5, max_feature + extra_space)
        ax2.set_xticks(np.arange(0, max_feature + extra_space + 1, 0.5))  # 调整为步长0.5

        # 设置刻度字体大小
        ax2.tick_params(axis='x', labelsize=20, which='major')  # 减小刻度字体
        ax2.tick_params(axis='y', labelsize=20, which='major')

        ax2.grid(False)

        # 加粗边框
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)

        # 设置图例
        legend = ax2.legend(fontsize=20)  # 减小图例字体
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # 调整子图间距
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(processed_path, filename), dpi=1000, bbox_inches='tight')
        plt.close()
        print(f"DEM组合分析图已保存: {filename}")

    except Exception as e:
        print(f"绘制DEM组合分析图时出错: {e}")
        plt.close('all')


def find_stabilization_point(feature_counts, mae_values, rmse_values, threshold=0.01, window=1):
    """根据MAE和RMSE的变化率，找到趋于平稳的特征数量"""
    try:
        print(f"正在查找稳定点 (阈值={threshold},窗口={window})...")

        if len(feature_counts) < 2 or len(mae_values) < 2 or len(rmse_values) < 2:
            print("警告: 数据点不足，返回最大特征数")
            return feature_counts[-1] if feature_counts else 1

        mae_diffs = np.abs(np.diff(mae_values)) / np.array(mae_values[:-1])
        rmse_diffs = np.abs(np.diff(rmse_values)) / np.array(rmse_values[:-1])
        combined_diffs = (mae_diffs + rmse_diffs) / 2

        print(f"\n{feature_counts} 变化率分析:")
        for i, (n, diff) in enumerate(zip(feature_counts[1:], combined_diffs)):
            print(f"特征数 {n}: 变化率 = {diff:.4f}")

        # 针对TIGGE和DEM特征使用不同的阈值
        if len(feature_counts) > 10:  # TIGGE特征
            threshold = 0.007
        else:  # DEM特征
            threshold = 0.015

        for i in range(len(combined_diffs) - window + 1):
            window_avg = np.mean(combined_diffs[i:i + window])
            if window_avg < threshold:
                print(f"稳定点找到: 特征数 = {feature_counts[i + window]}, 变化率 = {window_avg:.4f}")
                return feature_counts[i + window]
        print(f"警告: 未找到低于阈值 {threshold} 的稳定点。所有特征都将被选择。")
        print(f"返回最大特征数: {feature_counts[-1]}")
        return feature_counts[-1]

    except Exception as e:
        print(f"查找稳定点时出错: {e}")
        return feature_counts[-1] if feature_counts else 1



def print_memory_usage(label=""):
    """打印当前内存使用情况"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)
    print(f"内存使用 {label}: {memory_gb:.2f} GB")


# 修改点3: 添加进度跟踪函数
def log_progress(message, start_time=None):
    """记录进度信息，包括时间戳和可选的耗时"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time:
        elapsed = time.time() - start_time
        elapsed_min = elapsed // 60
        elapsed_sec = elapsed % 60
        print(f"[{timestamp}] {message} (耗时: {int(elapsed_min)}分{int(elapsed_sec)}秒)")
    else:
        print(f"[{timestamp}] {message}")
    return time.time()


# ==================== 数据加载和划分 ====================
def load_and_split_data():
    start_time = time.time()
    print("\n=== 开始加载数据 ===")
    print_memory_usage("加载前")

    # 添加错误处理
    try:
        log_progress("加载ERA5数据...")
        era5 = xr.open_dataset(
            os.path.join(raw_data_path["ERA5"], "ERA5_final_2021_2024.nc"))
        log_progress("加载TIGGE数据...")
        tigge = xr.open_dataset(os.path.join(raw_data_path["TIGGE"], "TIGGE_final_2021_2024.nc"))
        log_progress("加载DEM数据...")
        dem = xr.open_dataset(os.path.join(raw_data_path["DEM"], "DEM_final.nc"))
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 - {e}")
        raise
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

    # 合并数据集
    log_progress("合并数据集...")
    merged_data = xr.merge([era5, tigge, dem], join='inner')
    print(f"合并数据时间范围: {merged_data.time.values.min()} 至 {merged_data.time.values.max()}")
    print(f"合并数据时间点数: {len(merged_data.time)}")

    # 按年份划分数据
    log_progress("按年份划分数据...")
    # 确保时间格式正确
    merged_data['time'] = pd.to_datetime(merged_data.time.values)

    # 按年份划分
    train_data = merged_data.sel(time=slice('2021-01-01', '2022-12-31'))
    val_data = merged_data.sel(time=slice('2023-01-01', '2023-12-31'))
    test_data = merged_data.sel(time=slice('2024-01-01', '2024-12-31'))

    print(f"训练集(2021-2022): {len(train_data.time)} 时间点")
    print(f"验证集(2023): {len(val_data.time)} 时间点")
    print(f"测试集(2024): {len(test_data.time)} 时间点")

    nan_count = np.isnan(merged_data['era5_wind_speed'].values).sum()
    print(f"合并数据中era5_wind_speed的NaN数量: {nan_count}")

    # 释放内存
    log_progress("释放合并数据内存...")
    del merged_data
    gc.collect()
    print_memory_usage("数据划分后")

    log_progress("数据加载和划分完成", start_time)

    return train_data, val_data, test_data, tigge, dem


# ==================== 数据标准化（提前） ====================
def standardize_data_before_selection(train_data, val_data, test_data, tigge, dem):
    """提前标准化TIGGE、DEM和目标变量到[0, 1]范围"""
    start_time = time.time()
    log_progress("开始数据标准化...")
    print_memory_usage("标准化前")

    log_progress("转换训练数据为DataFrame...")
    train_df = train_data.to_dataframe().reset_index().dropna(
        subset=['era5_wind_speed'])
    log_progress("转换验证数据为DataFrame...")
    val_df = val_data.to_dataframe().reset_index().dropna(
        subset=['era5_wind_speed'])
    log_progress("转换测试数据为DataFrame...")
    test_df = test_data.to_dataframe().reset_index().dropna(
        subset=['era5_wind_speed'])

    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    print(f"测试集样本数: {len(test_df)}")

    # 不再随机抽样，使用全部训练数据
    print(f"特征筛选使用全部训练样本: {len(train_df)}")

    log_progress("提取特征列...")
    time_params = ['year', 'hour', 'day', 'month', 'season']
    tigge_features = [var for var in tigge.data_vars if var not in time_params and var != 'time']
    if 'tigge_wind_speed' not in tigge_features and 'tigge_wind_speed' in tigge.data_vars:
        tigge_features.append('tigge_wind_speed')
    tigge_features = [f for f in tigge_features if f in train_df.columns]
    dem_features = [f for f in dem.data_vars if f in train_df.columns]

    # 检查特征是否存在
    if not tigge_features:
        raise ValueError("没有找到有效的TIGGE特征")
    if not dem_features:
        raise ValueError("没有找到有效的DEM特征")

    print("\n====原始TIGGE特征顺序====")
    for i, feature in enumerate(tigge_features):
        print(f"{i}: {feature}")
    if 'tigge_wind_speed' in tigge_features:
        wind_speed_idx = tigge_features.index('tigge_wind_speed')
        print(f"\n>>> 重要提示：tigge_wind_speed在原始特征中的索引位置:{wind_speed_idx} <<< ")
        print(f">>> 请在验证和测试代码中使用此索引值({wind_speed_idx})来获取正确的风速数据 <<<\n ")
    else:
        print("\n>>> 警告：tigge_wind_speed不在特征列表中")

    # TIGGE 特征标准化
    log_progress("标准化TIGGE特征...")
    scaler_X_tigge = MinMaxScaler(feature_range=(0, 1))
    X_train_tigge = scaler_X_tigge.fit_transform(train_df[tigge_features].fillna(0))
    X_val_tigge = scaler_X_tigge.transform(val_df[tigge_features].fillna(0))
    X_test_tigge = scaler_X_tigge.transform(test_df[tigge_features].fillna(0))

    # DEM 特征标准化
    log_progress("标准化DEM特征...")
    scaler_X_dem = MinMaxScaler(feature_range=(0, 1))
    X_train_dem = scaler_X_dem.fit_transform(train_df[dem_features].fillna(0))
    X_val_dem = scaler_X_dem.transform(val_df[dem_features].fillna(0))
    X_test_dem = scaler_X_dem.transform(test_df[dem_features].fillna(0))

    # 目标变量标准化
    log_progress("标准化目标变量...")
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = scaler_y.fit_transform(
        train_df['era5_wind_speed'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(
        val_df['era5_wind_speed'].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(
        test_df['era5_wind_speed'].values.reshape(-1, 1)).flatten()

    # 时间特征处理
    log_progress("处理时间特征...")
    time_features = {}
    for phase, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        df['time'] = pd.to_datetime(df['time'])
        time_cols = pd.DataFrame({
            'year': df['time'].dt.year,
            'month': df['time'].dt.month,
            'day': df['time'].dt.day,
            'hour': df['time'].dt.hour,
            'season': df['time'].dt.month % 12 // 3 + 1
        })
        time_features[phase] = time_cols.values.astype(np.int64)

    # 保存标准化器
    log_progress("保存标准化器...")
    joblib.dump(scaler_X_tigge, os.path.join(processed_path, 'tigge_feature_scaler.pkl'))
    joblib.dump(scaler_X_dem, os.path.join(processed_path, 'dem_feature_scaler.pkl'))
    joblib.dump(scaler_y, os.path.join(processed_path, 'target_scaler.pkl'))
    print(f"标准化器保存至路径: {processed_path}")

    # 释放内存
    log_progress("释放DataFrame内存...")
    del train_df, val_df, test_df
    gc.collect()
    log_progress("数据标准化完成", start_time)

    return (X_train_tigge, X_val_tigge, X_test_tigge,
            X_train_dem, X_val_dem, X_test_dem,
            y_train, y_val, y_test,
            time_features, tigge_features, dem_features)


def scientific_feature_selection(X_train, y_train, features, data_type="TIGGE", target_count=None):
    try:
        log_progress(f"开始{data_type}科学特征选择...")

        # 1. 使用两阶段训练获取更稳定的特征重要性排名
        # 第一阶段：使用较大样本进行初步特征重要性评估
        log_progress(f"第一阶段：初步特征重要性评估...")

        # 使用更大的样本量，最多使用全部数据，但不超过500,000个样本
        sample_size_stage1 = min(500000, X_train.shape[0])
        if sample_size_stage1 < X_train.shape[0]:
            indices_stage1 = np.random.RandomState(42).choice(X_train.shape[0], sample_size_stage1, replace=False)
            X_subset_stage1 = X_train[indices_stage1]
            y_subset_stage1 = y_train[indices_stage1]
        else:
            X_subset_stage1 = X_train
            y_subset_stage1 = y_train

        print(f"第一阶段使用样本数: {sample_size_stage1}")

        # 使用更稳定的参数配置
        model_stage1 = cb.CatBoostRegressor(
            iterations=1500,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            loss_function='RMSE',
            random_seed=42,
            early_stopping_rounds=100,
            task_type='GPU',
            devices='0',
            verbose=100
        )

        # 使用5折交叉验证获取初步特征重要性
        from sklearn.model_selection import KFold
        kf_stage1 = KFold(n_splits=5, shuffle=True, random_state=42)
        importances_stage1 = np.zeros(len(features))

        for fold, (train_idx, val_idx) in enumerate(kf_stage1.split(X_subset_stage1)):
            log_progress(f"第一阶段 - 交叉验证折 {fold + 1}/5...")
            X_train_cv, X_val_cv = X_subset_stage1[train_idx], X_subset_stage1[val_idx]
            y_train_cv, y_val_cv = y_subset_stage1[train_idx], y_subset_stage1[val_idx]

            model_stage1.fit(
                X_train_cv, y_train_cv,
                eval_set=(X_val_cv, y_val_cv),
                use_best_model=True
            )

            fold_importances = model_stage1.get_feature_importance()
            importances_stage1 += fold_importances

        # 平均每个折叠的重要性
        importances_stage1 /= 5

        # 根据第一阶段结果排序特征
        initial_importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances_stage1
        }).sort_values('importance', ascending=False)

        sorted_features = initial_importance_df['feature'].tolist()

        # 第二阶段：针对排序后的特征进行更精细的重要性评估
        log_progress(f"第二阶段：精细特征重要性评估...")

        # 使用全部训练数据
        model_stage2 = cb.CatBoostRegressor(
            iterations=2000,
            depth=7,
            learning_rate=0.02,
            l2_leaf_reg=3.0,
            loss_function='RMSE',
            random_seed=42,
            early_stopping_rounds=150,
            task_type='GPU',
            devices='0',
            verbose=100
        )

        # 直接使用全部数据训练最终模型
        eval_size = min(100000, X_train.shape[0] // 5)  # 验证集大小
        if X_train.shape[0] > eval_size:
            # 分割训练集和验证集
            train_indices = np.random.RandomState(42).choice(
                X_train.shape[0], X_train.shape[0] - eval_size, replace=False)
            eval_indices = np.array(list(set(range(X_train.shape[0])) - set(train_indices)))

            X_train_final = X_train[train_indices]
            y_train_final = y_train[train_indices]
            X_eval_final = X_train[eval_indices]
            y_eval_final = y_train[eval_indices]
        else:
            # 如果数据量不够大，使用全部数据
            X_train_final = X_train
            y_train_final = y_train
            X_eval_final = X_train[:100]  # 使用少量样本作为验证集
            y_eval_final = y_train[:100]

        print(f"第二阶段训练样本数: {X_train_final.shape[0]}, 验证样本数: {X_eval_final.shape[0]}")

        # 训练最终模型
        model_stage2.fit(
            X_train_final, y_train_final,
            eval_set=(X_eval_final, y_eval_final),
            use_best_model=True
        )

        # 获取最终特征重要性
        final_importances = model_stage2.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': final_importances
        }).sort_values('importance', ascending=False)

        # 2. 评估不同特征数量的真实性能
        sorted_features = importance_df['feature'].tolist()

        # 使用适当的样本大小进行特征数量评估
        sample_size = min(200000, X_train.shape[0])  # 增加样本大小到200,000
        if sample_size < X_train.shape[0]:
            indices = np.random.RandomState(42).choice(X_train.shape[0], sample_size, replace=False)
            X_subset = X_train[indices]
            y_subset = y_train[indices]
        else:
            X_subset = X_train
            y_subset = y_train

        print(f"特征数量评估使用样本数: {sample_size}")

        results = {}
        max_features = min(len(features), 20)  # 限制最大特征数

        for n in range(1, max_features + 1):
            log_progress(f"评估{data_type}特征数量: {n}/{max_features}...")
            selected_features = sorted_features[:n]
            feature_indices = [features.index(f) for f in selected_features]
            X_selected = X_subset[:, feature_indices]

            # 使用交叉验证评估性能
            cv_scores = []
            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X_selected):
                X_train_cv, X_val_cv = X_selected[train_idx], X_selected[val_idx]
                y_train_cv, y_val_cv = y_subset[train_idx], y_subset[val_idx]

                eval_model = cb.CatBoostRegressor(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3.0,
                    loss_function='RMSE',
                    random_seed=42,
                    early_stopping_rounds=50,
                    verbose=0
                )

                eval_model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=(X_val_cv, y_val_cv),
                    use_best_model=True
                )

                y_pred = eval_model.predict(X_val_cv)

                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
                mae = mean_absolute_error(y_val_cv, y_pred)
                cv_scores.append({'RMSE': rmse, 'MAE': mae})

            # 计算平均性能
            avg_rmse = np.mean([score['RMSE'] for score in cv_scores])
            avg_mae = np.mean([score['MAE'] for score in cv_scores])
            std_rmse = np.std([score['RMSE'] for score in cv_scores])
            std_mae = np.std([score['MAE'] for score in cv_scores])

            results[n] = {
                'RMSE': avg_rmse,
                'MAE': avg_mae,
                'RMSE_std': std_rmse,
                'MAE_std': std_mae
            }

            print(f"{data_type} 特征数 {n}: RMSE={avg_rmse:.4f}±{std_rmse:.4f}, MAE={avg_mae:.4f}±{std_mae:.4f}")

        return importance_df, results, sorted_features

    except Exception as e:
        print(f"{data_type}科学特征选择时出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认值
        default_importance = pd.DataFrame({'feature': features, 'importance': np.ones(len(features))})
        default_results = {i: {'RMSE': 1.0, 'MAE': 1.0, 'RMSE_std': 0.1, 'MAE_std': 0.1} for i in
                           range(1, len(features) + 1)}
        return default_importance, default_results, features


def find_optimal_features_scientifically(results, method="elbow", data_type=""):
    """使用科学方法确定最优特征数量"""
    try:
        feature_counts = list(results.keys())
        rmse_values = [results[n]['RMSE'] for n in feature_counts]
        mae_values = [results[n]['MAE'] for n in feature_counts]

        if method == "elbow":
            # 肘部法则：寻找改进幅度显著下降的点
            rmse_improvements = []
            mae_improvements = []

            for i in range(1, len(rmse_values)):
                rmse_improvement = (rmse_values[i - 1] - rmse_values[i]) / rmse_values[i - 1]
                mae_improvement = (mae_values[i - 1] - mae_values[i]) / mae_values[i - 1]
                rmse_improvements.append(rmse_improvement)
                mae_improvements.append(mae_improvement)

            # 找到改进幅度下降最显著的点
            combined_improvements = [(r + m) / 2 for r, m in zip(rmse_improvements, mae_improvements)]

            # 寻找拐点：改进幅度突然变小的位置
            improvement_diffs = []
            for i in range(1, len(combined_improvements)):
                diff = combined_improvements[i - 1] - combined_improvements[i]
                improvement_diffs.append(diff)

            # 找到最大的改进幅度下降点
            if improvement_diffs:
                elbow_point = np.argmax(improvement_diffs) + 2  # +2因为索引偏移
                return min(elbow_point, len(feature_counts))

        elif method == "threshold":
            # 阈值法：当性能改进小于某个阈值时停止
            threshold = 0.01  # 1%的改进阈值

            for i in range(1, len(rmse_values)):
                rmse_improvement = (rmse_values[i - 1] - rmse_values[i]) / rmse_values[i - 1]
                mae_improvement = (mae_values[i - 1] - mae_values[i]) / mae_values[i - 1]
                avg_improvement = (rmse_improvement + mae_improvement) / 2

                if avg_improvement < threshold:
                    return feature_counts[i - 1]  # 返回改进开始变小之前的特征数

        elif method == "statistical":
            # 统计显著性法：考虑标准差，确保改进是显著的
            for i in range(1, len(rmse_values)):
                current_rmse = rmse_values[i]
                prev_rmse = rmse_values[i - 1]
                current_std = results[feature_counts[i]]['RMSE_std']
                prev_std = results[feature_counts[i - 1]]['RMSE_std']

                # 检查改进是否超过标准差范围（简单的显著性检验）
                improvement = prev_rmse - current_rmse
                combined_std = np.sqrt(current_std ** 2 + prev_std ** 2)

                if improvement < combined_std:  # 改进不显著
                    return feature_counts[i - 1]

        # 默认返回中等数量的特征
        return len(feature_counts) // 2

    except Exception as e:
        print(f"寻找最优特征数量时出错({data_type}): {e}")
        return len(results) // 2 if results else 1


def catboost_feature_selection(X_train_tigge, y_train, tigge_features, X_train_dem, dem_features):
    """基于CatBoost的科学特征重要性筛选"""
    start_time = time.time()
    log_progress("开始CatBoost科学特征筛选（使用全部训练数据）...")
    print_memory_usage("特征筛选前")

    tigge_results = []
    dem_results = []

    try:
        # TIGGE特征分析
        if tigge_features and X_train_tigge is not None:
            print("=== TIGGE特征科学分析 ===")
            tigge_importance, tigge_metrics, tigge_sorted = scientific_feature_selection(
                X_train_tigge, y_train, tigge_features, "TIGGE"
            )

            # 使用多种方法确定最优特征数
            elbow_n = find_optimal_features_scientifically(tigge_metrics, "elbow", "TIGGE")
            threshold_n = find_optimal_features_scientifically(tigge_metrics, "threshold", "TIGGE")
            statistical_n = find_optimal_features_scientifically(tigge_metrics, "statistical", "TIGGE")

            print(f"\nTIGGE最优特征数量分析:")
            print(f"肘部法则建议: {elbow_n}个特征")
            print(f"阈值法建议: {threshold_n}个特征")
            print(f"统计显著性建议: {statistical_n}个特征")

            # 综合决策（可以取中位数或根据具体情况选择）
            optimal_tigge = int(np.median([elbow_n, threshold_n, statistical_n]))

            tigge_results = tigge_sorted[:optimal_tigge]

            print(f"TIGGE最终选择: {optimal_tigge}个特征")
            print(f"选择的特征: {tigge_results}")

            # 绘制图表
            plot_feature_importance(tigge_importance, 'tigge_importance', 'Feature Importance (TIGGE - 27 Parameters)')
            tigge_counts, tigge_mae, tigge_rmse = plot_feature_count_metrics(
                tigge_metrics, 'tigge_feature_count_metrics.png', 'Feature Selection (TIGGE)'
            )
            plot_combined_tigge_analysis(tigge_importance, tigge_metrics, 'tigge_combined_analysis.png')

            # 保存特征重要性结果
            tigge_importance.to_csv(os.path.join(processed_path, 'tigge_importance.csv'), index=False)

        # DEM特征分析
        if dem_features and X_train_dem is not None:
            print("\n=== DEM特征科学分析 ===")
            dem_importance, dem_metrics, dem_sorted = scientific_feature_selection(
                X_train_dem, y_train, dem_features, "DEM"
            )

            # 使用多种方法确定最优特征数
            elbow_n = find_optimal_features_scientifically(dem_metrics, "elbow", "DEM")
            threshold_n = find_optimal_features_scientifically(dem_metrics, "threshold", "DEM")
            statistical_n = find_optimal_features_scientifically(dem_metrics, "statistical", "DEM")

            print(f"\nDEM最优特征数量分析:")
            print(f"肘部法则建议: {elbow_n}个特征")
            print(f"阈值法建议: {threshold_n}个特征")
            print(f"统计显著性建议: {statistical_n}个特征")

            optimal_dem = int(np.median([elbow_n, threshold_n, statistical_n]))

            dem_results = dem_sorted[:optimal_dem]

            print(f"DEM最终选择: {optimal_dem}个特征")
            print(f"选择的特征: {dem_results}")

            # 绘制图表
            plot_feature_importance(dem_importance, 'dem_importance', 'Feature Importance (DEM - 4 Parameters)')
            dem_counts, dem_mae, dem_rmse = plot_feature_count_metrics(
                dem_metrics, 'dem_feature_count_metrics.png', 'Feature Selection (DEM)'
            )
            plot_combined_dem_analysis(dem_importance, dem_metrics, 'dem_combined_analysis.png')

            # 保存特征重要性结果
            dem_importance.to_csv(os.path.join(processed_path, 'dem_importance.csv'), index=False)

    except Exception as e:
        print(f"特征选择过程中出错: {e}")
        # 使用默认特征选择
        if not tigge_results and tigge_features:
            tigge_results = tigge_features[:8]
        if not dem_results and dem_features:
            dem_results = dem_features[:3]
        print(f"使用默认特征选择: TIGGE={len(tigge_results)}, DEM={len(dem_results)}")

    log_progress("特征筛选完成", start_time)
    return tigge_results, dem_results

# ==================== 数据保存 ====================
def save_datasets(X_train_tigge, X_train_dem, y_train, X_val_tigge, X_val_dem, y_val,
                  X_test_tigge, X_test_dem, y_test, time_features, tigge_features, dem_features,
                  tigge_features_selected, dem_features_selected):
    """保存标准化后的数据集为NetCDF格式，TIGGE和DEM特征分开保存"""
    start_time = time.time()
    log_progress("开始保存数据集...")
    print("数据维度验证:")
    print(f"TIGGE特征: Train {X_train_tigge.shape}, Val {X_val_tigge.shape}, Test {X_test_tigge.shape}")
    print(f"DEM特征: Train {X_train_dem.shape}, Val {X_val_dem.shape}, Test {X_test_dem.shape}")
    print(f"目标变量: Train {y_train.shape}, Val {y_val.shape}, Test {y_test.shape}")
    print(f"时间特征: Train {time_features['train'].shape}, Val {time_features['val'].shape}, "
          f"Test {time_features['test'].shape}")

    # 添加空列表检查
    if not tigge_features_selected:
        log_progress("警告: 没有选择TIGGE特征。将使用所有TIGGE特征。")
        tigge_features_selected = tigge_features.copy()

    if not dem_features_selected:
        log_progress("警告: 没有选择DEM特征。将使用所有DEM特征。")
        dem_features_selected = dem_features.copy()

    # 修改：获取选定特征的索引
    log_progress("获取选定特征的索引...")
    tigge_indices = [tigge_features.index(f) for f in tigge_features_selected]
    dem_indices = [dem_features.index(f) for f in dem_features_selected]

    # 验证索引是否有效
    if tigge_indices and max(tigge_indices) >= X_train_tigge.shape[1]:
        raise ValueError(f"TIGGE特征索引超出范围: {max(tigge_indices)} >= {X_train_tigge.shape[1]}")
    if dem_indices and max(dem_indices) >= X_train_dem.shape[1]:
        raise ValueError(f"DEM特征索引超出范围: {max(dem_indices)} >= {X_train_dem.shape[1]}")

    time_feature_labels = ['year', 'month', 'day', 'hour', 'season']
    for phase in ['train', 'val', 'test']:
        n_samples = y_train.shape[0] if phase == 'train' else y_val.shape[0] if phase == 'val' else y_test.shape[0]
        assert time_features[phase].shape[0] == n_samples, f"时间特征在 {phase} 阶段样本数不匹配"

    # 创建训练集数据集
    log_progress("创建训练集数据集...")
    train_ds = xr.Dataset(
        data_vars={
            "tigge_features": (
                ["sample", "tigge_feature"], X_train_tigge[:, tigge_indices]),
            "dem_features": (
                ["sample", "dem_feature"], X_train_dem[:, dem_indices]),
            "target": (["sample"], y_train),
            "time_features": (["sample", "time_feature"], time_features['train'])
        },
        coords={
            "sample": np.arange(X_train_tigge.shape[0]),
            "tigge_feature": tigge_features_selected,
            "dem_feature": dem_features_selected,
            "time_feature": time_feature_labels
        }
    )

    # 创建验证集数据集
    log_progress("创建验证集数据集...")
    val_ds = xr.Dataset(
        data_vars={
            "tigge_features": (
                ["sample", "tigge_feature"], X_val_tigge[:, tigge_indices]),
            "dem_features": (
                ["sample", "dem_feature"], X_val_dem[:, dem_indices]),
            "target": (["sample"], y_val),
            "time_features": (["sample", "time_feature"], time_features['val'])
        },
        coords={
            "sample": np.arange(X_val_tigge.shape[0]),
            "tigge_feature": tigge_features_selected,
            "dem_feature": dem_features_selected,
            "time_feature": time_feature_labels
        }
    )

    # 创建测试集数据集
    log_progress("创建测试集数据集...")
    test_ds = xr.Dataset(
        data_vars={
            "tigge_features": (
                ["sample", "tigge_feature"], X_test_tigge[:, tigge_indices]),
            "dem_features": (
                ["sample", "dem_feature"], X_test_dem[:, dem_indices]),
            "target": (["sample"], y_test),
            "time_features": (["sample", "time_feature"], time_features['test'])
        },
        coords={
            "sample": np.arange(X_test_tigge.shape[0]),
            "tigge_feature": tigge_features_selected,
            "dem_feature": dem_features_selected,
            "time_feature": time_feature_labels
        }
    )

    # 保存到文件
    try:
        log_progress("保存训练集...")
        train_ds.to_netcdf(os.path.join(processed_path, "train.nc"))

        log_progress("保存验证集...")
        val_ds.to_netcdf(os.path.join(processed_path, "val.nc"))

        log_progress("保存测试集...")
        test_ds.to_netcdf(os.path.join(processed_path, "test.nc"))

        log_progress(f"数据集已保存至: {processed_path}")
    except Exception as e:
        print(f"保存数据集时出错: {e}")
        raise

    log_progress("数据保存完成", start_time)


# ==================== 主流程 ====================
if __name__ == "__main__":
    total_start_time = time.time()
    log_progress(f"开始数据处理流程 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # 步骤1: 创建目录
        log_progress("步骤1/5: 创建目录")
        create_directories()

        # 步骤2: 加载和划分数据
        log_progress("步骤2/5: 加载和划分数据")
        train_data, val_data, test_data, tigge_raw, dem_raw = load_and_split_data()

        # 步骤3: 提前标准化，使用全部训练数据
        log_progress("步骤3/5: 数据标准化")
        (X_train_tigge, X_val_tigge, X_test_tigge,
         X_train_dem, X_val_dem, X_test_dem,
         y_train, y_val, y_test,
         time_features, tigge_features_raw, dem_features_raw) = standardize_data_before_selection(
            train_data, val_data, test_data, tigge_raw, dem_raw
        )

        # 步骤4: 使用全部标准化训练数据进行特征筛选
        log_progress("步骤4/5: 特征筛选")
        tigge_features_selected, dem_features_selected = catboost_feature_selection(
            X_train_tigge, y_train, tigge_features_raw,
            X_train_dem, dem_features_raw
        )
        # 步骤5: 保存筛选后的数据集
        log_progress("步骤5/5: 保存数据集")
        save_datasets(X_train_tigge, X_train_dem, y_train,
                      X_val_tigge, X_val_dem, y_val,
                      X_test_tigge, X_test_dem, y_test,
                      time_features, tigge_features_raw, dem_features_raw,
                      tigge_features_selected, dem_features_selected)

        log_progress("\n数据处理全流程完成！最终数据集结构:")
        check_ds = xr.open_dataset(os.path.join(processed_path, "train.nc"))
        print(check_ds)

        # 打印选择的特征
        print("\n选择的TIGGE特征:")
        for i, feature in enumerate(tigge_features_selected):
            print(f"{i + 1}. {feature}")

        print("\n选择的DEM特征:")
        for i, feature in enumerate(dem_features_selected):
            print(f"{i + 1}. {feature}")

        print_memory_usage("处理完成")

        total_time = time.time() - total_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        log_progress(f"总处理时间: {hours}小时 {minutes}分钟 {seconds}秒")

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback

        traceback.print_exc()
