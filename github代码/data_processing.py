
# ==================== 环境准备 ====================
import os  # 系统模块
import glob  # 文件操作模块
import numpy as np  # 数值计算模块
import xarray as xr  # 数据处理模块
import pandas as pd  # 数据处理模块
from datetime import datetime  # 日期时间模块
import matplotlib.pyplot as plt  # 绘图模块

# ==================== 路径设置 ====================
raw_data_path = {
    "ERA5": "../processed_data/ERA5/",  # 修改：ERA5改为JRA55
    "TIGGE": "../processed_data/TIGGE/",
    "DEM": "../processed_data/DEM/",
}
processed_path = "./final_processed/"


# ==================== 辅助函数 ====================
def create_directories():
    """创建必要的目录结构"""
    for path in ["ERA5", "TIGGE", "DEM"]:  # 修改：ERA5改为JRA55
        os.makedirs(os.path.join(processed_path, path), exist_ok=True)


def validate_data(dataset, name):
    """验证数据集的有效性"""
    print(f"\n验证{name}数据集:")
    print(f"维度信息: {dataset.dims}")
    print(f"变量列表: {list(dataset.data_vars)}")

    # 仅对有时间维度的数据集显示时间范围
    if 'time' in dataset.dims:
        print(f"时间范围: {dataset.time.min().values} 到 {dataset.time.max().values}")

    # 检查缺失值
    for var in dataset.data_vars:
        missing = dataset[var].isnull().sum().values
        total = dataset[var].size
        print(f"{var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")


# ==================== 数据读取 ====================
def load_data():
    """加载预处理后的数据"""
    try:
        # 加载JRA55数据
        era5_path = os.path.join(raw_data_path["ERA5"], "ERA5_processed_2021_2024.nc")  # 修改：ERA5改为JRA55，时间范围改为2020-2023
        era5_data = xr.open_dataset(era5_path)  # 修改：ERA5改为JRA55
        validate_data(era5_data, "ERA5")  # 修改：ERA5改为JRA55

        # 加载TIGGE数据
        tigge_path = os.path.join(raw_data_path["TIGGE"], "TIGGE_processed_2021_2024.nc")  # 修改：时间范围改为2020-2023
        tigge_data = xr.open_dataset(tigge_path)
        validate_data(tigge_data, "TIGGE")

        # 加载DEM数据
        dem_path = os.path.join(raw_data_path["DEM"], "DEM_processed.nc")
        dem_data = xr.open_dataset(dem_path)
        validate_data(dem_data, "DEM")

        return era5_data, tigge_data, dem_data

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


# ==================== 数据处理 ====================
def process_data(era5_data, tigge_data, dem_data):
    try:
        print("\n开始数据处理...")

        # 1. 时间对齐
        print("执行时间对齐...")
        common_times = list(set(era5_data.time.values) & set(tigge_data.time.values))  # 修改：ERA5改为JRA55
        common_times.sort()

        era5_aligned = era5_data.sel(time=common_times)  # 修改：ERA5改为JRA55
        tigge_aligned = tigge_data.sel(time=common_times)

        # 2. 空间对齐（使用统一的经纬度网格）
        print("执行空间对齐...")
        target_lat = np.arange(35.13, 47, 0.25)
        target_lon = np.arange(103, 126.88, 0.25)



        for ds_name, ds in [("ERA5", era5_aligned), ("TIGGE", tigge_aligned), ("DEM", dem_data)]:
            print(f"\n{ds_name} 原始网格:")
            if 'lat' in ds.coords:
                print(f"纬度范围: {ds.lat.min().values:.2f} - {ds.lat.max().values:.2f}")
            if 'lon' in ds.coords:
                print(f"经度范围: {ds.lon.min().values:.2f} - {ds.lon.max().values:.2f}")

        # 统一坐标名称
        if 'latitude' in era5_aligned.dims and 'longitude' in era5_aligned.dims:
            era5_aligned = era5_aligned.rename({'latitude': 'lat', 'longitude': 'lon'})

        # 将所有数据集插值到统一网格
        era5_aligned = era5_aligned.interp(lat=target_lat, lon=target_lon)
        tigge_aligned = tigge_aligned.interp(lat=target_lat, lon=target_lon)
        dem_aligned = dem_data.interp(lat=target_lat, lon=target_lon)

        # 3. 异常值处理（使用3σ法则）
        print("处理异常值...")
        for var in era5_aligned.data_vars:
            mean = era5_aligned[var].mean()
            std = era5_aligned[var].std()
            era5_aligned[var] = xr.where(
                abs(era5_aligned[var] - mean) > 3 * std,
                np.nan,
                era5_aligned[var]
            )

        for var in tigge_aligned.data_vars:
            mean = tigge_aligned[var].mean()
            std = tigge_aligned[var].std()
            tigge_aligned[var] = xr.where(
                abs(tigge_aligned[var] - mean) > 3 * std,
                np.nan,
                tigge_aligned[var]
            )

        # DEM数据异常值处理
        for var in dem_aligned.data_vars:
            mean = dem_aligned[var].mean()
            std = dem_aligned[var].std()
            dem_aligned[var] = xr.where(
                abs(dem_aligned[var] - mean) > 3 * std,
                np.nan,
                dem_aligned[var]
            )

        # 4. 增强版缺失值填充
        print("填充缺失值（增强版）...")

        # 4.1 首先检查每个数据集的缺失值情况
        print("\n缺失值统计（处理前）:")

        print("\nERA5数据集缺失值统计:")
        for var in era5_aligned.data_vars:
            missing = era5_aligned[var].isnull().sum().values
            total = era5_aligned[var].size
            missing_pct = (missing / total) * 100
            print(f"ERA5变量 {var}: {missing}/{total} 缺失值 ({missing_pct:.2f}%)")

        # TIGGE缺失值统计
        print("\nTIGGE数据集缺失值统计:")
        for var in tigge_aligned.data_vars:
            missing = tigge_aligned[var].isnull().sum().values
            total = tigge_aligned[var].size
            missing_pct = (missing / total) * 100
            print(f"TIGGE变量 {var}: {missing}/{total} 缺失值 ({missing_pct:.2f}%)")

        # DEM缺失值统计
        print("\nDEM数据集缺失值统计:")
        for var in dem_aligned.data_vars:
            missing = dem_aligned[var].isnull().sum().values
            total = dem_aligned[var].size
            missing_pct = (missing / total) * 100
            print(f"DEM变量 {var}: {missing}/{total} 缺失值 ({missing_pct:.2f}%)")

        # 4.2 时间维度插值
        print("\n执行时间维度插值...")
        era5_aligned = era5_aligned.interpolate_na(dim='time', method='linear')
        tigge_aligned = tigge_aligned.interpolate_na(dim='time', method='linear')

        # 4.3 空间维度插值
        print("执行空间维度插值...")
        for ds_name, ds in [("ERA5", era5_aligned), ("TIGGE", tigge_aligned), ("DEM", dem_aligned)]:  # 修改：ERA5改为JRA55
            print(f"\n处理{ds_name}数据集空间插值...")
            for var in ds.data_vars:
                if 'lat' in ds[var].dims and 'lon' in ds[var].dims:
                    # 先使用线性插值
                    missing_before = ds[var].isnull().sum().values
                    ds[var] = ds[var].interpolate_na(dim='lat', method='linear')
                    ds[var] = ds[var].interpolate_na(dim='lon', method='linear')
                    missing_after_linear = ds[var].isnull().sum().values

                    # 对于仍然存在的缺失值，使用最近邻插值
                    if ds[var].isnull().any():
                        ds[var] = ds[var].interpolate_na(dim='lat', method='nearest')
                        ds[var] = ds[var].interpolate_na(dim='lon', method='nearest')
                        missing_after_nearest = ds[var].isnull().sum().values

                        print(
                            f"  变量 {var}: 原始缺失值 {missing_before} -> 线性插值后 {missing_after_linear} -> 最近邻插值后 {missing_after_nearest}")
                    else:
                        print(
                            f"  变量 {var}: 原始缺失值 {missing_before} -> 线性插值后 {missing_after_linear} (已完全填充)")

        # 4.4 对于仍然存在的缺失值，使用变量平均值填充
        print("\n使用平均值填充剩余缺失值...")
        for ds_name, ds in [("ERA5", era5_aligned), ("TIGGE", tigge_aligned), ("DEM", dem_aligned)]:  # 修改：ERA5改为JRA55
            print(f"\n处理{ds_name}数据集剩余缺失值...")
            for var in ds.data_vars:
                if ds[var].isnull().any():
                    missing_before = ds[var].isnull().sum().values
                    # 计算变量的全局平均值（忽略NaN）
                    var_mean = float(ds[var].mean(skipna=True).values)
                    # 使用平均值填充
                    ds[var] = ds[var].fillna(var_mean)
                    missing_after = ds[var].isnull().sum().values
                    print(
                        f"  变量 {var}: 使用平均值 {var_mean:.4f} 填充 {missing_before} 个缺失值 -> 剩余缺失值 {missing_after}")
                else:
                    print(f"  变量 {var}: 无需填充 (无缺失值)")

        # 4.5 检查填充后的缺失值情况
        print("\n缺失值统计（处理后）:")


        print("\nERA5数据集缺失值统计:")
        for var in era5_aligned.data_vars:
            missing = era5_aligned[var].isnull().sum().values
            total = era5_aligned[var].size
            missing_pct = (missing / total) * 100 if total > 0 else 0
            print(f"ERA5变量 {var}: {missing}/{total} 缺失值 ({missing_pct:.2f}%)")

        # TIGGE缺失值统计
        print("\nTIGGE数据集缺失值统计:")
        for var in tigge_aligned.data_vars:
            missing = tigge_aligned[var].isnull().sum().values
            total = tigge_aligned[var].size
            missing_pct = (missing / total) * 100 if total > 0 else 0
            print(f"TIGGE变量 {var}: {missing}/{total} 缺失值 ({missing_pct:.2f}%)")

        # DEM缺失值统计
        print("\nDEM数据集缺失值统计:")
        for var in dem_aligned.data_vars:
            missing = dem_aligned[var].isnull().sum().values
            total = dem_aligned[var].size
            missing_pct = (missing / total) * 100 if total > 0 else 0
            print(f"DEM变量 {var}: {missing}/{total} 缺失值 ({missing_pct:.2f}%)")

        # 5. 特征工程
        print("\n执行特征工程...")
        # 添加时间特征
        time_features = pd.DataFrame({
            'hour': pd.to_datetime(era5_aligned.time.values).hour,
            'day': pd.to_datetime(era5_aligned.time.values).day,
            'month': pd.to_datetime(era5_aligned.time.values).month,
            'season': pd.to_datetime(era5_aligned.time.values).month % 12 // 3 + 1
        })

        # 将时间特征添加到数据集
        for col in time_features.columns:
            era5_aligned[col] = ('time', time_features[col].values)
            tigge_aligned[col] = ('time', time_features[col].values)

        return era5_aligned, tigge_aligned, dem_aligned

    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        raise


# ==================== 数据输出 ====================
def save_data(era5_data, tigge_data, dem_data):
    """保存处理后的数据"""
    try:
        print("\n保存处理后的数据...")

        era5_output_path = os.path.join(processed_path, "ERA5", "ERA5_final_2021_2024.nc")
        era5_data.to_netcdf(era5_output_path)
        print(f"ERA5数据已保存至: {era5_output_path}")

        # 保存TIGGE数据
        tigge_output_path = os.path.join(processed_path, "TIGGE", "TIGGE_final_2021_2024.nc")
        tigge_data.to_netcdf(tigge_output_path)
        print(f"TIGGE数据已保存至: {tigge_output_path}")

        # 保存DEM数据
        dem_output_path = os.path.join(processed_path, "DEM", "DEM_final.nc")
        dem_data.to_netcdf(dem_output_path)
        print(f"DEM数据已保存至: {dem_output_path}")

    except Exception as e:
        print(f"数据保存失败: {str(e)}")
        raise


# ==================== 主函数调用 ====================
if __name__ == "__main__":
    try:
        # 创建目录
        create_directories()

        # 加载数据
        print("开始数据处理流程...")
        era5_data, tigge_data, dem_data = load_data()

        # 处理数据
        era5_processed, tigge_processed, dem_processed = process_data(
            era5_data, tigge_data, dem_data
        )

        # 保存结果
        save_data(era5_processed, tigge_processed, dem_processed)

        print("\n数据处理完成！")

    except Exception as e:
        print(f"程序执行失败: {str(e)}")