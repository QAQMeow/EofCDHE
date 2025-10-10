# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 19:00:58 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
import joblib
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from rasterio import features
from rasterio.transform import from_origin
import netCDF4 as nc
def getASCMask():
    with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
        Maskdata = joblib.load(f)
        f.close()
    Mask = Maskdata['Mask']
    
    lat = Maskdata['lat']
    lon = Maskdata['lon']
    LON,LAT = np.meshgrid(lon,lat) 
    # 1. 读取 Shapefile
    shapefile_path = 'H:\\65国家\\Asia\\AsiaSub.shp'
    #shapefile_path ='H:\\65国家\\一带一路\\BRArea.shp'
    gdf = gpd.read_file(shapefile_path)
    
    # 2. 定义栅格参数
    # 假设已有一个二维矩阵需要赋值
    rows, cols = len(lat), len(lon)  # 矩阵大小
    
    # 定义栅格范围（与Shapefile空间范围匹配）
    xmin, ymin, xmax, ymax = np.nanmin(lon),np.nanmin(lat),np.nanmax(lon),np.nanmax(lat)
    
    # 计算分辨率
    x_res =  0.5
    y_res = 0.5
    
    # 创建栅格变换（左上角坐标和像元大小）
    transform = from_origin(xmin, ymax, x_res, y_res)
    
    # 3. 创建空矩阵（初始值为0或np.nan）
    output_matrix = np.nan*np.zeros((rows, cols), dtype=np.float32)
    
    # 4. 按分区赋值
    for idx, row in gdf.iterrows():
        # 获取多边形几何和属性值（假设属性名为 'value'）
        geom = row['geometry']
        value = row['OBJECTID_1']  # 替换为实际的属性列名
        
        # 生成多边形对应的布尔掩码（True表示多边形覆盖的区域）
        mask = features.geometry_mask(
            [geom],
            out_shape=(rows, cols),
            transform=transform,
            all_touched=False,  # 是否包含所有与多边形接触的像素
            invert=True  # True表示多边形区域为True
        )
        
        # 将掩码对应的矩阵位置赋值
        output_matrix[mask] = value
    return output_matrix
# 5. 可视化结果
# plt.figure(figsize=(10, 8))
# plt.imshow(getBRCMask(), cmap='viridis', )
# plt.colorbar(label='属性值')
# #gdf.boundary.plot(ax=plt.gca(), color='red', linewidth=1)
# plt.title('按Shapefile分区赋值的二维矩阵')
# plt.show()
