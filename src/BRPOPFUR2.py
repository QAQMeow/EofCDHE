# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:17:16 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 

import rasterio 
import joblib
import pickle
import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import griddata, RectBivariateSpline
import netCDF4 as nc
from PIL import Image


with open('../data/Mask.pkl', 'rb') as f:  # 读取pickle文件
    Maskdata = joblib.load(f)
    f.close()
Mask = Maskdata['mask']

lat1 = Maskdata['lat']
lon1 = Maskdata['lon']

data = nc.Dataset('H:/BR/Data/BR/scdhi/br_scdhi.nc')
lonBR = data['lon']
latBR = data['lat']
scdhi = data['scdhi']
[LONB,LATB] = np.meshgrid(lonBR,latBR)

# lat = mat_data['BR_grid'][0]
# lon = mat_data['BR_grid'][1]
points  = np.array([LONB[~np.isnan(scdhi[:,:,200])].data,LATB[~np.isnan(scdhi[:,:,200])].data]).T


def fliplrMap(data):
    '''
    change  central meridian 0 to 180 

    Parameters                       
    -------      
    data : numpy.array,MxN          
           Array with central meridian 0    
    
    Returns
    -------
    d : numpy.array,MxN
        Array with central meridian 180
     
    '''
    d = data.copy()
    S  = np.shape(d)
    m = int(S[1]/2)
    pp = data[:,m:].copy()
    pp2 = data[:,:m].copy()
    d[:,:m] = pp
    d[:,m:] = pp2
    return d



# 方法1：使用 griddata 进行插值（适用于不规则网格）
def interpolate_with_griddata(X, Y, Z, points):
    # 将网格数据转换为点列表
    grid_points = np.vstack([X , Y ]).T
    grid_values = Z 
    # 使用 griddata 进行插值
    # method 可以是 'linear', 'nearest', 'cubic'
    interpolated_values = griddata(grid_points, grid_values, points, method='nearest')
    
    return interpolated_values

# 方法2：使用 RectBivariateSpline 进行插值（适用于规则网格，效率更高）
def interpolate_with_spline(x, y, Z, points):
    # 创建插值函数
    spline = RectBivariateSpline(x, y, Z.T)  # 注意这里需要转置Z
    
    # 提取要插值的点的坐标
    xi = points[:, 0]
    yi = points[:, 1]
    
    # 进行插值
    interpolated_values = spline.ev(xi, yi)
    
    return interpolated_values



def sq2grid(data,LAT,LON,points):
    
    gridd =np.zeros_like(LAT)*np.nan
    for i in range(len(points)):
        y = np.where((LAT==points[i,1])&(LON==points[i,0]))[0][0]
        x = np.where((LAT==points[i,1])&(LON==points[i,0]))[1][0]
        gridd[y,x] = data[i]
    return gridd
    
def calculate_grid_area(lat: float, lon: float, delta: float) -> float:
    """
    计算特定经纬度处指定大小的格网面积（考虑地球椭球形状）
    
    参数:
    lat (float): 中心点纬度（度）
    lon (float): 中心点经度（度）
 
    返回:
    float: 网格面积（平方米）
    """
    # 创建WGS84椭球体的Geod对象
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    
    # 计算网格四个顶点的经纬度
    lat_min = lat - delta/2
    lat_max = lat + delta/2
    lon_min = lon - delta/2
    lon_max = lon + delta/2
    
    # 定义网格四个顶点的坐标（顺时针或逆时针顺序）
    lons = [lon_min, lon_min, lon_max, lon_max, lon_min]  # 闭合多边形
    lats = [lat_min, lat_max, lat_max, lat_min, lat_min]  # 闭合多边形
    
    # 计算多边形面积（单位：平方米）
    area, _ = geod.polygon_area_perimeter(lons, lats)
    
    return abs(area)

A1 = []
A05 = np.zeros([210,380])
popdata1 = pd.read_excel('../data/futurepop2/pop_B&R_ssp1_v1.xlsx')
lat2 = popdata1['lat'].values
lon2 = popdata1['lon'].values
lon2[lon2<0] = 360+lon2[lon2<0]
for i in range(len(lat2)):
 
         A1.append(calculate_grid_area(lat2[i], lon2[i], 0.5))
A1 = np.array(A1)
for i in range(210):
    for j in range(380):
        A05[i,j] = calculate_grid_area(latBR[i], lonBR[j], 0.5)


POP = {}

for s in ['ssp1','ssp2','ssp3','ssp5']:
    fr ='../data/futurepop2/pop_B&R_'+s+'_v1.xlsx'
    data = pd.read_excel(fr)
    PopBR = {}
    for i in range(2020,2101,1):
         
        
        
        X = lon2
        Y = lat2
        Z = data[i].values/A1
        #interp_data =  interpolate_with_spline(Maskdata['lon'][:], Maskdata['lat'][:], Z, points )
        interp_data =  interpolate_with_griddata(X,Y, Z, points )
        Brp  = sq2grid(interp_data, LATB, LONB, points) 
        PopBR[str(i)] = Brp*A05
        print(s+' '+str(i))
    POP[s] = PopBR
        
    
    
with open('../data/Future_BRPOP2.pkl', 'wb') as f:
 	pickle.dump(POP, f)