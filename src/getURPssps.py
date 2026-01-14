# -*- coding: utf-8 -*-
 

"""

Created on Mon Jun  2 13:32:43 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

import rasterio 
import joblib
import pickle
import numpy as np
import h5py
from scipy.interpolate import griddata, RectBivariateSpline
import netCDF4 as nc
from PIL import Image



with open('../data/Mask.pkl', 'rb') as f:  # 读取pickle文件
    Maskdata = joblib.load(f)
    f.close()
Mask = Maskdata['mask']

lat1 = Maskdata['lat']
lon1 = Maskdata['lon']

data = nc.Dataset('H:/BR/Data/Asia/scdhi/asia_scdhi_1981_2020.nc')
lonAS = data['lon']
latAS = data['lat']
scdhi = data['scdhi']
m1 = scdhi[200,:,:].copy()
m1[~np.isnan(m1)] = 1
[LONA,LATA] = np.meshgrid(lonAS,latAS)

# lat = mat_data['BR_grid'][0]
# lon = mat_data['BR_grid'][1]
points  = np.array([LONA[~np.isnan(scdhi[200,:,:])].data,LATA[~np.isnan(scdhi[200,:,:])].data]).T


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
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    grid_values = Z.ravel()
    
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
    
def calculate_grid_area(lat: float, delta: float, use_ellipsoid=True) :
    """
     
    :param lat: 格网中心纬度（或南边界纬度，误差可忽略），单位：度
    :param use_ellipsoid: 是否使用WGS84椭球（False则用球体简化）
    :return: 格网面积（平方米）
    """
    
    # WGS84椭球参数
    WGS84_A = 6378137.0  # 长半轴 (m)
    WGS84_E2 = 0.00669437999014  # 第一偏心率平方 e² = (a²-b²)/a²
    DEG2RAD = np.pi / 180.0    # 角度转弧度
    
    # 1. 转换为弧度
    lat_rad = lat * DEG2RAD
    delta_lon_rad = delta * DEG2RAD  # 经度跨度 °
    delta_lat_rad = delta * DEG2RAD  # 纬度跨度 °

    if use_ellipsoid:
        # 2. 椭球模型（WGS84）计算
        sin_lat =np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        
        # 卯酉圈曲率半径 N(φ)
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat **2)
        # 子午圈曲率半径 M(φ)
        m = WGS84_A * (1 - WGS84_E2) / ((1 - WGS84_E2 * sin_lat** 2) ** 1.5)
        
        # 格网面积 = M×Δlat × N×cos(φ)×Δlon
        area = m * delta_lat_rad * (n * cos_lat) * delta_lon_rad
    else:
        # 3. 球体简化模型（地球半径取WGS84平均半径）
        r = 6371008.7714  # 地球平均半径 (m)
        # 球体格网面积公式：r² × |sin(lat1)-sin(lat2)| × Δlon
        lat1 = lat - delta/2  # 格网南边界纬度
        lat2 = lat + delta/2  # 格网北边界纬度
        area = (r **2) * abs(np.sin(lat2*DEG2RAD) - np.sin(lat1*DEG2RAD)) * delta_lon_rad
    
    return abs(area/1e6)


def get05data(pop,X,Y):
    Z = pop/A0125
    Z[Z<0] = 0
    
    #interp_data =  interpolate_with_spline(Maskdata['lon'][:], Maskdata['lat'][:], Z, points )
    interp_data =  interpolate_with_griddata(X,Y,Z, points )
    ASp  = sq2grid(interp_data, LATA, LONA, points) 
    PopAS = ASp*A05*m1
    PopAS[PopAS>3e10] = 0
    return PopAS
        
D = {}
SN = ["SSP1","SSP3","SSP5"]
for s in SN:
    ASPR = []
    ASPU = []
    ASPT = []
    for y in range(2020,2101,10):
        
        dsr = nc.Dataset('../data/popf2/'+s+'/Rural/NetCDF/'+str.lower(s)+'rur'+str(y)+'.nc')
        dsu = nc.Dataset('../data/popf2/'+s+'/Urban/NetCDF/'+str.lower(s)+'urb'+str(y)+'.nc')
        dst = nc.Dataset('../data/popf2/'+s+'/Total/NetCDF/'+str.lower(s)+'_'+str(y)+'.nc')
        lon1 = dsr['lon'][:]
        lon1[lon1<0] = lon1[lon1<0]+360
        lat1 = dsr['lat'][:]
        [LON1,LAT1] = np.meshgrid(lon1,lat1)
        A0125 = calculate_grid_area(LAT1, 0.125)
        A05 = calculate_grid_area(LATA, 0.5)
            
        pop_r = dsr[str.lower(s)+'rur'+str(y)]
        pop_u = dsu[str.lower(s)+'urb'+str(y)]
        pop_t = dst[str.lower(s)+'_'+str(y)]
        
        
        X = LON1
        Y = LAT1
        aspr = get05data(pop_r,X,Y)    
        aspu = get05data(pop_u,X,Y)  
        aspt = get05data(pop_t,X,Y) 
        ASPR.append(aspr)
        ASPU.append(aspu)
        ASPT.append(aspt)
    
        print(s+' '+str(y))
    
    D[s] = {'Total':np.array(ASPT)*m1,'Rural':np.array(ASPR)*m1,'Urban':np.array(ASPU)*m1}
    
    
with open('../data/ASPOP_URssps.pkl', 'wb') as f:
  	pickle.dump(D, f)# -*- coding: utf-8 -*-
