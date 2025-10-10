# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 12:11:12 2025

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
import pandas as pd


data = nc.Dataset('H:/BR/Data/BR/scdhi/br_scdhi.nc')
lonBR = data['lon']
latBR = data['lat']
scdhi = data['scdhi']
[LONB,LATB] = np.meshgrid(lonBR,latBR)
time = data.variables['time']
# 生成日期序列 
dates = pd.date_range(start='1979-01-01', end='2023-12-31')
def calculate_grid_area(lat, dlat=0.5, dlon=0.5):
    """
    计算给定纬度处经纬度网格的面积
    
    参数:
    lat : float 或 array_like - 纬度（单位：度）
    dlat : float - 纬度差（单位：度）
    dlon : float - 经度差（单位：度）

    返回:
    float 或 array_like - 网格面积（单位：平方米）
    """
    a = 6378137.0  # 长半轴（赤道半径，单位：米）
    f = 1/298.257223563  # 扁率
    b = a * (1 - f)  # 短半轴（极半径，单位：米）
    # 将角度转换为弧度
    lat_rad = np.radians(lat)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)

    # 计算纬度圈半径
    r_lat = a * np.cos(lat_rad) / np.sqrt(1 - (f * (2 - f)) * (np.sin(lat_rad) ** 2))

    # 计算子午线弧长
    M = a * (1 - f) ** 2 / (1 - (f * (2 - f)) * (np.sin(lat_rad) ** 2)) ** 1.5

    # 计算面积
    area = r_lat * M * dlat_rad * dlon_rad
    return area/1e6

Ar = calculate_grid_area(LATB)


def getEvents(data,th):
    '''
    The event consists of a series of minor events occurring on consecutive days

    Parameters
    ----------
    data : list or numpy.array,int,,Nx1, a sequence only contains 0 and 1,
           0 means that the minor event did not occur, 1 is the opposite
         
    th : int
         Minimum number of consecutive days to be recognized as an event
         

    Returns
    -------
    x : a sequence that only contains 0 and 1,and only contains events which consecutive days >= th
        0 means that the minor event did not occur, 1 is the opposite
        
    f : frequence of events(consecutive days >= th) in data

    '''
    x = np.zeros_like(data)
    d1 = np.append(data,0)
    d1 = np.insert(d1,0,0)
    b = d1[:-1]-d1[1:]

    s = np.where(b==-1)[0]
    t = np.where(b==1)[0]
    c = (t-s)
    e = np.where(c>=th)[0]
    for i in range(len(e)):
        n = e[i]
        x[s[n]:t[n]] = 1
    f = len(e)
    return x,f







M1 = scdhi[:,:,200].copy()
M1[M1==M1] = 1
CDHDs = []
CDHAs = []
CDHIs = []
CDHFs = []
for y in range(1990,2021):
    
    yd = scdhi[:,:,dates.year==y].copy()
    ei = yd.copy()
    ef = np.zeros([210,380])*np.nan
    yd[yd>=-2] = 0
    yd[yd<-2] = 1
    cdh_days = yd
    for i in range(210):
        for j in range(380):
            if M1[i,j]==1:
                cdh_days[i,j,:],ef[i,j] = getEvents(yd[i,j,:], 3)

    ei = np.abs(ei*cdh_days)
    ei[ei==0] = np.nan
    Ycdh_days = np.nansum(cdh_days,axis=2)
     
    Ycdh_i = np.nansum(ei,axis=2)
    
    cdh_area = Ycdh_days.copy()
    cdh_area[cdh_area>0] = 1
    cdh_area[cdh_area!=1] = 0
    cdh_area = cdh_area*Ar
    CDHDs.append(Ycdh_days*M1)
    CDHAs.append(cdh_area*M1)
    CDHIs.append(Ycdh_i*M1)
    CDHFs.append(ef*M1)
    print(y)


CDHEs_data  = {}
CDHEs_data['CDHE_frequency'] = np.array(CDHFs)
CDHEs_data['CDHE_days']  = np.array(CDHDs)
CDHEs_data['CDHE_area'] = np.array(CDHAs)   
CDHEs_data['CDHE_intensity'] = np.array(CDHIs)  
#CDHEs_data = pd.DataFrame(CDHEs_data)
with open('../data/CDHEs_data.pkl', 'wb') as f:
	pickle.dump(CDHEs_data, f)