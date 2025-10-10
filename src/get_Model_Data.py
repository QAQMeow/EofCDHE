# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:54:10 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 
import os
import numpy as np
import joblib
import pandas as pd
import netCDF4 as nc
import threading 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

 

dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')
with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']

 
 
MODEL = ['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0']


Period1 =['historical'] 
Period2 =['historical']
tp = ['1981_1990','1991_2000','2001_2010','2011_2014']
d1 = 'H:/data'

Th = np.zeros_like(mask)
sha = mask.shape
SCDHI1s = {}
SAPEI1s = {}
STI1s = {}
for period1 in Period1:
    
    for period2 in Period2:
        SCDHI = []
        SAPEI = []
        STI = []
        for timep in tp:
                           
               
            dscdhi = 'MME/scdhi/'+period1
            dsapei = 'MME/sapei/'+period1
            dsti = 'MME/sti/'+period1
            scdhi_dir = d1+'/'+dscdhi+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc'
            sapei_dir = d1+'/'+dsapei+'/'+period2+'_sapei_asia_daily_'+timep+'.nc'
            sti_dir = d1+'/'+dsti+'/'+period2+'_sti_asia_daily_'+timep+'.nc'
            
            scdhidataset = nc.Dataset(scdhi_dir)
            sapeidataset = nc.Dataset(sapei_dir)
            stidataset = nc.Dataset(sti_dir)
            
            lat = scdhidataset['lat']
            lon = scdhidataset['lon']
            [LONB,LATB] = np.meshgrid(lon,lat)
            scdhi = scdhidataset['scdhi'][:]
            sapei = sapeidataset['sapei'][:]
            sti = stidataset['sti'][:]
            dates = pd.date_range(start= timep[:4]+'-01-01', end=timep[5:]+'-12-31')
            
            for y in range(int(timep[:4]),int(timep[5:])+1):
                
                yscdhi = scdhi[dates.year==y,:,:].copy()
                ysapei = sapei[dates.year==y,:,:].copy()
                ysti = sti[dates.year==y,:,:].copy()
  
                SCDHI.append(np.nansum(yscdhi,axis=0))
                SAPEI.append(np.nansum(ysapei,axis=0))
                STI.append(np.nanmean(ysti,axis=0))
                print(y)
        SCDHI1s[period2] = np.array(SCDHI)
        SAPEI1s[period2] = np.array(SAPEI)
        STI1s[period2] = np.array(STI)  
Period1 =['SSPs'] 
Period2 =['ssp126','ssp370','ssp585']
tp = ['2015_2020','2021_2030','2031_2040','2041_2050','2051_2060','2061_2070','2071_2080','2081_2090','2091_2100']
d1 = 'H:/data'

Th = np.zeros_like(mask)
sha = mask.shape

SCDHI2s = {}
SAPEI2s = {}
STI2s = {}
for period1 in Period1:
    
    for period2 in Period2:
        SCDHI = []
        SAPEI = []
        STI = []
        
        for timep in tp:
                           
               
            dscdhi = 'MME/scdhi/'+period1
            dsapei = 'MME/sapei/'+period1
            dsti = 'MME/sti/'+period1
            scdhi_dir = d1+'/'+dscdhi+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc'
            sapei_dir = d1+'/'+dsapei+'/'+period2+'_sapei_asia_daily_'+timep+'.nc'
            sti_dir = d1+'/'+dsti+'/'+period2+'_sti_asia_daily_'+timep+'.nc'
            
            scdhidataset = nc.Dataset(scdhi_dir)
            sapeidataset = nc.Dataset(sapei_dir)
            stidataset = nc.Dataset(sti_dir)
            
            lat = scdhidataset['lat']
            lon = scdhidataset['lon']
            [LONB,LATB] = np.meshgrid(lon,lat)
            scdhi = scdhidataset['scdhi'][:]
            sapei = sapeidataset['sapei'][:]
            sti = stidataset['sti'][:]
            dates = pd.date_range(start= timep[:4]+'-01-01', end=timep[5:]+'-12-31')
            
            for y in range(int(timep[:4]),int(timep[5:])+1):
                
                yscdhi = scdhi[dates.year==y,:,:].copy()
                ysapei = sapei[dates.year==y,:,:].copy()
                ysti = sti[dates.year==y,:,:].copy()
  
                SCDHI.append(np.nansum(yscdhi,axis=0))
                SAPEI.append(np.nansum(ysapei,axis=0))
                STI.append(np.nanmean(ysti,axis=0))
                print(period2+' '+str(y))
                
        SCDHI2s[period2] = np.array(SCDHI)
        SAPEI2s[period2] = np.array(SAPEI)
        STI2s[period2] = np.array(STI) 
 

Model_Data  = {}
Model_Data['historical_sapei'] = SAPEI1s
Model_Data['historical_sti']  = STI1s
Model_Data['historical_scdhi'] = SCDHI1s   
  
Model_Data['future_sapei'] = SAPEI2s
Model_Data['future_sti']  = STI2s
Model_Data['future_scdhi'] = SCDHI2s   
 
#CDHEs_data = pd.DataFrame(CDHEs_data)
import pickle
with open('../data/Model_Data.pkl', 'wb') as f:
	pickle.dump(Model_Data, f)            



