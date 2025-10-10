# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:12:36 2025

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

for period1 in Period1:
    
    for period2 in Period2:
        CDHDs = []
        CDHIs = []
        CDHFs = []
        CDHAs = []
        for timep in tp:
                           
               
            dscdhi = 'MME/scdhi/'+period1
             
            scdhi_dir = d1+'/'+dscdhi+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc'
          
            scdhidataset = nc.Dataset(scdhi_dir)
            lat = scdhidataset['lat']
            lon = scdhidataset['lon']
            [LONB,LATB] = np.meshgrid(lon,lat)
            scdhi = scdhidataset['scdhi'][:]
            dates = pd.date_range(start= timep[:4]+'-01-01', end=timep[5:]+'-12-31')
            
            Ar = calculate_grid_area(LATB)
            th = -1.3
            for y in range(int(timep[:4]),int(timep[5:])+1):
                
                yd = scdhi[dates.year==y,:,:].copy()
                ei = scdhi[dates.year==y,:,:].copy()
                ef = np.zeros([210,341])*np.nan
                yd[yd>th] = 0
                yd[yd<=th] = 1
                cdh_days = np.nan*np.zeros_like(yd)
                for i in range(210):
                    for j in range(341):
                        if mask[i,j]==1:
                            cdh_days[:,i,j],ef[i,j] = getEvents(yd[:,i,j], 3)

                ei = np.abs(ei*cdh_days)
                ei[ei==0] = np.nan
                Ycdh_days = np.nansum(cdh_days,axis=0)
                 
                Ycdh_i = np.nansum(ei,axis=0)
                
                cdh_area = Ycdh_days.copy()
                cdh_area[cdh_area>0] = 1
                cdh_area[cdh_area!=1] = 0
                cdh_area = cdh_area*Ar
                CDHDs.append(Ycdh_days)
                CDHAs.append(cdh_area)
                CDHIs.append(Ycdh_i)
                CDHFs.append(ef)
                print(y)
            
CDHD1s = np.array(CDHDs)*mask
CDHA1s = np.array(CDHAs)*mask
CDHF1s = np.array(CDHFs)*mask
CDHI1s = np.array(CDHIs)*mask

            
Period1 =['SSPs'] 
Period2 =['ssp126','ssp370','ssp585']
tp = ['2015_2020','2021_2030','2031_2040','2041_2050','2051_2060','2061_2070','2071_2080','2081_2090','2091_2100']
d1 = 'H:/data'

Th = np.zeros_like(mask)
sha = mask.shape
CDHI2s = {}
CDHA2s = {}
CDHF2s = {}
CDHD2s = {}
for period1 in Period1:
    
    for period2 in Period2:
        CDHDs = []
        CDHIs = []
        CDHFs = []
        CDHAs = []
        for timep in tp:
                           
               
            dscdhi = 'MME/scdhi/'+period1
             
            scdhi_dir = d1+'/'+dscdhi+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc'
          
            scdhidataset = nc.Dataset(scdhi_dir)
            lat = scdhidataset['lat']
            lon = scdhidataset['lon']
            [LONB,LATB] = np.meshgrid(lon,lat)
            scdhi = scdhidataset['scdhi'][:]
            dates = pd.date_range(start= timep[:4]+'-01-01', end=timep[5:]+'-12-31')
            
            Ar = calculate_grid_area(LATB)
            th = -1.3
            for y in range(int(timep[:4]),int(timep[5:])+1):
                
                yd = scdhi[dates.year==y,:,:].copy()
                ei = scdhi[dates.year==y,:,:].copy()
                
                ef = np.zeros([210,341])*np.nan
                yd[yd>th] = 0
                yd[yd<=th] = 1
                cdh_days = np.nan*np.zeros_like(yd)
                for i in range(210):
                    for j in range(341):
                        if mask[i,j]==1:
                            cdh_days[:,i,j],ef[i,j] = getEvents(yd[:,i,j], 3)

                ei = np.abs(ei*cdh_days)
                ei[ei==0] = np.nan
                Ycdh_days = np.nansum(cdh_days,axis=0)
                 
                Ycdh_i = np.nansum(ei,axis=0)
                
                cdh_area = Ycdh_days.copy()
                cdh_area[cdh_area>0] = 1
                cdh_area[cdh_area!=1] = 0
                cdh_area = cdh_area*Ar
                CDHDs.append(Ycdh_days)
                CDHAs.append(cdh_area)
                CDHIs.append(Ycdh_i)
                CDHFs.append(ef)
                print(period2+str(y))
        CDHD2s[period2] = np.array(CDHDs)*mask
        CDHA2s[period2] = np.array(CDHAs)*mask
        CDHF2s[period2] = np.array(CDHFs)*mask
        CDHI2s[period2] = np.array(CDHIs)*mask

                
import pickle
CDHEs_data  = {}
CDHEs_data['historical_frequency'] = CDHF1s
CDHEs_data['historical_days']  = CDHD1s
CDHEs_data['historical_area'] = CDHA1s   
CDHEs_data['historical_intensity'] = CDHI1s  
CDHEs_data['future_frequency'] = CDHF2s
CDHEs_data['future_days']  = CDHD2s
CDHEs_data['future_area'] = CDHA2s   
CDHEs_data['future_intensity'] = CDHI2s 

#CDHEs_data = pd.DataFrame(CDHEs_data)
with open('../data/Model_CDHEs_data.pkl', 'wb') as f:
	pickle.dump(CDHEs_data, f)            




width_cm = 18# 设置图形宽度
height_cm = 8 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
# 使用inch指定图形大小
fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(width_inch, height_inch),dpi=300)
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 12,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Arial'],#宋体
            'axes.unicode_minus': False, # 处理负号，即-号
            'figure.facecolor':'#FFFFFF'
         }
rcParams.update(config)


lines = []
 
 
line, = ax.plot(np.arange(1981,2015),np.nanmean(np.nanmean(CDHI1s,axis=1),axis=1),color = '#000000')
lines.append(line)
 
colors = ['#9AC9DB','#F8AC8C','#C82423']
for c in range(3):
    sn = Period2[c]

    pl = np.nanmean(np.nanmean(CDHI2s[sn],axis=1),axis=1)
    line, = plt.plot(np.arange(2015,2101),pl[:],color = colors[c])
    lines.append(line)
 
#ax.set_ylim([-1.2,-0.78])
ax.set_xlim([1980,2100])
#ax.vlines(x=2015,ymin=-1.2, ymax=-0.78,colors='#000000',ls = '--')

legend2 = ax.legend(lines,['historical','ssp126','ssp370','ssp585'],edgecolor= 'none',facecolor='none')


plt.title('CDHI')



