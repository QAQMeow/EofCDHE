# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 20:48:09 2025

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



dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')
with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']

 
 
MODEL = ['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','MME']

var = ['sti','sapei','scdhi']

Period1 =['historical'] 
Period2 =['historical']
tp = ['1981_1990','1991_2000','2001_2010','2011_2014']
d1 = 'H:/data'

Th = np.zeros_like(mask)
sha = mask.shape
YH = {}
for period1 in Period1:
 
    for period2 in Period2:
        YY = { }
        for m in MODEL:
            Y1 = []
            Y2 = []
            Y3 = []
            for timep in tp:
                               
                
               
                scdhidataset = nc.Dataset(d1+'/'+m+'/'+var[2]+'/'+period1+'/'+period2+'_'+var[2]+'_asia_daily_'+timep+'.nc')
                scdhi = scdhidataset[var[2]][:]
                sapeidataset = nc.Dataset(d1+'/'+m+'/'+var[1]+'/'+period1+'/'+period2+'_'+var[1]+'_asia_daily_'+timep+'.nc')
                sapei = sapeidataset[var[1]][:]
                stidataset = nc.Dataset(d1+'/'+m+'/'+var[0]+'/'+period1+'/'+period2+'_'+var[0]+'_asia_daily_'+timep+'.nc')
                sti = stidataset[var[0]][:]
                lat = scdhidataset['lat']
                lon = scdhidataset['lon']
                [LONB,LATB] = np.meshgrid(lon,lat)
                
                dates = pd.date_range(start= timep[:4]+'-01-01', end=timep[5:]+'-12-31')
                for y in range(int(timep[:4]),int(timep[5:])+1):
                    
                    
                    Y1.append(np.nanmean(sti[dates.year==y,:,:].copy(),axis=0))
                    Y2.append(np.nanmean(sapei[dates.year==y,:,:].copy(),axis=0))
                    Y3.append(np.nanmean(scdhi[dates.year==y,:,:].copy(),axis=0))
                    print(m+period2+str(y))
                
                
            Y1 = np.array(Y1)*mask   
            Y2 = np.array(Y2)*mask  
            Y3 = np.array(Y3)*mask  
            Data = {'sti':np.nanmean(np.nanmean(Y1,axis=1),axis=1),
                    'sapei':np.nanmean(np.nanmean(Y2,axis=1),axis=1),
                    'scdhi':np.nanmean(np.nanmean(Y3,axis=1),axis=1)}
            YY[m] = Data
        
        YH[period2] = YY


Period1 =['SSPs'] 
Period2 =['ssp126','ssp370','ssp585']
tp = ['2015_2020','2021_2030','2031_2040','2041_2050','2051_2060','2061_2070','2071_2080','2081_2090','2091_2100']
d1 = 'H:/data'

Th = np.zeros_like(mask)
sha = mask.shape
YF = {}
for period1 in Period1:

    for period2 in Period2:
        YY = {}
        for m in MODEL:
            Y1 = []
            Y2 = []
            Y3 = []
            for timep in tp:
                               
                   
                scdhidataset = nc.Dataset(d1+'/'+m+'/'+var[2]+'/'+period1+'/'+period2+'_'+var[2]+'_asia_daily_'+timep+'.nc')
                scdhi = scdhidataset[var[2]][:]
                sapeidataset = nc.Dataset(d1+'/'+m+'/'+var[1]+'/'+period1+'/'+period2+'_'+var[1]+'_asia_daily_'+timep+'.nc')
                sapei = sapeidataset[var[1]][:]
                stidataset = nc.Dataset(d1+'/'+m+'/'+var[0]+'/'+period1+'/'+period2+'_'+var[0]+'_asia_daily_'+timep+'.nc')
                sti = stidataset[var[0]][:]
                lat = scdhidataset['lat']
                lon = scdhidataset['lon']
                [LONB,LATB] = np.meshgrid(lon,lat)
                dates = pd.date_range(start= timep[:4]+'-01-01', end=timep[5:]+'-12-31')
                
                for y in range(int(timep[:4]),int(timep[5:])+1):
                    
                    Y1.append(np.nanmean(sti[dates.year==y,:,:].copy(),axis=0))
                    Y2.append(np.nanmean(sapei[dates.year==y,:,:].copy(),axis=0))
                    Y3.append(np.nanmean(scdhi[dates.year==y,:,:].copy(),axis=0))
                    print(m+period2+str(y))
                     
        
            Y1 = np.array(Y1)*mask   
            Y2 = np.array(Y2)*mask  
            Y3 = np.array(Y3)*mask  
            Data = {'sti':np.nanmean(np.nanmean(Y1,axis=1),axis=1),
                    'sapei':np.nanmean(np.nanmean(Y2,axis=1),axis=1),
                    'scdhi':np.nanmean(np.nanmean(Y3,axis=1),axis=1)}
            YY[m] = Data
        YF[period2] = YY

plt.plot(np.arange(1982,2015),YH['historical']['MME']['scdhi'][1:]);
plt.plot(np.arange(2016,2101),YF['ssp126']['MME']['scdhi'][1:]);
plt.plot(np.arange(2016,2101),YF['ssp370']['MME']['scdhi'][1:]);
plt.plot(np.arange(2016,2101),YF['ssp585']['MME']['scdhi'][1:]);
plt.legend(['h','126','370','585'])
Data = {'historical':YH,'future':YF}
import pickle
with open('../data/Model_data2.pkl', 'wb') as f:
	pickle.dump(Data, f)     