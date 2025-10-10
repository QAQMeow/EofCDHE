# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:06:32 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from PIL import Image
import threading
from scipy import stats
from datetime import datetime, timedelta
with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']

with open('../data/phatSTI.pkl', 'rb') as f:  # 读取pickle文件
    Phat = joblib.load(f)
    f.close()
    
dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')




 
MODEL = ['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0']


Period1 =['SSPs'] 
Period2 =['ssp126','ssp370','ssp585']
tp = ['2015_2020','2021_2030','2031_2040','2041_2050','2051_2060','2061_2070','2071_2080','2081_2090','2091_2100']
d1 = 'H:/data'
 
TemMu = Phat['Mu']
TemStd = Phat['Std'] 

for model in MODEL:
    for period1 in Period1:
        for period2 in Period2:
            
            for timep in tp:
                
               
                d2tas = model+'/tas/'+period1
                
                tas_dir = d1+'/'+d2tas+'/'+period2+'_tas_asia_daily_'+timep+'.nc'
                
                tasdataset = nc.Dataset(tas_dir)
                
                tas = tasdataset['tas'][:]
                
                
                tasdataset.close()
               
                tas = tas[:,mask==1]
             
                sha = np.shape(tas)
                ts = np.int16(timep[:4])
                te = np.int16(timep[5:])
                date_index = pd.date_range(start=timep[:4]+'-01-01', end=timep[5:]+'-12-31', freq='D')
                di2 =  pd.date_range(start='2004-01-01', end='2004-12-31', freq='D')
    
                STI = []
                for i in range(sha[0]):
                   
                    dt = date_index[i]
                    m = dt.month
                    d = dt.day
                    mu = np.squeeze(TemMu[(di2.month==m)&(di2.day==d),:])
                    std = np.squeeze(TemStd[(di2.month==m)&(di2.day==d),:])
                    x = tas[i,:]
                    z = []
                    
                    
                    cdf = stats.norm.cdf(x, mu, std)
                    
                    cdf[cdf <1e-16] = 1e-16
                    cdf[cdf >1-1e-16]= 1-1e-16
                    z = stats.norm.ppf(cdf,0,1)
                        
                    STI.append(z)
                  
    
                STI = np.squeeze(np.array(STI))
                sti = np.zeros([sha[0],210,341])*np.nan
                for i in range(sha[0]):
                    sti[i,mask==1] = STI[i,:]
                
            
                
        
                ts = np.int16(timep[:4])
                te = np.int16(timep[5:])
                
                stif = sti
                d2 = model+'/sti/'+period1
                if not os.path.exists(d1+'/'+d2):
                    os.makedirs(d1+'/'+d2)
                NewData = nc.Dataset(d1+'/'+d2+'/'+period2+'_sti_asia_daily_'+timep+'.nc', 'w', format='NETCDF4')
                NewData.description = 'the Standardized Temperature Index in Asia from '+timep[:4]+' to '+timep[5:]
    
                time = NewData.createDimension('time', None)
                lat = NewData.createDimension('lat', 210)
                lon = NewData.createDimension('lon', 341)
    
                times = NewData.createVariable("time", "f8", ("time",))
                times.units = 'days since 2015-1-1 00:00:00'
                times.axis = 'T'
                times.calendar = 'proleptic_gregorian'
                start_date = datetime(ts, 1, 1)
                end_date = datetime(te, 12, 31)
                delta = end_date - start_date
                num_days = delta.days + 1
                dates = [start_date + timedelta(days=i) for i in range(num_days)]
                time_values = nc.date2num(dates, units='days since 2015-1-1 00:00:00', calendar='proleptic_gregorian')
                times[:] = time_values
    
    
                latitudes = NewData.createVariable("lat", "f8", ("lat",))
                latitudes.units = dataBR.variables['lat'].units
                latitudes.axis = 'Y'
                latitudes[:] = dataBR.variables['lat'][:]
    
                longitudes = NewData.createVariable("lon", "f4", ("lon",))
                longitudes.units = dataBR.variables['lon'].units
                longitudes.axis = 'X'
                longitudes[:] = dataBR.variables['lon'][:]
    
                Gdata = NewData.createVariable('sti', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                                               least_significant_digit=3)
                Gdata.units = ' '
                Gdata.standard_name = 'the Standardized Temperature Index'
                Gdata.missing_value = -9999
                Gdata[:, :, :] = sti
    
                NewData.close()    
                
                print(model+' '+period1+' '+period2+' '+timep)
                
                
                
                
                