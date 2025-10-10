# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:28:28 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
 

import os
import numpy as np
import joblib
import pandas as pd
import netCDF4 as nc
import threading 

from datetime import datetime, timedelta


dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')
with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']

with open('../data/phatSCDHI.pkl', 'rb') as f:  # 读取pickle文件
    Phat = joblib.load(f)
    f.close()

 
MODEL = ['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0']


Period1 =['historical'] 
Period2 =['historical']
tp = ['1981_1990','1991_2000','2001_2010','2011_2014']
d1 = 'H:/data'

Th = np.zeros_like(mask)
sha = mask.shape

for period1 in Period1:
    for period2 in Period2:
        sc = []
        for timep in tp:
            k  = 0
            for model in MODEL:                
               
                dscdhi = model+'/scdhi/'+period1
                 
                scdhi_dir = d1+'/'+dscdhi+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc'
              
                scdhidataset = nc.Dataset(scdhi_dir)
               
                scdhi = scdhidataset['scdhi'][:]
                if k==0:
                    SCDHI = scdhi/4
                else:
                    SCDHI = SCDHI+scdhi/4
                k+=1
            
             
            ts = np.int16(timep[:4])
            te = np.int16(timep[5:])
            d2 = 'MME/scdhi/'+period1
            if not os.path.exists(d1+'/'+d2):
                os.makedirs(d1+'/'+d2)
            NewData = nc.Dataset(d1+'/'+d2+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc', 'w', format='NETCDF4')
            NewData.description = 'the Standardized Compound Dry and Hot Index in Asia from '+timep[:4]+' to '+timep[5:]
            
            time = NewData.createDimension('time', None)
            lat = NewData.createDimension('lat', 210)
            lon = NewData.createDimension('lon', 341)
            
            times = NewData.createVariable("time", "f8", ("time",))
            times.units = 'days since 1981-1-1 00:00:00'
            times.axis = 'T'
            times.calendar = 'proleptic_gregorian'
            start_date = datetime(ts, 1, 1)
            end_date = datetime(te, 12, 31)
            delta = end_date - start_date
            num_days = delta.days + 1
            dates = [start_date + timedelta(days=i) for i in range(num_days)]
            time_values = nc.date2num(dates, units='days since 1981-1-1 00:00:00', calendar='proleptic_gregorian')
            times[:] = time_values
            
            
            latitudes = NewData.createVariable("lat", "f8", ("lat",))
            latitudes.units = dataBR.variables['lat'].units
            latitudes.axis = 'Y'
            latitudes[:] = dataBR.variables['lat'][:]
            
            longitudes = NewData.createVariable("lon", "f4", ("lon",))
            longitudes.units = dataBR.variables['lon'].units
            longitudes.axis = 'X'
            longitudes[:] = dataBR.variables['lon'][:]
            
            Gdata = NewData.createVariable('scdhi', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                                           least_significant_digit=3)
            Gdata.units = ' '
            Gdata.standard_name = 'the Standardized Compound Dry and Hot Index'
            Gdata.missing_value = -9999
            Gdata[:, :, :] = SCDHI
            
            NewData.close()    
            
            print(period1+' '+period2+' '+timep)
            
            
            
            
            
            
            
            
            
            
# Period1 =['SSPs'] 
# Period2 =['ssp126','ssp370','ssp585']
# tp = ['2015_2020','2021_2030','2031_2040','2041_2050','2051_2060','2061_2070','2071_2080','2081_2090','2091_2100']
# d1 = 'H:/data'
 

# Th = np.zeros_like(mask)
# sha = mask.shape

# for period1 in Period1:
#     for period2 in Period2:
        
#         for timep in tp:
#             k = 0
#             for model in MODEL:                
               
#                 dscdhi = model+'/scdhi/'+period1
                 
#                 scdhi_dir = d1+'/'+dscdhi+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc'
              
#                 scdhidataset = nc.Dataset(scdhi_dir)
               
#                 scdhi = scdhidataset['scdhi'][:]
#                 if k==0:
#                     SCDHI = scdhi/4
#                 else:
#                     SCDHI = SCDHI+scdhi/4
#                 k+=1
            
            
#             ts = np.int16(timep[:4])
#             te = np.int16(timep[5:])
#             d2 = 'MME/scdhi/'+period1
#             if not os.path.exists(d1+'/'+d2):
#                 os.makedirs(d1+'/'+d2)
#             NewData = nc.Dataset(d1+'/'+d2+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc', 'w', format='NETCDF4')
#             NewData.description = 'the Standardized Compound Dry and Hot Index in Asia from '+timep[:4]+' to '+timep[5:]
            
#             time = NewData.createDimension('time', None)
#             lat = NewData.createDimension('lat', 210)
#             lon = NewData.createDimension('lon', 341)
            
#             times = NewData.createVariable("time", "f8", ("time",))
#             times.units = 'days since 2015-1-1 00:00:00'
#             times.axis = 'T'
#             times.calendar = 'proleptic_gregorian'
#             start_date = datetime(ts, 1, 1)
#             end_date = datetime(te, 12, 31)
#             delta = end_date - start_date
#             num_days = delta.days + 1
#             dates = [start_date + timedelta(days=i) for i in range(num_days)]
#             time_values = nc.date2num(dates, units='days since 2015-1-1 00:00:00', calendar='proleptic_gregorian')
#             times[:] = time_values
            
            
#             latitudes = NewData.createVariable("lat", "f8", ("lat",))
#             latitudes.units = dataBR.variables['lat'].units
#             latitudes.axis = 'Y'
#             latitudes[:] = dataBR.variables['lat'][:]
            
#             longitudes = NewData.createVariable("lon", "f4", ("lon",))
#             longitudes.units = dataBR.variables['lon'].units
#             longitudes.axis = 'X'
#             longitudes[:] = dataBR.variables['lon'][:]
            
#             Gdata = NewData.createVariable('scdhi', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
#                                            least_significant_digit=3)
#             Gdata.units = ' '
#             Gdata.standard_name = 'the Standardized Compound Dry and Hot Index'
#             Gdata.missing_value = -9999
#             Gdata[:, :, :] = SCDHI
            
#             NewData.close()    
            
#             print(period1+' '+period2+' '+timep)