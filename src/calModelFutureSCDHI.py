# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:43:31 2025

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
from getSCDHI2 import getSCDHI

dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')
with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']

with open('../data/phatSCDHI.pkl', 'rb') as f:  # 读取pickle文件
    Phat = joblib.load(f)
    f.close()

 
MODEL = ['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0']


Period1 =['SSPs'] 
Period2 =['ssp126','ssp370','ssp585']
tp = ['2015_2020','2021_2030','2031_2040','2041_2050','2051_2060','2061_2070','2071_2080','2081_2090','2091_2100']
d1 = 'H:/data'
 

Th = np.zeros_like(mask)
sha = mask.shape
for model in MODEL:
    for period1 in Period1:
        for period2 in Period2:
            
            for timep in tp:
                
               
                d2sapei = model+'/sapei/'+period1
                d2sti = model+'/sti/'+period1
                sapei_dir = d1+'/'+d2sapei+'/'+period2+'_sapei_asia_daily_'+timep+'.nc'
                sti_dir = d1+'/'+d2sti+'/'+period2+'_sti_asia_daily_'+timep+'.nc'
                sapeidataset = nc.Dataset(sapei_dir)
                stidataset = nc.Dataset(sti_dir)
                sapeid = sapeidataset['sapei'][:]
                stid = stidataset['sti'][:]
                
                SCDHI = np.zeros_like(sapeid)*np.nan
                for i in range(sha[0]):
                    for j in range(sha[1]):
                        if mask[i,j]==1:
                            dx1 = sapeid[:,i,j]
                            dx2 = stid[:,i,j]
                            th = Phat['p0'][i,j]
                            t1 = Phat['p1'][:,i,j]
                            t2 = Phat['p2'][:,i,j]
                            scdhi = getSCDHI(dx1,dx2,th,t1,t2)
                            SCDHI[:,i,j] = scdhi
                                 
                        #print([i,j])
                
            
                ts = np.int16(timep[:4])
                te = np.int16(timep[5:])
                d2 = model+'/scdhi/'+period1
                if not os.path.exists(d1+'/'+d2):
                    os.makedirs(d1+'/'+d2)
                NewData = nc.Dataset(d1+'/'+d2+'/'+period2+'_scdhi_asia_daily_'+timep+'.nc', 'w', format='NETCDF4')
                NewData.description = 'the Standardized Compound Dry and Hot Index in Asia from '+timep[:4]+' to '+timep[5:]
                
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
                
                Gdata = NewData.createVariable('scdhi', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                                               least_significant_digit=3)
                Gdata.units = ' '
                Gdata.standard_name = 'the Standardized Compound Dry and Hot Index'
                Gdata.missing_value = -9999
                Gdata[:, :, :] = SCDHI
                
                NewData.close()    
                
                print(model+' '+period1+' '+period2+' '+timep)