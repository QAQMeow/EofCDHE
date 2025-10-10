# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:09:24 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

import os
import numpy as np
import joblib
import netCDF4 as nc
import threading 

from datetime import datetime, timedelta
from getSCDHI import getSCDHI

with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']


N = 90
dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')

SAPEIDATA = nc.Dataset('H:/BR/Data/Asia/sapei/asia_sapei_1981_2020.nc')
SAPEI = SAPEIDATA['sapei'][:]

STIDATA = nc.Dataset('H:/BR/Data/Asia/sti/asia_sti_1981_2020.nc')
STI = STIDATA['sti'][:]

Th = np.zeros_like(mask)
Thucdf1 = np.zeros([2,np.shape(mask)[0],np.shape(mask)[1]])
Thucdf2 = np.zeros([2,np.shape(mask)[0],np.shape(mask)[1]])
sha = mask.shape
SCDHI = np.zeros_like(STI)
for i in range(sha[0]):
    for j in range(sha[1]):
        if mask[i,j]==1:
            d1 = SAPEI[N:,i,j]
            d2 = STI[N:,i,j]
            scdhi,th,t1,t2 = getSCDHI(d1,d2)
            SCDHI[N:,i,j] = scdhi
            Thucdf1[:,i,j] = t1
            Thucdf2[:,i,j] = t2
            Th[i,j] = th     
        print([i,j])

phat = {'p0':Th,'p1':Thucdf1,'p2':Thucdf2}
with open('../data/phatSCDHI.pkl', 'wb') as f:  # 读取pickle文件

    joblib.dump(phat,f)
    f.close()

SCDHI[:,np.isnan(mask)]= np.nan




NewData = nc.Dataset('H:/BR/Data/Asia/scdhi/asia_scdhi_1981_2020.nc', 'w', format='NETCDF4')
NewData.description = 'the Standardized Compound Dry and Hot Index in Asia from 1981 to 2020, 5days average'

time = NewData.createDimension('time', None)
lat = NewData.createDimension('lat', 210)
lon = NewData.createDimension('lon', 341)

times = NewData.createVariable("time", "f8", ("time",))
times.units = 'days since 1981-1-1 00:00:00'
times.axis = 'T'
times.calendar = 'proleptic_gregorian'
start_date = datetime(1981, 1, 1)
end_date = datetime(2020, 12, 31)
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
Gdata.standard_name = 'the Standardized Compound Dry and Hot Index '
Gdata.missing_value = -9999
Gdata[:, :, :] = SCDHI

NewData.close()    










