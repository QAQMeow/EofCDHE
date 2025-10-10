# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 19:21:11 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""



import os
import numpy as np
import pandas as pd
import joblib
import netCDF4 as nc
import threading 
from scipy import stats
from datetime import datetime, timedelta


with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']


dataBR = nc.Dataset('H:/BR/Data/ASia/pre/asia_pre_'+str(1979)+'.nc')

Tem = []

for y in range(1981,2021):
    temdata = nc.Dataset('H:/BR/Data/Asia/tem/asia_tem_'+str(y)+'.nc')
    tem = np.array(temdata.variables['tem'])
    
    if y == 1981:
        Tem = tem[mask==1,:]
        
    else:
        Tem = np.concatenate((Tem,tem[mask==1,:]),axis=1,out=None)
        
    print(y)
    temdata.close()

sha = Tem.shape

TemMu = []#np.zeros(366,210,380)
TemStd = []# np.zeros(366,210,380)
date_index = pd.date_range(start='1981-01-01', end='2020-12-31', freq='D')
Md = [31,28,31,30,31,30,31,31,30,31,30,31]
for m in range(1,13):
    for d in range(1,Md[m-1]+1):
        data = Tem[:,(date_index.month == m)&(date_index.day==d)]
        
        TemMu.append(np.nanmean(data,axis=1))
        TemStd.append(np.nanstd(data,axis=1))
TemMu.append(np.nanmean(data,axis=1))
TemMu = np.array(TemMu)
TemStd.append(np.nanstd(data,axis=1))
TemStd = np.array(TemStd)
# mu = np.nanmean(Tem,axis = 1)
# std= np.nanstd(Tem,axis = 1)
di2 =  pd.date_range(start='2004-01-01', end='2004-12-31', freq='D')


phat = {'Mu':TemMu,'Std':TemStd}
with open('../data/phatSTI.pkl', 'wb') as f:  # 读取pickle文件

    joblib.dump(phat,f)
    f.close()



Z = []
for i in range(sha[1]):
   
    dt = date_index[i]
    m = dt.month
    d = dt.day
    mu = np.squeeze(TemMu[(di2.month==m)&(di2.day==d),:])
    std = np.squeeze(TemStd[(di2.month==m)&(di2.day==d),:])
    x = Tem[:,i]
    z = []
    
    
    cdf = stats.norm.cdf(x, mu, std)
    
    cdf[cdf <1e-16] = 1e-16
    cdf[cdf >1-1e-16]= 1-1e-16
    z = stats.norm.ppf(cdf,0,1)
        
    Z.append(z)
    print(i)

Z = np.squeeze(np.array(Z))




# Z = np.zeros_like(Tem)
# Z= []
# for i in range(len(mu)):
#     x = TemX[i,N:]
#     cdf = stats.norm.cdf(x, mu[i], std[i])
#     z = stats.norm.ppf(cdf,0,1)
#     #Z[:,N:] = z
#     Z.append(z)
#     print(i)

# Z = np.array(Z)


STI = np.zeros([sha[1],210,341])*np.nan
for i in range(sha[1]):
    print(i)
    STI[i,mask==1] = Z[i,:]


NewData = nc.Dataset('H:/BR/Data/Asia/sti/asia_sti_1981_2020.nc', 'w', format='NETCDF4')
NewData.description = 'the standardized temperature index in Asia from 1981 to 2020, 5days average'

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

Gdata = NewData.createVariable('sti', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                               least_significant_digit=3)
Gdata.units = ' '
Gdata.standard_name = 'the standardized temperature index '
Gdata.missing_value = -9999
Gdata[:, :, :] = STI

NewData.close()    