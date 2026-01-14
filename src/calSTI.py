# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:09:06 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

import os
import numpy as np
import joblib
import netCDF4 as nc
import threading 
from scipy import stats
from datetime import datetime, timedelta


with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']


dataBR = nc.Dataset('H:/BR/Data/BR/pre/br_pre_'+str(1979)+'.nc')

Tem = []

for y in range(1981,2021):
    temdata = nc.Dataset('H:/BR/Data/BR/tem/br_tem_'+str(y)+'.nc')
    tem = np.array(temdata.variables['tem'])
    
    if y == 1981:
        Tem = tem[mask==1,:]
        
    else:
        Tem = np.concatenate((Tem,tem[mask==1,:]),axis=1,out=None)
        
    print(y)
    temdata.close()

sha = Tem.shape
N = 5
TemX = np.zeros_like(Tem)
for i in range(N,sha[1]):
    print(i)    
    TemX[:,i] = np.nanmean(Tem[:,i-N:i],axis=1)
        

mu = np.nanmean(Tem,axis = 1)
std= np.nanstd(Tem,axis = 1)


Z = np.zeros_like(Tem)
Z= []
for i in range(len(mu)):
    x = TemX[i,N:]
    cdf = stats.norm.cdf(x, mu[i], std[i])
    z = stats.norm.ppf(cdf,0,1)
    #Z[:,N:] = z
    Z.append(z)
    print(i)

Z = np.array(Z)
# def process2(Zx,TX,m,st):
    
#     for i in range(len(m)):
#         x = TX[i,N:]
#         cdf = stats.norm.cdf(x, m[i], st[i])
#         z = stats.norm.ppf(cdf,0,1)
#         Zx[:,N:] = z
#         print(i)
            
# s1 = np.array([ 0,0,0,0,0,0,0,0,0,0,0],dtype = int)
# e1 = np.array([2000,4000,6000,80000,10000,12000,14000,16000,18000,20000,sha[0]],dtype = int)

# T = {}
# for i in range(len(s1)):
#     T[i] = threading.Thread(target=process2,args=(Z[s1[i]:e1[i],:],TemX[s1[i]:e1[i],:],mu[s1[i]:e1[i]],std[s1[i]:e1[i]]))
# for i in range(len(s1)):
#     T[i].start()
# for i in range(len(s1)):
#     T[i].join()


STI = np.zeros([sha[1],210,380])*np.nan
for i in range(N,sha[1]):
    print(i)
    STI[i,mask==1] = Z[:,i-N]


NewData = nc.Dataset('H:/BR/Data/BR/sti/br_sti_1981_2020.nc', 'w', format='NETCDF4')
NewData.description = 'the standardized temperature index in Belt and Road from 1981 to 2020, 5days average'

time = NewData.createDimension('time', None)
lat = NewData.createDimension('lat', 210)
lon = NewData.createDimension('lon', 380)

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