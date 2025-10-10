# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:16:29 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from PIL import Image
from GetPET import GetPET

dataAs = nc.Dataset('H:/BR/Data/Asia/scdhi/asia_scdhi_1981_2020.nc')

with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
Mask = maskdata['Mask']


with open('../data/Asia_DEM.pkl', 'rb') as f:  # 读取pickle文件
    alt = joblib.load(f)
    f.close()





MODEL = ['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0']


Period1 =['historical'] 
Period2 =['historical']
tp = ['1981_1990','1991_2000','2001_2010','2011_2014']
d1 = 'H:/data'

for model in MODEL:

    for timep in tp:
        for period1 in Period1:
            for period2 in Period2:
                d2 = model+'/pet/'+period1
                if not os.path.exists(d1+'/'+d2):
                    os.makedirs(d1+'/'+d2) 
                if not os.path.isfile(d1+'/'+d2+'/'+period2+'_pet_asia_daily_'+timep+'.nc'):
                   Dir4 = ['tasmax','tasmin','rlds','rsds','hurs','sfcWind']
                   Data = {}
                   for d4 in Dir4:
                   
                  
                   
                       d2s = model+'/'+d4+'/'+period1
                       File_dir = d1+'/'+d2s+'/'+period2+'_'+d4+'_asia_daily_'+timep+'.nc'
                   
                       dataset = nc.Dataset(File_dir)
                       Lat = dataset.variables['lat'][:]
                       Lon = dataset.variables['lon'][:]
                       T = dataset.variables['time'][:]
                       d = dataset.variables[d4][:]
                       PET = np.nan*np.zeros_like(d)
                 
                       Data[d4]= np.array(d[:,Mask==1]).astype(np.float32)
                       
   
               
                                   #tmax         tmin             rlds           rsds      hurs          wind10
                   pet = GetPET(Data[Dir4[0]], Data[Dir4[1]], Data[Dir4[2]], Data[Dir4[3]], Data[Dir4[4]], Data[Dir4[5]], alt[Mask==1])
                   PET[:,Mask==1] =pet
                   
                   NewData = nc.Dataset(d1+'/'+d2+'/'+period2+'_pet_asia_daily_'+timep+'.nc', 'w', format='NETCDF4')
                   NewData.description = period2+'_pet_asia_daily_'+timep
               
                   time = NewData.createDimension('time', None)
                   lat = NewData.createDimension('lat', 210)
                   lon = NewData.createDimension('lon', 341)
               
                   times = NewData.createVariable("time", "f8", ("time",))
                   times.units = dataset.variables['time'].units
                   times.axis = dataset.variables['time'].axis
                   times.calendar = dataset.variables['time'].calendar
                   times[:] = dataset.variables['time'][:]
               
                   latitudes = NewData.createVariable("lat", "f8", ("lat",))
                   latitudes.units = dataAs.variables['lat'].units
                   latitudes.axis = dataset.variables['lat'].axis
                   latitudes[:] = dataAs.variables['lat'][:]
               
                   longitudes = NewData.createVariable("lon", "f4", ("lon",))
                   longitudes.units = dataAs.variables['lon'].units
                   longitudes.axis = dataset.variables['lon'].axis
                   longitudes[:] = dataAs.variables['lon'][:]
               
                   Gdata = NewData.createVariable('pet', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                                                  least_significant_digit=3)
                   Gdata.units = 'mm/day'
                   Gdata.standard_name = 'potential evapotranspiration'
                   Gdata.missing_value = -9999
                   Gdata[:, :, :] = PET
               
                   NewData.close()
                   
                   print(model+' '+period2+'_pet_asia_daily_'+timep)
                  
                        
                        
                        