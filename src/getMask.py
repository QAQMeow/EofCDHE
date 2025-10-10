# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:09:47 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

import os
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from PIL import Image
from GetPET import GetPET

dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_1979.nc')

Mask = dataBR['pre'][:,:,100]
Mask[~np.isnan(Mask)] = 1
lat = dataBR['lat'][:]
lon = dataBR['lon'][:]
LON,LAT = np.meshgrid(lon,lat)


Maskdata = {'Mask':Mask,'lat':lat,'lon':lon,'LON':LON,'LAT':LAT}

with open('../data/Maskdata.pkl', 'wb') as f:
	pickle.dump(Maskdata, f)







