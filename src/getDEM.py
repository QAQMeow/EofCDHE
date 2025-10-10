# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:29:09 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

import os
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from PIL import Image
from scipy.interpolate import griddata, RectBivariateSpline

with open('../data/Mask.pkl', 'rb') as f:  # 读取pickle文件
    Maskdata = joblib.load(f)
    
    f.close()
    
LON = Maskdata['LON']
LAT = Maskdata['LAT']    
 
img = Image.open('H:/65国家/ASIA/GlobalDEM.tif')
alt = np.array(img)

a1 = alt[:,360:].copy()
a2 = alt[:,:360].copy()
alt[:,:360] = a1
alt[:,360:] = a2
altA = alt[0:210,49:390]

#altA = np.flipud(altA)


with open('../data/Asia_DEM.pkl', 'wb') as f:
 	pickle.dump(altA, f)

