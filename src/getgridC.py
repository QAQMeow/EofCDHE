# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 17:14:32 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
import rasterio 
import joblib
import pickle
import numpy as np
import h5py
from scipy.interpolate import griddata, RectBivariateSpline
import netCDF4 as nc
from PIL import Image
import matplotlib.pyplot as plt
def getCG():
    
    
    
    fr ='H:/65国家/ASIA/AsiaGrid.tif'
    tif =  Image.open(fr )
    countries = np.float32(tif)
    def fliplrMap(data):
        '''
        change  central meridian 0 to 180 
    
        Parameters                       
        -------      
        data : numpy.array,MxN          
               Array with central meridian 0    
        
        Returns
        -------
        d : numpy.array,MxN
            Array with central meridian 180
         
        '''
        d = data.copy()
        S  = np.shape(d)
        m = int(S[1]/2)
        pp = data[:,m:].copy()
        pp2 = data[:,:m].copy()
        d[:,:m] = pp
        d[:,m:] = pp2
        return d
    
    
    countries = fliplrMap(countries)
    countries = countries[:,49:390]
    countries[countries==127] = np.nan
    countries[countries==0] = np.nan
    return countries