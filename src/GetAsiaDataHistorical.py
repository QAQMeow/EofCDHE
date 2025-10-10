# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:15:36 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
import os
import joblib
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RectBivariateSpline
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
    m = int(S[2]/2)
    pp = data[:,:,m:].copy()
    pp2 = data[:,:,:m].copy()
    d[:,:,:m] = pp
    d[:,:,m:] = pp2
    return d


# 方法1：使用 griddata 进行插值（适用于不规则网格）
def interpolate_with_griddata(X, Y, Z, points):
    # 将网格数据转换为点列表
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    grid_values = Z.ravel()
    
    # 使用 griddata 进行插值
    # method 可以是 'linear', 'nearest', 'cubic'
    interpolated_values = griddata(grid_points, grid_values, points, method='nearest')
    
    return interpolated_values

# 方法2：使用 RectBivariateSpline 进行插值（适用于规则网格，效率更高）
def interpolate_with_spline(x, y, Z, points):
    # 创建插值函数
    spline = RectBivariateSpline(x, y, Z.T)  # 注意这里需要转置Z
    
    # 提取要插值的点的坐标
    xi = points[:, 0]
    yi = points[:, 1]
    
    # 进行插值
    interpolated_values = spline.ev(xi, yi)
    
    return interpolated_values

def sq2grid(data,LAT,LON,S,points):
    
    gridd =np.zeros(S)*np.nan
    for i in range(len(points)):
        y = np.where((LAT==points[i,1])&(LON==points[i,0]))[0][0]
        x = np.where((LAT==points[i,1])&(LON==points[i,0]))[1][0]
        gridd[:,y,x] = data[:,i]
    return gridd

with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']



MODEL = ['GFDL-ESM4',]#'GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0'

VAR  = ['hurs','huss','pr','rlds','rsds','sfcWind','tas','tasmin','tasmax']

Period1 =['historical'] 
Period2 =['historical']
tp = ['1981_1990','1991_2000','2001_2010','2011_2014']
d1 = 'H:/data'

for model in MODEL:
    for var in VAR:
        for timep in tp:
            for period1 in Period1:
                for period2 in Period2:
                    d2 = model+'/'+var+'/'+period1
                    if not os.path.isfile(d1+'/'+d2+'/'+period2+'_'+var+'_asia_daily_'+timep+'.nc'):
                        dataset = nc.Dataset('I:/ISIMIP3b/'+model+'/'+var+'/'+period1+'/'+period2+'_'+var+'_global_daily_'+timep+'.nc')
                        lat = dataset['lat'][:]
                        lon = dataset['lon'][:]+180
                        time = dataset['time']
                        data = dataset[var+'Adjust'][:]
                        
                        [LON,LAT]  = np.meshgrid(lon,lat)
                        data = fliplrMap(data)
                        
                        dataBR = nc.Dataset('H:/BR/Data/Asia/scdhi/asia_scdhi_1981_2020.nc')
                        lonBR = dataBR['lon']
                        latBR = dataBR['lat']
                        scdhi = dataBR['scdhi']
                        [LONB,LATB] = np.meshgrid(lonBR,latBR)
                        
                         
                        # lat = mat_data['BR_grid'][0]
                        # lon = mat_data['BR_grid'][1]
                        points  = np.array([LONB[mask==1].data,LATB[mask==1].data]).T
                        
                        X = lon
                        Y = lat[::-1]
                        BRD = []
                        for i in range(len(time)):
                        
                        
                            Z = data[i]
                            Z = np.flipud(Z)
                         #interp_data =  interpolate_with_spline(Maskdata['lon'][:], Maskdata['lat'][:], Z, points )
                        
                            interp_data =  interpolate_with_spline(X,Y, Z, points)
                            #Brd  = sq2grid(interp_data, LATB, LONB, points) 
                            BRD.append(interp_data)
                        
                        
                        BRD = np.array(BRD)
                        S = [len(time),210,341]
                        BRD2 =sq2grid(BRD, LATB, LONB,S, points) 
                        
                       
                        
                        NewData = nc.Dataset(d1+'/'+d2+'/'+period2+'_'+var+'_asia_daily_'+timep+'.nc', 'w', format='NETCDF4')
                        NewData.description = period2+'_'+var+'_global_daily_'+timep
                    
                        time = NewData.createDimension('time', None)
                        lat = NewData.createDimension('lat', 210)
                        lon = NewData.createDimension('lon', 341)
                    
                        times = NewData.createVariable("time", "f8", ("time",))
                        times.units = dataset.variables['time'].units
                        times.axis = dataset.variables['time'].axis
                        times.calendar = dataset.variables['time'].calendar
                        times[:] = dataset.variables['time'][:]
                    
                        latitudes = NewData.createVariable("lat", "f8", ("lat",))
                        latitudes.units = dataBR.variables['lat'].units
                        latitudes.axis = dataset.variables['lat'].axis
                        latitudes[:] = dataBR.variables['lat'][:]
                    
                        longitudes = NewData.createVariable("lon", "f4", ("lon",))
                        longitudes.units = dataBR.variables['lon'].units
                        longitudes.axis = dataset.variables['lon'].axis
                        longitudes[:] = dataBR.variables['lon'][:]
                    
                        Gdata = NewData.createVariable(var, "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                                                       least_significant_digit=3)
                        Gdata.units = dataset.variables[var+'Adjust'].units
                        Gdata.standard_name = var
                        Gdata.missing_value = -9999
                        Gdata[:, :, :] = BRD2
                    
                        NewData.close()
                        
                    print(model+' '+period2+'_'+var+'_global_daily_'+timep)












