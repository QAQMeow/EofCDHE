# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:31:05 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
import joblib
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']


with open('../data/AgegroupGridData.pkl', 'rb') as f:  # 读取pickle文件
    Age_group = joblib.load(f)
    f.close()


OldH = Age_group['old_h']
BabyH = Age_group['baby_h']
OldF = Age_group['old_f']
BabyF = Age_group['baby_f']

with open('../data/Historical_BRPOP.pkl', 'rb') as f:  # 读取pickle文件
    PopH = joblib.load(f)
    f.close()    

with open('../data/Future_BRPOP2.pkl', 'rb') as f:  # 读取pickle文件
    PopF = joblib.load(f)
    f.close()

#dataset = nc.Dataset('H:/BR/Data/BR/scdhi/br_scdhi_1981_2020.nc')
dataset = nc.Dataset('H:/data/GFDL-ESM4/scdhi/historical/historical_scdhi_br_daily_2011_2014.nc')
scdhi_f =dataset.variables['scdhi']
lat = dataset.variables['lat']
lon = dataset.variables['lon']
LON,LAT = np.meshgrid(lon,lat) 

 
import pandas as pd
from rasterstats import zonal_stats
from affine import Affine
 
import geopandas as gpd

import numpy as np
 
pt = pd.date_range(start='2011-01-01', end='2014-12-31',freq='D')
 
 
data1 = scdhi_f[pt.year==2014,:,:]
data1[data1>=-1] =0
data1[data1<=-1] = 1 
data2 = PopH[-1]*OldH[2020]
data = np.nanmean(data1,axis=0)*data2


d3 = np.nansum(data1,axis=0)[data2>0]
d4 = data2[data2>0]

dfx = pd.DataFrame({'x':d3,'y':d4})
dfx2 = dfx.sort_values(by = ['x'])
from scipy import stats
x =  dfx2['x']
y = dfx2['y']
 
cdfy = []
dy = 0
for i in range(len(y)):
    if i==0:
        dy = dy+0
    else:
        dy = dy+y[i]
    cdfy.append(dy)
    
plt.plot(x,np.array(cdfy)/1e6,c = '#000000')





dataset = nc.Dataset('H:/data/GFDL-ESM4/scdhi/SSPs/ssp585_scdhi_br_daily_2051_2060.nc')
scdhi_f =dataset.variables['scdhi']
lat = dataset.variables['lat']
lon = dataset.variables['lon']
LON,LAT = np.meshgrid(lon,lat) 

 
   
 
pt = pd.date_range(start='2051-01-01', end='2060-12-31',freq='D')
 
 
data1 = scdhi_f[pt.year==2051,:,:]
data1[data1>=-1] =0
data1[data1<=-1] = 1 
data2 =PopF['ssp1']['2050']*OldF['SSP1'][2050]
data = np.nanmean(data1,axis=0)*data2


d3 =  np.nansum(data1,axis=0)[data2>0]
d4 = data2[data2>0]

dfx = pd.DataFrame({'x':d3,'y':d4})
dfx2 = dfx.sort_values(by = ['x'])
from scipy import stats
x2 =  dfx2['x']
y2 = dfx2['y']

cdfy2 = []
dy = 0
for i in range(len(y2)):
    if i==0:
        dy = dy+0
    else:
        dy = dy+y2[i]
    cdfy2.append(dy)
    
plt.plot(x2,cdfy2,c = '#FCEC8D')
 



data1 = scdhi_f[pt.year==2051,:,:]
data1[data1>=-1] =0
data1[data1<=-1] = 1 
data2 =PopF['ssp3']['2050']*OldF['SSP3'][2050]
data = np.nanmean(data1,axis=0)*data2


d3 = np.nansum(data1,axis=0)[data2>0]
d4 = data2[data2>0]

dfx = pd.DataFrame({'x':d3,'y':d4})
dfx2 = dfx.sort_values(by = ['x'])
from scipy import stats
x3 =  dfx2['x']
y3 = dfx2['y']
 
cdfy3 = []
dy = 0
for i in range(len(y3)):
    if i==0:
        dy = dy+0
    else:
        dy = dy+y3[i]
    cdfy3.append(dy)
    
plt.plot(x3,cdfy3,c = '#FF9830')
 


data1 = scdhi_f[pt.year==2051,:,:]
data1[data1>=-1] =0
data1[data1<=-1] = 1 
data2 =PopF['ssp5']['2050']*OldF['SSP5'][2050]
data = np.nanmean(data1,axis=0)*data2


d3 = np.nansum(data1,axis=0)[data2>0]
d4 = data2[data2>0]

dfx = pd.DataFrame({'x':d3,'y':d4})
dfx2 = dfx.sort_values(by = ['x'])
from scipy import stats
x3 =  dfx2['x']
y3 = dfx2['y']
 
cdfy3 = []
dy = 0
for i in range(len(y3)):
    if i==0:
        dy = dy+0
    else:
        dy = dy+y3[i]
    cdfy3.append(dy)
    
plt.plot(x3,cdfy3,c = '#B51209')#plt.xlim([0,2])
plt.legend(['h','ssp1','ssp3','ssp5'])
# x_min = np.nanmin(lon)   # 左上角经度
# y_max = np.nanmax(lat) #左上角纬度
# pixel_width = 0.5  # 像元宽度（度）
# pixel_height = -0.5  # 像元高度（负值表示从上到下）

# # 创建 affine 变换对象
# affine = Affine(pixel_width, 0, x_min, 
#                 0, pixel_height, y_max)

# # 3. 读取矢量数据（替换为实际 Shapefile 路径）
# #vector_path ='H:\\65国家\\一带一路\\BRArea.shp'
# vector_path ='H:\\65国家\\一带一路\\Export_Output_6.shp'
# gdf = gpd.read_file(vector_path)

# # 4. 执行分区统计
# stats = zonal_stats(
#     vectors=gdf,              # 矢量数据
#     raster=data,              # 栅格数据（ndarray）
#     affine=affine,            # 地理参考
#     #stats=["count", "min", "max", "mean", "std", "sum"],  # 统计量
#     stats= ['sum', "mean"],
#     nodata=-9999,            # 指定无数据值
#     geojson_out=True          # 返回 GeoJSON 格式（保留几何信息）
# )

# # 5. 转换结果为 DataFrame
# stats_df = gpd.GeoDataFrame.from_features(stats)
# df2 = stats_df.drop_duplicates(subset=['SOC'], keep='last')
# # 打印结果
# print(df2[["NAME","sum"]])  # 假设矢量有 "NAME" 字段

 

