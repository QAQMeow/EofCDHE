# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:30:41 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile as tf
import joblib
import cmaps
from getgridC import getCG
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

BR_Countries = pd.read_excel('../data/Asia_c.xlsx')

with open('../data/Model_CDHEs_data.pkl', 'rb') as f:
    CDHEs_data = joblib.load(f)   
    f.close()

cg = getCG()

with open('../data/Historical_ASPOP.pkl', 'rb') as f:  # 读取pickle文件
    PopH = joblib.load(f)
    f.close()
    
with open('../data/Future_ASPOP.pkl', 'rb') as f:
    PopF = joblib.load(f)
    f.close()
     
with open('../data/AgegroupGridData.pkl', 'rb') as f:  # 读取pickle文件
    Age_group = joblib.load(f)
    f.close()    


with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']
lat = maskdata['lat']
lon = maskdata['lon']

CDHF1s = CDHEs_data['historical_frequency']
CDHD1s = CDHEs_data['historical_days'] 
CDHA1s = CDHEs_data['historical_area'] 
CDHI1s = CDHEs_data['historical_intensity'] 
CDHF2s = CDHEs_data['future_frequency'] 
CDHD2s = CDHEs_data['future_days']  
CDHA2s = CDHEs_data['future_area'] 
CDHI2s = CDHEs_data['future_intensity'] 

hy = np.arange(1981,2015)
fy = np.arange(2015,2101)



with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']
lat = maskdata['lat']
lon = maskdata['lon']

def PLot1(ax,mlon,mlat,data,i,j,c,vmax,xl,yl):
    #(c = cmaps.NCV_jaisnd)
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import  BoundaryNorm
    
    #plt.title(y)
    #m = Basemap(projection='robin',resolution='l',lon_0=0)
    m = Basemap(ax = ax,projection = 'cyl',resolution='l',lon_0=0,llcrnrlon = 0, llcrnrlat = -15, urcrnrlon = 200, urcrnrlat = 90)
    m.fillcontinents(color = '#FFFFFF')
    #plt.text(153,40,y,fontsize = 7)
    LON,LAT = np.meshgrid(mlon,mlat)
    Lon = LON 
    xi, yi = m(Lon, LAT)
    #c =cmaps.sunshine_9lev
    # if vmax >100:
    #     bounds = np.arange(0,vmax,np.round(np.linspace(0, vmax,c.N)/10)[1]*10)
    #     bounds = np.insert(bounds,0,np.round(np.nanmin(data)/10)*10)
    # else:
    #     bounds = np.round(np.linspace(0, vmax,c.N),2)
    #     bounds = np.insert(bounds,0,np.round(np.nanmin(data),2))
    norm = mpl.colors.Normalize(vmin=0, vmax=np.nanpercentile(data,99))
    
    p = ax.pcolor(xi,yi,data, cmap=c,norm = norm,alpha = 0.9)
    
    #m.drawcoastlines() 
    #m.readshapefile('H:/65国家/一带一路/Export_Output_2', 'BR_countries',drawbounds=True,linewidth = 0.25,)
    m.readshapefile('H:/65国家/ASIA/AsiaSub', 'AS_countries',drawbounds=True,linewidth = 0.45)
    #m.readshapefile('H:/65国家/一带一路/Export_Output_6', 'BR_countries',drawbounds=True,linewidth = 0.45)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    from matplotlib.colors import Normalize
    #ax.text(150,20,[i,j])
    
    
    # num_bins = len(bounds)
    # d = data[data==data]
    # bin_indices = np.digitize(d, bounds,right=True) 
    # bin_indices = np.clip(bin_indices, 0, num_bins-1)
    # counts = np.bincount(bin_indices.flatten(), minlength=num_bins)
    
    # ax2 = ax.inset_axes([0.75, 0.1, .2, .36])
    # ca = np.concatenate([c._colors, np.full((c._colors.shape[0], 1), 0.9, dtype=np.float32)],axis=1)
    # ca = np.vstack((np.array([1,1,1,0]),ca))
    # ax2.pie(
    #    counts , colors=ca,
       
    #    startangle=140, radius=1.2, center=(1, 1),
    #    wedgeprops=dict(width=0.4, edgecolor='white',linewidth=0.4)
    #    ) 
    
    if i==0:
        ax.set_xlabel(xl, fontsize=12)
      
        ax.xaxis.set_label_coords(0.5, 1.15, transform=ax.transAxes)  # (x=0.5居中, y=1.05在轴上方)
        
    if j == 2:
        ax.set_ylabel(yl, fontsize=12,rotation = 270)
      
        ax.yaxis.set_label_coords(1.02, 0.5, transform=ax.transAxes) 
    #if i==2:    
    ax.set_xticks([ 0 ,30, 60, 90, 120, 150, 180],['0°' ,'30° E', '60° E', '90° E', '120° E', '150° E', '180°'])
    if j==0:    
        ax.set_yticks([-15,0, 30, 60, 90],['15° S' ,'0°', '30° N', '60° N', '90° N'])
    return p
    
    
width_cm = 20# 设置图形宽度
height_cm = 5 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54

# 使用inch指定图形大小
#fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)

gs = gridspec.GridSpec(1,3, hspace=0.1)

fig = plt.figure(figsize=(width_inch, height_inch),dpi=300)

from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 8,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Arial'],#宋体
            'axes.unicode_minus': False ,# 处理负号，即-号
            "figure.facecolor": (1,1,1,1)
         }
rcParams.update(config)  
va = ['d','f','i']
sn = ['ssp126','ssp370','ssp585']
van = ['CDHD','CDHF','CDHI']

cl = [ cmaps.MPL_pink_r, cmaps.MPL_PuBuGn,cmaps.MPL_Oranges]
Data = {'d':{'d1':CDHD1s,'d2':CDHD2s},'f':{'d1':CDHF1s,'d2':CDHF2s},'i':{'d1':CDHI1s,'d2':CDHI2s}}
barls = []
 
for j in range(3):
    ax = fig.add_subplot(gs[j])
    d1 = np.nanmean(Data[va[j]]['d1'][(hy>=1995),:,:],axis=0)
    #d2 = np.nanmean(Data[va[j]]['d2'][sn[2]][(fy>=2081)&(fy<=2100),:,:],axis=0)
    vmax = np.nanpercentile(d1,25)
    detla_d = d1
    p = PLot1(ax,lon,lat,detla_d,0,j,cl[j],vmax,van[j],'baseline')
    
    barls.append(p)


for j in range(3):            
    cbar_ax = fig.add_axes([0.05+j*0.34, 0.02, 0.23, 0.04])

    cbar = fig.colorbar(barls[j], cax=cbar_ax, orientation='horizontal',extend='max',shrink=0.2, aspect=40,
    pad=1,fraction=0.02,anchor=(0.1,0.1))
    cbar_ax.xaxis.set_ticks_position('bottom')
    #tic = cbar.get_ticks()[np.arange(0,len(cbar.get_ticks()),2)]
    #cbar.set_ticks(tic)
fig.subplots_adjust(left=0,right=1,top=1,bottom=0.1,wspace=0.02,hspace=0.01)




