# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:25:51 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 
import os
import numpy as np
import joblib
import pandas as pd
import netCDF4 as nc
import threading 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta



dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')
with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']

with open('../data/phatSCDHI.pkl', 'rb') as f:  # 读取pickle文件
    Phat = joblib.load(f)
    f.close()

 
MODEL = ['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','MME']

var = ['sti','sapei','scdhi']

Period1 =['historical'] 
Period2 =['historical']
tp = ['1981_1990','1991_2000','2001_2010','2011_2014']
d1 = 'H:/data'

# Th = np.zeros_like(mask)
# sha = mask.shape
# YH = {}
# for period1 in Period1:
 
#     for period2 in Period2:
#         YY = { }
#         for m in MODEL:
#             Y1 = []
#             Y2 = []
#             Y3 = []
#             for timep in tp:
                               
                
               
#                 scdhidataset = nc.Dataset(d1+'/'+m+'/'+var[2]+'/'+period1+'/'+period2+'_'+var[2]+'_br_daily_'+timep+'.nc')
#                 scdhi = scdhidataset[var[2]][:]
#                 sapeidataset = nc.Dataset(d1+'/'+m+'/'+var[1]+'/'+period1+'/'+period2+'_'+var[1]+'_br_daily_'+timep+'.nc')
#                 sapei = sapeidataset[var[1]][:]
#                 stidataset = nc.Dataset(d1+'/'+m+'/'+var[0]+'/'+period1+'/'+period2+'_'+var[0]+'_br_daily_'+timep+'.nc')
#                 sti = stidataset[var[0]][:]
#                 lat = scdhidataset['lat']
#                 lon = scdhidataset['lon']
#                 [LONB,LATB] = np.meshgrid(lon,lat)
                
#                 dates = pd.date_range(start= timep[:4]+'-01-01', end=timep[5:]+'-12-31')
#                 for y in range(int(timep[:4]),int(timep[5:])+1):
                    
                    
#                     Y1.append(np.nanmean(sti[dates.year==y,:,:].copy(),axis=0))
#                     Y2.append(np.nanmean(sapei[dates.year==y,:,:].copy(),axis=0))
#                     Y3.append(np.nanmean(scdhi[dates.year==y,:,:].copy(),axis=0))
#                     print(m+period2+str(y))
                
                
#             Y1 = np.array(Y1)*mask   
#             Y2 = np.array(Y2)*mask  
#             Y3 = np.array(Y3)*mask  
#             Data = {'sti':np.nanmean(np.nanmean(Y1,axis=1),axis=1),
#                     'sapei':np.nanmean(np.nanmean(Y2,axis=1),axis=1),
#                     'scdhi':np.nanmean(np.nanmean(Y3,axis=1),axis=1)}
#             YY[m] = Data
        
#         YH[period2] = YY


# Period1 =['SSPs'] 
# Period2 =['ssp126','ssp370','ssp585']
# tp = ['2015_2020','2021_2030','2031_2040','2041_2050','2051_2060','2061_2070','2071_2080','2081_2090','2091_2100']
# d1 = 'H:/data'

# Th = np.zeros_like(mask)
# sha = mask.shape
# YF = {}
# for period1 in Period1:

#     for period2 in Period2:
#         YY = {}
#         for m in MODEL:
#             Y1 = []
#             Y2 = []
#             Y3 = []
#             for timep in tp:
                               
                   
#                 scdhidataset = nc.Dataset(d1+'/'+m+'/'+var[2]+'/'+period1+'/'+period2+'_'+var[2]+'_br_daily_'+timep+'.nc')
#                 scdhi = scdhidataset[var[2]][:]
#                 sapeidataset = nc.Dataset(d1+'/'+m+'/'+var[1]+'/'+period1+'/'+period2+'_'+var[1]+'_br_daily_'+timep+'.nc')
#                 sapei = sapeidataset[var[1]][:]
#                 stidataset = nc.Dataset(d1+'/'+m+'/'+var[0]+'/'+period1+'/'+period2+'_'+var[0]+'_br_daily_'+timep+'.nc')
#                 sti = stidataset[var[0]][:]
#                 lat = scdhidataset['lat']
#                 lon = scdhidataset['lon']
#                 [LONB,LATB] = np.meshgrid(lon,lat)
#                 dates = pd.date_range(start= timep[:4]+'-01-01', end=timep[5:]+'-12-31')
                
#                 for y in range(int(timep[:4]),int(timep[5:])+1):
                    
#                     Y1.append(np.nanmean(sti[dates.year==y,:,:].copy(),axis=0))
#                     Y2.append(np.nanmean(sapei[dates.year==y,:,:].copy(),axis=0))
#                     Y3.append(np.nanmean(scdhi[dates.year==y,:,:].copy(),axis=0))
#                     print(m+period2+str(y))
                     
        
#             Y1 = np.array(Y1)*mask   
#             Y2 = np.array(Y2)*mask  
#             Y3 = np.array(Y3)*mask  
#             Data = {'sti':np.nanmean(np.nanmean(Y1,axis=1),axis=1),
#                     'sapei':np.nanmean(np.nanmean(Y2,axis=1),axis=1),
#                     'scdhi':np.nanmean(np.nanmean(Y3,axis=1),axis=1)}
#             YY[m] = Data
#         YF[period2] = YY

# plt.plot(np.arange(1982,2015),np.nanmean(np.nanmean(Y1,axis=1),axis=1)[1:]);
# plt.plot(np.arange(2016,2101),YF[0][1:]);
# plt.plot(np.arange(2016,2101),YF[1][1:]);
# plt.plot(np.arange(2016,2101),YF[2][1:]);plt.legend(['h','126','370','585'])
# Data = {'historical':YH,'future':YF}
# import pickle
# with open('../data/Model_data2.pkl', 'wb') as f:
# 	pickle.dump(Data, f)     

with open('../data/Model_data2.pkl', 'rb') as f:  # 读取pickle文件
    Data = joblib.load(f)
    f.close()
YH = Data['historical']
YF = Data['future']
Period2 =['ssp126','ssp370','ssp585']
YL = ['STI','SAPEI','SCDHI']
tit  = ['(a)','(b)','(c)']
width_cm = 16# 设置图形宽度
height_cm = 4 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
# 使用inch指定图形大小
fig,axes = plt.subplots(nrows=1, ncols=3,figsize=(16,4),dpi=300)
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 12,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Arial'],#宋体
            'axes.unicode_minus': False, # 处理负号，即-号
            'figure.facecolor':'#FFFFFF'
         }
rcParams.update(config)
for i in range(3):
    ax = axes[i]
    Dp = []
    for k in YH['historical'].keys():
        Dp.append(YH['historical'][k][var[i]])

    Dp=np.array(Dp)
    pub = np.nanmax(Dp,axis=0)
    plb = np.nanmin(Dp,axis=0)
    pl = YH['historical']['MME'][var[i]]

    lines = []
    fills = []
    f = ax.fill_between(np.arange(1982,2015),pub[1:],plb[1:],color = '#000000',alpha = 0.2,edgecolor = 'none')
    line, = ax.plot(np.arange(1982,2015),pl[1:],color = '#000000')
    lines.append(line)
    fills.append(f)
    colors = ['#FCEC8D', '#FF9830', '#B51209']
    for c in range(3):
        sn = Period2[c]
        Dp = []
        for k in YF[sn].keys():
            Dp.append(YF[sn][k][var[i]])

        Dp=np.array(Dp)
        pub = np.nanmax(Dp,axis=0)
        plb = np.nanmin(Dp,axis=0)
        pl = YF[sn]['MME'][var[i]]



        f = ax.fill_between(np.arange(2016,2101),pub[1:],plb[1:],color = colors[c],alpha = 0.3,edgecolor = 'none')
        line, = ax.plot(np.arange(2016,2101),pl[1:],color = colors[c])
        lines.append(line)
        fills.append(f)
    #ax.set_ylim([-1.2,-0.78])
    ax.set_xlim([1980,2100])
    ax.set_ylabel(YL[i])
    ax.vlines(x=2015,ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],colors='#444444',ls = '--')
    ax.text(0.01,0.94, tit[i], transform=ax.transAxes)
    #legend1 = ax.legend(fills,['historical','ssp126','ssp370','ssp585'],edgecolor= 'none')
    #legend2 = ax.legend(lines,['historical','ssp126','ssp370','ssp585'],edgecolor= 'none',facecolor='none')
fig.legend(fills, ['historical','ssp126','ssp370','ssp585'],loc = 'lower center',edgecolor = 'none',ncols = 4,bbox_to_anchor=(0.5, -0.06))
fig.legend(lines, ['historical','ssp126','ssp370','ssp585'],loc = 'lower center',edgecolor = 'none',facecolor='none',ncols = 4,bbox_to_anchor=(0.5, -0.06))
fig.subplots_adjust(left=0.15,right=0.85,top=0.75,bottom=0.1,wspace=0.4,hspace=0.01)


