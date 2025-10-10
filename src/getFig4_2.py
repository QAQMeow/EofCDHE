# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:20:38 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 
import os
import numpy as np
import joblib
import pandas as pd
import netCDF4 as nc
from getgridC import getCG
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

BR_Countries = pd.read_excel('../data/Asia_c.xlsx')
 

with open('../data/Model_CDHEs_data2.pkl', 'rb') as f:
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
    
with open('../data/Model_CDHEs_data2.pkl', 'rb') as f:  # 读取pickle文件
    DATA = joblib.load(f)
    f.close()
mask = maskdata['Mask']
lat = maskdata['lat']
lon = maskdata['lon']

MODEL = ['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','MME']

 


SN1 = ['historical','SSP1','SSP3','SSP5']
SN2 = ['historical','ssp126','ssp370','ssp585']

AreaC = ['东亚','中亚','西亚','北亚','南亚','东南亚']
AreaE = ['Europe','East Asia','Cental Asia','Western Asia','Southern Asia','Southeast Asia']
 
fy =  np.arange(2015,2101,1) 
Years = np.arange(2020,2101,5)
Data = {}


Years2 = np.arange(1990,2021,1)
Years3 = np.arange(1981,2015,1)
def getData(D,mo):
    CDHF1s = D['h'][mo]['historical_frequency']
    CDHD1s = D['h'][mo]['historical_days'] 
    CDHA1s = D['h'][mo]['historical_area'] 
    CDHI1s = D['h'][mo]['historical_intensity'] 
    
    CDHF2s = D['f'][mo]['future_frequency'] 
    CDHD2s = D['f'][mo]['future_days']  
    CDHA2s = D['f'][mo]['future_area'] 
    CDHI2s = D['f'][mo]['future_intensity'] 
    cpop = []
    cbaby = []
    cold = []
    p1 = []
    p2 = []
    p3 = []
    e = []
    f = []
    d = []
    ss = []
    
    for y in range(1995,2015):
        pop = np.squeeze(PopH[Years2==y])
        fb = Age_group['baby_h'][y]
        fo = Age_group['old_h'][y]
         
        p1.append(pop)
        p2.append(pop*(fb+fo))
        p3.append(pop*(fb+fo))
        f.append(CDHF1s[Years3==y])
        d.append(CDHD1s[Years3==y])
        ss.append(CDHI1s[Years3==y])
        e.append(pop*(fb+fo)*CDHD1s[Years3==y])
    pa = np.array(p1)
    pb = np.array(p2)
    po = np.array(p3)
    pe = np.array(e)
         
    
    Data['historical'] ={'pop_all':np.squeeze(pa),'pop_old':np.squeeze(po),'pop_baby':np.squeeze(pb),
                         'e':np.squeeze(pe),'f':np.squeeze(np.array(f)),'d':np.squeeze(np.array(d)),'i':np.squeeze(np.array(ss))
                         }

    for s in range(1,4):
       
        cpop = []
        cbaby = []
        cold = []
        p1 = []
        p2 = []
        p3 = []
        e = []
        f = []
        d = []
        ss = []
        for i in range(17):
            pop = np.squeeze(PopF[SN1[s]][str(Years[i])])
            fb = Age_group['baby_f'][SN1[s]][Years[i]]
            fo = Age_group['old_f'][SN1[s]][Years[i]]
             
            p1.append(pop)
            p2.append(pop*(fb+fo))
            p3.append(pop*(fb+fo))
            f.append(CDHF2s[SN2[s]][fy==Years[i]])
            d.append((CDHD2s[SN2[s]][fy==Years[i]]))
            ss.append((CDHI2s[SN2[s]][fy==Years[i]]))
            e.append(pop*(fb+fo)*CDHD2s[SN2[s]][fy==Years[i]])
        pa = np.array(p1)
        pb = np.array(p2)
        po = np.array(p3)
        pe = np.array(e)
             
        
        Data[SN1[s]] = {'pop_all':np.squeeze(pa),'pop_old':np.squeeze(po),'pop_baby':np.squeeze(pb),
                        'e':np.squeeze(pe),'f':np.squeeze(np.array(f)),'d':np.squeeze(np.array(d)),'i':np.squeeze(np.array(ss))
                        
                        }
    return Data


def getfill(DATA,s,v,timep):
    X = []
    Y = []
    for m in MODEL:
        Data = getData(DATA, m)
            
        if s == 'historical':
            x = np.squeeze(Data[s][v][-1])
            #x0 = Data[s][v][-1]
            #x1 = x0.copy()
            #x1[x1>0] = 1
            #x1[x1!=1] = 0
            #x = x0*x1
            #y0 = Data[s]['pop_baby'][-1]
            #y = y0*x1
            y = np.squeeze(Data[s]['pop_baby'][-1])
            linex = np.nanmean(x)
            
        else:
            #x0 = np.nanmean(Data[s][v][(Years>=2020)&(Years<=2040)],axis=0)
            #x1 = x0.copy()
            #x1[x1>0] = 1
            #x1[x1!=1] = 0
            #x = x0*x1
            x = np.squeeze(np.nanmean(Data[s][v][timep],axis=0))
            #y0 = np.nanmean(Data[s]['pop_baby'][(Years>=2020)&(Years<=2040)],axis=0)
            #y = y0*x1
            y = np.squeeze(np.nanmean(Data[s]['pop_baby'][timep],axis=0))
        x = x[y==y]
        y = y[y==y]
         
        d1 = pd.DataFrame({'x':x,'y':y })
        d2 = d1.sort_values(by='x')
        
        y2 = [ ]
        yy = 0
        
        for i in range(len(x)):
            if np.isnan(d2['y'][i]):
                yy = yy+0
            else:
                yy = yy+d2['y'][i]
            y2.append(yy)
        
        
        xp,yp = smooth_line(d2['x'], np.array(y2), new_points=100, sigma = 3.5)
        X.append(xp)
        Y.append(yp)
    Y = np.array(Y)
    X = np.array(X)
    
    return X[0],np.nanmax(Y,axis=0),np.nanmin(Y,axis=0),np.nanmedian(Y,axis=0)






from scipy.interpolate import make_interp_spline, interp1d
def smooth_line(x, y, new_points=200, sigma=1.5):
    """
    调整x坐标（增加采样点）并应用高斯滤波平滑
    
    参数:
        x: 原始x坐标
        y: 原始y值
        new_points: 新的x坐标数量（增加采样密度）
        sigma: 高斯滤波标准差
        
    返回:
        x_new: 新的密集x坐标
        y_smoothed: 平滑后的y值
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d

    # 步骤1：生成更密集的x坐标
    x_new = np.linspace(x.min(), x.max(), new_points)
    
    # 步骤2：插值得到密集y值（保持原始趋势）
    interp_func = interp1d(x, y, kind='linear')
    y_dense = interp_func(x_new)
    
    # 步骤3：对密集y值应用高斯滤波
    y_smoothed = gaussian_filter1d(y_dense, sigma=sigma)
    
    return x_new, y_smoothed



timep = (Years>=2020)&(Years<=2040)

colors = ['#222222','#FCEC8D', '#FF9830', '#B51209']
    
tit  = ['(d)','(e)','(f)']
fig,axs = plt.subplots(nrows=1, ncols=3,figsize=(16,4),dpi=300)
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 12,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Arial'],#宋体
            'axes.unicode_minus': False ,# 处理负号，即-号
            "figure.facecolor": (1,1,1,1)
         }
rcParams.update(config)  
xl  = ['CDHD','CDHF','CDHI']
va = ['d','f','i']

Data = getData(DATA,'MME')
for a in  range(3): 
    k = 0
    lines = []
    fills = []                                                                                    
    for s in ['historical','SSP1','SSP3','SSP5']:
        xf,ub,lb,ym = getfill(DATA,s,va[a],timep)
        if s == 'historical':
                x = np.squeeze(Data[s][va[a]][-1])
                #x0 = Data[s][va[a]][-1]
                #x1 = x0.copy()
                #x1[x1>0] = 1
                #x1[x1!=1] = 0
                #x = x0*x1
                #y0 = Data[s]['pop_baby'][-1]
                #y = y0*x1
                y = np.squeeze(Data[s]['pop_baby'][-1])
                
                
        else:
            #x0 = np.nanmean(Data[s][va[a]][(Years>=2020)&(Years<=2040)],axis=0)
            #x1 = x0.copy()
            #x1[x1>0] = 1
            #x1[x1!=1] = 0
            #x = x0*x1
            x = np.squeeze(np.nanmean(Data[s][va[a]][timep],axis=0))
            #y0 = np.nanmean(Data[s]['pop_baby'][(Years>=2020)&(Years<=2040)],axis=0)
            #y = y0*x1
            y = np.squeeze(np.nanmean(Data[s]['pop_baby'][timep],axis=0))
        #y = y[x==x]
        x = x[y==y]
        y = y[y==y]
         
        d1 = pd.DataFrame({'x':x,'y':y })
        d2 = d1.sort_values(by='x')
        
        y2 = [ ]
        yy = 0
        
        for i in range(len(x)):
            if np.isnan(d2['y'][i]):
                yy = yy+0
            else:
                yy = yy+d2['y'][i]
            y2.append(yy)
        
        
        xp,yp = smooth_line(d2['x'], np.array(y2), new_points=100, sigma = 3.5)
        line, = axs[a].plot(xf,ym/1e9,color = colors[k])
         
        f = axs[a].fill_between(xf,ub/1e9,lb/1e9,color = colors[k],alpha = 0.3,edgecolor = 'none')
        lines.append(line)
        fills.append(f)
        k+=1
        print(s+' '+str(np.nanmax(ym[-1]/1e9)))
    #axs[a].vlines(linex, ymin=axs[a].get_ylim()[0], ymax=axs[a].get_ylim()[1],color = 'black',ls = '--')
    #axs[a].legend(['historical','SSP1','SSP3','SSP5'])
    #axs[a].set_xlim([axs[a].get_xlim()[0],axs[a].get_xlim()[1]])
    #axs[a].set_ylim([axs[a].get_ylim()[0],axs[a].get_ylim()[1]])
    axs[a].set_xlabel(xl[a])
    axs[a].set_ylabel(' vulnerable people population,billion' )
    axs[a].text(0.01,1.02, tit[a], transform=axs[a].transAxes)
    


fig.legend(fills, ['historical','ssp126','ssp370','ssp585'],loc = 'lower center',edgecolor = 'none',ncols = 4,bbox_to_anchor=(0.5, -0.1))
fig.legend(lines, ['historical','ssp126','ssp370','ssp585'],loc = 'lower center',edgecolor = 'none',facecolor='none',ncols = 4,bbox_to_anchor=(0.5, -0.1))
#fig.subplots_adjust(left=0.15,right=0.85,top=0.75,bottom=0.1,wspace=0.4,hspace=0.01)
    
    