# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 13:10:06 2025

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
with open('../data/Exposure.pkl', 'rb') as f:
    Exposure_data = joblib.load(f)   
    f.close()

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


ph = np.nansum(PopH)

Ps1 = PopF['SSP1']
Ps3 = PopF['SSP3']
Ps5 = PopF['SSP5']
SN1 = ['historical','SSP1','SSP3','SSP5']
SN2 = ['historical','ssp126','ssp370','ssp585']

AreaC = ['东亚','中亚','西亚','北亚','南亚','东南亚']
AreaE = ['East Asia','Central Asia','Western Asia','Northern Asia','Southern Asia','Southeast Asia']
 


fy =  np.arange(2015,2101,1) 
Years = np.arange(2020,2101,5)
Data = {}

Years0 = np.arange(1995,2015,1)
Years2 = np.arange(1990,2021,1)
Years3 = np.arange(1981,2015,1)
pa = {}
po = {}
pb = {}
Pwd = {}
Pwf = {}
Pwi = {}
for c in BR_Countries['ISO']:
    ca = cg.copy()
    
    data  =  BR_Countries[BR_Countries['ISO']==c]
    cid = data['OBJECTID_1'].values[0]
    ca[ca!=cid] = np.nan
    ca[ca==cid] = 1
    cpop = []
    cbaby = []
    cold = []
    p1 = []
    p2 = []
    p3 = []
    pwd = []
    pwf = []
    pwi  = []
    c_i = []
    c_d = []
    c_f = []
    c_a = []
    
    for y in range(1995,2015):
        pop = PopH[Years2==y]
        fb = Age_group['baby_h'][y]
        fo = Age_group['old_h'][y]
        
        p1.append(np.nansum(ca*pop*(fb+fo)))
        p2.append(np.nansum(ca*pop*fb))
        p3.append(np.nansum(ca*pop*fo))
        pwd.append(np.nansum(ca*pop*CDHD1s[Years3==y,:,:])/np.nansum(ca*pop))
        pwf.append(np.nansum(ca*pop*CDHF1s[Years3==y,:,:])/np.nansum(ca*pop))
        pwi.append(np.nansum(ca*pop*CDHI1s[Years3==y,:,:])/np.nansum(ca*pop))

    pa[c] = np.array(p1)
    pb[c] = np.array(p2)
    po[c] = np.array(p3)
    Pwd[c] = np.array(pwd)
    Pwf[c] = np.array(pwf)
    Pwi[c] = np.array(pwi)
pa =  pd.DataFrame(pa)
pb =  pd.DataFrame(pb)
po =  pd.DataFrame(po)


Pwd = pd.DataFrame(Pwd)
Pwf = pd.DataFrame(Pwf)
Pwi = pd.DataFrame(Pwi)
Data['historical'] = {'d':Pwd,'f':Pwf,'i':Pwi,'p':pa}



for s in range(1,4):
    pa = {}
    po = {}
    pb = {}
    Pwd = {}
    Pwf = {}
    Pwi = {}
    for c in BR_Countries['ISO']:
        ca = cg.copy()
        
        data  =  BR_Countries[BR_Countries['ISO']==c]
        cid = data['OBJECTID_1'].values[0]
        ca[cg!=cid] = np.nan
        ca[cg==cid] = 1
        cpop = []
        cbaby = []
        cold = []
        p1 = []
        p2 = []
        p3 = []
        pwd = []
        pwf = []
        pwi  = []
        c_i = []
        c_d = []
        c_f = []
        c_a = []
        for i in range(17):
            pop = PopF[SN1[s]][str(Years[i])]
            fb = Age_group['baby_f'][SN1[s]][Years[i]]
            fo = Age_group['old_f'][SN1[s]][Years[i]]
            
            p1.append(np.nansum(ca*pop*(fb+fo)))
            p2.append(np.nansum(ca*pop*fb))
            p3.append(np.nansum(ca*pop*fo))
            pwd.append(np.nansum(ca*pop*CDHD2s[SN2[s]][fy==Years[i],:,:])/np.nansum(ca*pop))
            pwf.append(np.nansum(ca*pop*CDHF2s[SN2[s]][fy==Years[i],:,:])/np.nansum(ca*pop))
            pwi.append(np.nansum(ca*pop*CDHI2s[SN2[s]][fy==Years[i],:,:])/np.nansum(ca*pop))
    
        pa[c] = np.array(p1)
        pb[c] = np.array(p2)
        po[c] = np.array(p3)
        Pwd[c] = np.array(pwd)
        Pwf[c] = np.array(pwf)
        Pwi[c] = np.array(pwi)
    pa =  pd.DataFrame(pa)
    pb =  pd.DataFrame(pb)
    po =  pd.DataFrame(po)
    
    
    Pwd = pd.DataFrame(Pwd)
    Pwf = pd.DataFrame(Pwf)
    Pwi = pd.DataFrame(Pwi)
    Data[SN1[s]] = {'d':Pwd,'f':Pwf,'i':Pwi,'p':pa}


X = []
Y = []
L = []
S = []
C = []
cs = np.array([
    [51, 153, 204],   
    [204, 51, 51],  
    [204, 153, 51],  
    [76, 76, 176],   
    [255, 191, 0],
    [51, 187, 102]
])/255
col = pd.DataFrame({
   AreaC[0]:cs[0],   
   AreaC[1]:cs[1],  
   AreaC[2]:cs[2],   
   AreaC[3]:cs[3],   
   AreaC[4]:cs[4],   
   AreaC[5]:cs[5]  
})
for c in BR_Countries['ISO']:
    ps = np.nanmean(Data['SSP5']['p'][c][(Years>=2080)&(Years<=2100)])
    pb = np.nanmean(Data['historical']['p'][c])

    hs = np.nanmean(Data['SSP5']['d'][c][(Years>=2080)&(Years<=2100)])
    hb = np.nanmean(Data['historical']['d'][c])

    es = ps*hs
    if pb<1:
        dp = 1
    else:
        dp  = (ps-pb)/pb
    if hb<1:
        dh = (hs-hb)/1
    else:
        dh = (hs-hb)/hb

    X.append(dh)
    Y.append(dp)
    L.append(c)
    S.append(es)
    C.append(np.array(col[BR_Countries[BR_Countries['ISO']==c]['AREA'].values[0]]))
X = np.array(X)
Y = np.array(Y)
S = np.array(S)
L = np.array(L)
C = np.array(C)
def func(x,a=1):
    y = (a-1-x)/(x+1)
    return y


x  = np.linspace(0, 250,1000)

fig,axes = plt.subplots( figsize=(5, 5),dpi = 300)
plt.rcParams.update({
    'axes.grid': False,
    'grid.alpha': 0.3,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'font.size': 13,
    "figure.facecolor": (1,1,1,1)

})
for a in [2,16,64,126,256]:
    y = func(x,a = a)
    axes.plot(x,y,c = '#000000',ls = '--',alpha=0.5)
    axes.text(np.sqrt(a)-1,func(np.sqrt(a)-1,a = a),str(a)+'×')
axes.scatter(X, Y,np.power(S/1e6,0.8),c = C,alpha=0.5)
axes.set_xlim([1e-2,250])
axes.set_ylim([1e-2,250])
axes.plot([0, 250], [0,250], label='对角线', color='#7F0B0B')
handles = []
for i in range(6):
    # 用虚拟散点（大小为 0 等不可见）来占位做图例
    dummy_scatter = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col[AreaC[i]], markersize=12, label=AreaE[i],alpha = 0.5)
    handles.append(dummy_scatter)

axes.legend(handles=handles, labels=AreaE, bbox_to_anchor=(-0.2, -0.3), ncol =3,loc='lower left',edgecolor = 'none',fontsize = 12)
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_xlabel(r'$\Delta$CDHD/CDHD$_{Baseline}$')
axes.set_ylabel(r'$\Delta$VP/VP$_{Baseline}$')
axes.text(0.01,1.04, '(e)', transform=axes.transAxes)
plt.subplots_adjust(
    left=0.08,    # 左间距：画布左边缘到Axes左边框占画布宽度的8%
    right=0.95,   # 右间距：画布右边缘到Axes右边框占画布宽度的5%（1-0.95=5%）
    bottom=0.08,  # 下间距：画布下边缘到Axes下边框占画布高度的8%
    top=0.92      # 上间距：画布上边缘到Axes上边框占画布高度的8%（1-0.92=8%）
)
