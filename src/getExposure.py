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
AreaE = ['East Asia','Cental Asia','Western Asia','Northern Asia','Southern Asia','Southeast Asia']
 


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
        
        p1.append(np.nansum(ca*pop*(fb+fo)*CDHD1s[Years3==y,:,:]))
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
Data['historical'] = {'d':Pwd,'f':Pwf,'i':Pwi,'E':pa}



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
            
            p1.append(np.nansum(ca*pop*(fb+fo)*CDHD2s[SN2[s]][fy==Years[i],:,:]))
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
    Data[SN1[s]] = {'d':Pwd,'b':pb,'o':po,'E':pa}

 


d1 = Data['SSP1']['E'][fy==Years[i],:]



