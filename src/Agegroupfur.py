# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:16:17 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import threading
import warnings
warnings.filterwarnings('ignore')

Asia_Countries = pd.read_excel('../data/Asia_c.xlsx',engine='openpyxl')
data_total = pd.read_excel('../data/popagestruct/total.xlsx',engine='openpyxl')
data_m = pd.read_excel('../data/popagestruct/Male_0-4.xlsx',engine='openpyxl')
data_f = pd.read_excel('../data/popagestruct/Female_0-4.xlsx',engine='openpyxl')
ASCI = Asia_Countries['ISO']

data_ageh = pd.read_excel('../data/Age_group.xlsx',engine='openpyxl')

AGFB = {} # 0-4
AGFS = {} # 5-14
AGFW = {}  # 15=64
AGFO = {}  # 65+
BABY = {}
for i in ASCI:
   
    df = data_f[data_f['Region']==i]
    dm = data_m[data_m['Region']==i]
    da = data_total[data_total['Region']==i]
    dx = (df.values[:,5:]+dm.values[:,5:])/da.values[:,5:]
    dx = pd.DataFrame(data=dx,  columns = da.columns[5:],dtype=np.float32)

    for s in [4,3,2,1,0]:
        dx.insert(loc = 0,column = da.columns[s],value = da[da.columns[s]].values)
        
    BABY[i] = dx


AGHB = {} # 0-4
AGHS = {} # 5-14
AGHW = {}  # 15=64
AGHO = {}  # 65+

for i in ASCI:
    dh = data_ageh[(data_ageh['Code']==i)&(data_ageh['Year']>=1990)&(data_ageh['Year']<=2020)].sort_values(by='Year')
    dhb = dh['Age:0-4'].values
    dhs = dh['Age:5-14'].values
    dhw = dh['Age:15-24'].values+ dh['Age:25-64'].values
    dho = dh['Age:65+'].values
    AGHB[i] = dhb
    AGHS[i] = dhs
    AGHW[i] = dhw
    AGHO[i] = dho
    


def procgetg(ag,ASCI,DS):
     
    for i in ASCI:
        da = data_total[data_total['Region']==i]
      
        for g in ag:
            data_m = pd.read_excel('../data/popagestruct/Male_'+g+'.xlsx')
            data_f = pd.read_excel('../data/popagestruct/Female_'+g+'.xlsx')
        
            df = data_f[data_f['Region']==i]
            dm = data_m[data_m['Region']==i]
            dx = df.values[:,5:]+dm.values[:,5:]
            if g == ag[0]:
                p = dx
            else:
                p = p+dx
        r = p/da.values[:,5:]
        
        r = pd.DataFrame(data=r,  columns = da.columns[5:],dtype=np.float32)

        for s in [4,3,2,1,0]:
            r.insert(loc = 0,column = da.columns[s],value = da[da.columns[s]].values)
        
        DS[i] = r
        print(i+' '+g)


babg = ['0-4']
stug = ['5-9','10-14']
wokg = ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64']
oldg = ['65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+']

AGG = [babg,stug,wokg,oldg]
DS = [AGFB,AGFS,AGFW,AGFO]
Thr = {}
for i in range(4):
    Thr[i] = threading.Thread(target=procgetg,args=(AGG[i],ASCI,DS[i]))
for i in range(4):
    Thr[i].start()
for i in range(4):
    Thr[i].join()




for i in ASCI:
    da = data_total[data_total['Region']==i]
  
    for g in stug:
        data_m = pd.read_excel('../data/popagestruct/Male_'+g+'.xlsx')
        data_f = pd.read_excel('../data/popagestruct/Female_'+g+'.xlsx')
    
        df = data_f[data_f['Region']==i]
        dm = data_m[data_m['Region']==i]
        dx = df.values[:,5:]+dm.values[:,5:]
        if g == '5-9':
            stup = dx
        else:
            stup = stup+dx
    stur = stup/da.values[:,5:]
    
    stur = pd.DataFrame(data=stur,  columns = da.columns[5:],dtype=np.float32)

    for s in [4,3,2,1,0]:
        stur.insert(loc = 0,column = da.columns[s],value = da[da.columns[s]].values)
    
    STUDENT[i] = stur
    print(i)
    
    


OLD = {}
for i in ASCI:
    da = data_total[data_total['Region']==i]
  
    for g in oldg:
        data_m = pd.read_excel('../data/popagestruct/Male_'+g+'.xlsx')
        data_f = pd.read_excel('../data/popagestruct/Female_'+g+'.xlsx')
    
        df = data_f[data_f['Region']==i]
        dm = data_m[data_m['Region']==i]
        dx = df.values[:,5:]+dm.values[:,5:]
        if g == '65-69':
            oldp = dx
        else:
            oldp = oldp+dx
    oldr = oldp/da.values[:,5:]
    
    oldr = pd.DataFrame(data=oldr,  columns = da.columns[5:],dtype=np.float32)

    for s in [4,3,2,1,0]:
        oldr.insert(loc = 0,column = da.columns[s],value = da[da.columns[s]].values)
    
    OLD[i] = oldr
    print(i)



Age_grope_f = {'Baby':AGFB,'Student':AGFS,'Worker':AGFW,'Old':AGFO}
Age_grope_h = {'Baby':AGHB,'Student':AGHS,'Worker':AGHW,'Old':AGHO}
with open('../data/Age_groupF.pkl', 'wb') as f:
 	pickle.dump(Age_grope_f, f)
     
with open('../data/Age_groupH.pkl', 'wb') as f:
    pickle.dump(Age_grope_h, f)
     
     