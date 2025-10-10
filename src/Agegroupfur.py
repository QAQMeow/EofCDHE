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
import warnings
warnings.filterwarnings('ignore')

Asia_Countries = pd.read_excel('../data/Asia_c.xlsx',engine='openpyxl')
data_total = pd.read_excel('../data/popagestruct/total.xlsx',engine='openpyxl')
data_m = pd.read_excel('../data/popagestruct/Male_0-4.xlsx',engine='openpyxl')
data_f = pd.read_excel('../data/popagestruct/Female_0-4.xlsx',engine='openpyxl')
ASCI = Asia_Countries['ISO']

data_ageh = pd.read_excel('../data/Age_group.xlsx',engine='openpyxl')

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

AGHB = {}
AGHO = {}
for i in ASCI:
    dh = data_ageh[(data_ageh['Code']==i)&(data_ageh['Year']>=1990)&(data_ageh['Year']<=2020)].sort_values(by='Year')
    dho = dh['Age:65+'].values
   
    dhb = dh['Age:0-4'].values
    
    AGHB[i] = dhb
    AGHO[i] = dho
    
    
    
oldg = ['65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+']

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



Age_grope_f = {'Baby':BABY,'Old':OLD}
Age_grope_h = {'Baby':AGHB,'Old':AGHO}
with open('../data/Age_groupF.pkl', 'wb') as f:
 	pickle.dump(Age_grope_f, f)
     
with open('../data/Age_groupH.pkl', 'wb') as f:
    pickle.dump(Age_grope_h, f)
     
     