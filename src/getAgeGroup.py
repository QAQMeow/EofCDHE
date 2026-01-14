# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:40:55 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
import joblib
import pickle
import numpy as np
import pandas as pd
from getASCMask import getASCMask

with open('../data/Age_groupH.pkl', 'rb') as f:  # 读取pickle文件
    Age_groupH = joblib.load(f)
    f.close()

with open('../data/Age_groupF.pkl', 'rb') as f:  # 读取pickle文件
    Age_groupF = joblib.load(f)
    f.close()
    
fr ='../data/Asia_c.xlsx'
data = pd.read_excel(fr)
Scenarios = ['SSP1','SSP2','SSP3','SSP4','SSP5']
CMask = getASCMask()


def getAGHM(agg):
    AGMaskH = {}
    for y in range(31):
        m = np.nan*np.zeros_like(CMask)
        for soc in data['ISO']:
            ids = data[data['ISO']==soc].OBJECTID_1
            frac = Age_groupH[agg][soc]
            f=frac[y]
            m[CMask==ids.values[0]] = f
        
        AGMaskH[1990+y] = m
    return AGMaskH

def getAGFM(agg):
    AGMaskF = {}
    for s in Scenarios:
        M = {}
        for y in range(2020,2101,5):
            m = np.nan*np.zeros_like(CMask)
            for soc in data['ISO']:
                ids = data[data['ISO']==soc].OBJECTID_1
                frac = Age_groupF[agg][soc]
                f=frac[frac['Scenario']==s][y]
                m[CMask==ids.values[0]] = f
            M[y] = m
        AGMaskF[s] = M

    return AGMaskF



OldMaskH = getAGHM('Old')
BabyMaskH = getAGHM('Baby')
StudentMaskH = getAGHM('Student')
WorkerMaskH = getAGHM('Worker')


OldMaskF = getAGFM('Old')
BabyMaskF = getAGFM('Baby')
StudentMaskF = getAGFM('Student')
WorkerMaskF = getAGFM('Worker')


OldMaskF = {}
for s in Scenarios:
    M = {}
    for y in range(2020,2101,5):
        m = np.nan*np.zeros_like(CMask)
        for soc in data['ISO']:
            ids = data[data['ISO']==soc].OBJECTID_1
            frac = Age_groupF['Old'][soc]
            f=frac[frac['Scenario']==s][y]
            m[CMask==ids.values[0]] = f
        M[y] = m
    OldMaskF[s] = M
    

BabyMaskF = {}
for s in Scenarios:
    M = {}
    for y in range(2020,2101,5):
        m = np.nan*np.zeros_like(CMask)
        for soc in data['ISO']:
            ids = data[data['ISO']==soc].OBJECTID_1
            frac = Age_groupF['Baby'][soc]
            f=frac[frac['Scenario']==s][y]
            m[CMask==ids.values[0]] = f
        M[y] = m
    BabyMaskF[s] = M


OldMaskH = {}
 
for y in range(31):
    m = np.nan*np.zeros_like(CMask)
    for soc in data['ISO']:
        ids = data[data['ISO']==soc].OBJECTID_1
        frac = Age_groupH['Old'][soc]
        f=frac[y]
        m[CMask==ids.values[0]] = f
    
    OldMaskH[1990+y] = m




BabyMaskH = {}

 
for y in range(31):
    m = np.nan*np.zeros_like(CMask)
    for soc in data['ISO']:
        ids = data[data['ISO']==soc].OBJECTID_1
        frac = Age_groupH['Baby'][soc]
        f=frac[y]
        m[CMask==ids.values[0]] = f
    
    BabyMaskH[1990+y] = m



Group = {'old_f':OldMaskF,
         'worker_f':WorkerMaskF,
         'student_f':StudentMaskF,
         'baby_f':BabyMaskF,
         
         'old_h':OldMaskH,
         'worker_h':WorkerMaskH,
         'student_h':StudentMaskH,
         'baby_h':BabyMaskH,
         
         
         }
with open('../data/AgegroupGridData.pkl', 'wb') as f:
 	
    pickle.dump(Group, f)



 



