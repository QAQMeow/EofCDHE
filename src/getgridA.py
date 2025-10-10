# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 17:07:45 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
from getgridC import getCG
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def getAG():
    BR_Countries = pd.read_excel('../data/Asia_c.xlsx')
    cg = getCG()
    ca = cg.copy()
    Ar = ['东亚','中亚','西亚','北亚','南亚','东南亚']
    k = 0
    for a in Ar:
        CL = BR_Countries[BR_Countries['AREA']==a]
        for c in CL['ISO']:
            
             
            
            data  =  CL[CL['ISO']==c]
            cid = data['OBJECTID_1'].values[0]
            ca[cg==cid] = k
        k+=1
   
    return ca