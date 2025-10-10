# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:25:20 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 

import numpy as np
from scipy import stats
from statsmodels.distributions.copula.api import FrankCopula


def interpolate_nans(data):
    """
    线性插值填补NumPy数组中的NaN值
    
    参数:
    data: 包含NaN的1D数组
    
    返回:
    插值后的数组
    """
    # 创建布尔掩码标记NaN位置
    mask = np.isnan(data)
    
    # 获取非NaN值的索引和值
    indices = np.arange(len(data))
    valid_indices = indices[~mask]
    valid_values = data[~mask]
    
    # 对NaN位置进行线性插值
    if len(valid_indices) > 0:  # 确保有有效值可供插值
        data[mask] = np.interp(indices[mask], valid_indices, valid_values)
    
    return data

def getSCDHI(sapei,sti,th,t1,t2):
    sapei = interpolate_nans(sapei)
    
    a1, b1 = t1[0],t1[1]
    s_cdf = stats.uniform.cdf(sapei, a1, b1-a1)
    sti = interpolate_nans(sti)
     
    
    a2, b2 =  t2[0],t2[1]
    t_cdf = stats.uniform.cdf(sti, a2, b2-a2)
   
    data = np.column_stack((s_cdf,t_cdf))
    copula = FrankCopula( )
    
    CP = FrankCopula(theta=th,k_dim=2)
    
    c = CP.cdf(data)
    
    p = s_cdf - c
    p[p<1e-16] = 1e-16
    scdhi = stats.norm.ppf(p,0,1)
    return scdhi
    
    
    