# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:38:43 2025

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
from scipy.stats import gaussian_kde
from getgridC import getCG
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


with open('../data/Model_data.pkl', 'rb') as f:
    Model_Data = joblib.load(f)   
    f.close()

cg = getCG()

 

with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']
lat = maskdata['lat']
lon = maskdata['lon']

hy = np.arange(1981,2015,1)
fy  = np.arange(2015,2101,1)
SAPEI1s = Model_Data['historical_sapei']  
STI1s = Model_Data['historical_sti']  
SCDHI1s = Model_Data['historical_scdhi']  
  
SAPEI2s = Model_Data['future_sapei'] 
STI2s = Model_Data['future_sti']  
SCDHI2s = Model_Data['future_scdhi'] 


x1 = np.nanmean( np.nanmean( SAPEI1s['historical'][hy>1995,:,:],axis=1),axis=1)
y1 = np.nanmean(np.nanmean(STI1s['historical'][hy>1995,:,:],axis=1),axis=1)




X = {}
Y = {}

SN = ['ssp126','ssp370','ssp585']
Pd = [[2021,2041],[2051,2071],[2081,2101 ]]
for s in SN:
    X2 = []
    Y2 = []
    for p in Pd:
        x1 = np.nanmean( np.nanmean( SAPEI2s[s][(fy>=p[0])&(fy<p[1]),:,:],axis=1),axis=1)
        y1 = np.nanmean(np.nanmean(STI2s[s][(fy>=p[0])&(fy<p[1]),:,:],axis=1),axis=1)
        X2.append(x1)
        Y2.append(y1)
    X[s] = np.array(X2)
    Y[s] = np.array(Y2)
    



def plot_single_scatter(fig, main_gs, row, col, x, y, x_label, y_label, color,tit):
    """绘制单个带边际分布的散点图"""
    # 创建子图网格（主图+顶部边际+右侧边际）
    sub_gs = GridSpecFromSubplotSpec(
        2, 2, 
        subplot_spec=main_gs[row, col],
        width_ratios=[4, 1],  # 主图:右侧边际宽度比
        height_ratios=[1, 4], # 顶部边际:主图高度比
        wspace=0.0,
        hspace=0.0
    )
    
    # 创建三个子图区域
    ax_main = fig.add_subplot(sub_gs[1, 0])         # 主散点图
    ax_marg_x = fig.add_subplot(sub_gs[0, 0], sharex=ax_main)  # 顶部边际
    ax_marg_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_main)  # 右侧边际
    
    # 绘制主散点图
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    d1 = ax_main.scatter(x0, y0, alpha=0.8, s=50, c=color[0], edgecolors='white', linewidth=0.3)
    d2 = ax_main.scatter(x1, y1, alpha=0.8, s=50, c=color[1], edgecolors='white', linewidth=0.3)
    d3 = ax_main.scatter(x2, y2, alpha=0.8, s=50, c=color[2], edgecolors='white', linewidth=0.3)
    ax_main.set_xlabel(x_label, )
    ax_main.set_ylabel(y_label,  )
    ax_main.tick_params(axis='both', )
    #ax_main.grid(True, linestyle='--', alpha=0.3)
    
    # 绘制顶部x变量KDE
    kde_x = gaussian_kde(x0)
    x_range = np.linspace(x.min(), x.max(), 200)
    ax_marg_x.fill_between(x_range, kde_x(x_range), color=color[0], alpha=0.8)
    
   
    kde_x1 = gaussian_kde(x1)
    x1_range = np.linspace(x.min(), x.max(), 200)
    ax_marg_x.fill_between(x1_range, kde_x1(x1_range), color=color[1], alpha=0.8)
    
    
    kde_x2 = gaussian_kde(x2)
    x2_range = np.linspace(x.min(), x.max(), 200)
    ax_marg_x.fill_between(x2_range, kde_x2(x2_range), color=color[2], alpha=0.8)
    #ax_marg_x.set_ylim(0, kde_x(x_range).max() * 1.1)
    ax_marg_x.set_yticks([])
    ax_marg_x.spines['right'].set_visible(False)
    ax_marg_x.spines['top'].set_visible(False)
    ax_marg_x.spines['left'].set_visible(False)
    ax_marg_x.spines['bottom'].set_visible(False)
    
    # 绘制右侧y变量KDE
    kde_y = gaussian_kde(y0)
    y_range = np.linspace(y.min(), y.max(), 200)
    ax_marg_y.fill_betweenx(y_range, kde_y(y_range), color=color[0], alpha=0.8)
    
    
    kde_y1 = gaussian_kde(y1)
    y1_range = np.linspace(y.min(), y.max(), 200)
    ax_marg_y.fill_betweenx(y1_range, kde_y1(y1_range), color=color[1], alpha=0.8)
    
    
    kde_y2 = gaussian_kde(y2)
    y2_range = np.linspace(y.min(), y.max(), 200)
    ax_marg_y.fill_betweenx(y2_range, kde_y2(y2_range), color=color[2], alpha=0.8)
    
    #ax_marg_y.set_xlim(0, kde_y2(y2_range).max() * 1.1)
    ax_marg_y.set_xticks([])
    ax_marg_y.spines['right'].set_visible(False)
    ax_marg_y.spines['top'].set_visible(False)
    #ax_marg_y.spines['left'].set_visible(False)
    ax_marg_y.spines['bottom'].set_visible(False)
    # 隐藏共享轴的刻度标签
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_marg_x.text(0,0.99, tit, transform=ax_marg_x.transAxes)
    return [d1,d2,d3]










fig = plt.figure(figsize=(16, 4),dpi=300)
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 14,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Arial'],#宋体
            'axes.unicode_minus': False ,# 处理负号，即-号
            "figure.facecolor": (1,1,1,1)
         }
rcParams.update(config)  
main_gs = GridSpec(1, 3, wspace=0.4, hspace=0.5)

colors = [ '#F3E065', '#DFAD5A', '#B15726']
tit  = ['(d) ssp126','(e) ssp370','(f) ssp585']
# 绘制6个子图
for i in range(3):
    
    
        x1 = Y[SN[i]]
        y1 = X[SN[i]]
        x_label = 'STI'
        y_label = 'SAPEI'
        color = colors
        D =  plot_single_scatter(fig, main_gs,0,i, x1, y1, x_label, y_label, color,tit[i])
        
plt.legend(D, ['Early-Century','Mid-Century','Late-Century'],loc = 'lower center',edgecolor = 'none',ncols = 3,bbox_to_anchor=(-9, -0.4))
 
plt.show()




