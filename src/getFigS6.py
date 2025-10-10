# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 12:24:25 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 
 
import os
import numpy as np
import joblib
import pandas as pd
import netCDF4 as nc
from getgridC import getCG
from getgridA import getAG
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

BR_Countries = pd.read_excel('../data/Asia_c.xlsx')

with open('../data/Model_CDHEs_data.pkl', 'rb') as f:
    CDHEs_data = joblib.load(f)   
    f.close()

cg = getCG()
ag = getAG()
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


SN1 = ['historical','SSP1','SSP3','SSP5']
SN2 = ['historical','ssp126','ssp370','ssp585']

AreaC = ['东亚','中亚','西亚','北亚','南亚','东南亚']
AreaE  =  ['East Asia','Cental Asia','West Asia','North Asia','South Asia','Southeast Asia']
 
 


fy =  np.arange(2015,2101,1) 
Years = np.arange(2020,2101,5)
Data = {}


Years2 = np.arange(1990,2021,1)
 

cpop = []
cbaby = []
cold = []
p1 = []
p2 = []
p3 = []
p4 = []
e = []
f = []
Years3 = np.arange(1981,2015,1)
for y in range(1995,2015):
    pop = PopH[Years2==y]
    fb = Age_group['baby_h'][y]
    fo = Age_group['old_h'][y]
        
    p1.append(pop)
    p2.append(pop*fb)
    p3.append(pop*fo)
    p4.append(fb+fo)
    f.append(CDHD1s[Years3==y])
    e.append(pop*(fb+fo)*CDHD1s[Years3==y])
pa = np.array(p1)
pb = np.array(p2)
po = np.array(p3)
pv = np.array(p4)
pe = np.array(e)
     

Data['historical'] ={'pop_all':np.squeeze(pa),'pop_old':np.squeeze(po),'pop_baby':np.squeeze(pb),'pop_v':np.squeeze(pv),'e':np.squeeze(pe),'h':np.squeeze(np.array(f))}


for s in range(1,4):
   
    cpop = []
    cbaby = []
    cold = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    e = []
    f = []
    for i in range(17):
        pop = PopF[SN1[s]][str(Years[i])]
        fb = Age_group['baby_f'][SN1[s]][Years[i]]
        fo = Age_group['old_f'][SN1[s]][Years[i]]
        
        p1.append(pop)
        p2.append(pop*fb)
        p3.append(pop*fo)
        p4.append(pop*(fb+fo))
        f.append(CDHD2s[SN2[s]][fy==Years[i]])
        e.append(pop*(fb+fo)*CDHD2s[SN2[s]][fy==Years[i]])
    pa = np.array(p1)
    pb = np.array(p2)
    po = np.array(p3)
    pe = np.array(e)
    pv = np.array(p4)    
    
    Data[SN1[s]] = {'pop_all':np.squeeze(pa),'pop_old':np.squeeze(po),'pop_baby':np.squeeze(pb),'pop_v':np.squeeze(pv),'e':np.squeeze(pe),'h':np.squeeze(np.array(f))}


A = {}
Ar = ['东亚','中亚','西亚','北亚','南亚','东南亚']
regions =  ['East Asia','Cental Asia','West Asia','North Asia','South Asia','Southeast Asia']
 
for s in ['SSP1','SSP3','SSP5']:
   
    PBA = Data[s]['pop_all']
    PBO = Data[s]['pop_old']
    PBB = Data[s]['pop_baby']
    PBV = Data[s]['pop_v']
    

    B = {}
    for a in range(7):
        if a<6:
            ca = ag.copy()
            
            ca[ca!=a] = np.nan
            ca[ca==a] = 1
            
            
            pba = np.nansum(np.nansum(ca*PBA,axis=1 ),axis=1 )
            pbb = np.nansum(np.nansum(ca*PBB,axis=1 ),axis=1 )
            pbo = np.nansum(np.nansum(ca*PBO,axis=1 ),axis=1 )
            pbv = np.nansum(np.nansum(ca*PBV,axis=1 ),axis=1 )
            B[regions[a]] = np.array([pba,pbb,pbo,pbv])
        else:
            ca = ag.copy()
            
             
             
            ca[ca<0] = np.nan
            ca[ca>=0] = 1
            pba = np.nansum(np.nansum(ca*PBA,axis=1 ),axis=1 )
            pbb = np.nansum(np.nansum(ca*PBB,axis=1 ),axis=1 )
            pbo = np.nansum(np.nansum(ca*PBO,axis=1 ),axis=1 )
            pbv = np.nansum(np.nansum(ca*PBV,axis=1 ),axis=1 )
             
            B['Asia'] = np.array([pba,pbb,pbo,pbv])
    A[s] = B
    
    

 


# 定义区域和情景
Ar = ['东亚','中亚','西亚','北亚','南亚','东南亚']
regions =  ['East Asia','Cental Asia','West Asia','North Asia','South Asia','Southeast Asia']
 
scenarios = ['SSP126',  'SSP370', 'SSP585']
 
all_regions = regions + ['Asia']  

 
# 创建画布和子图网格
# 子图布局：6个区域子图（2行3列） + 1个全球子图，共7个子图，这里用 gridspec 更灵活控制布局
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
fig = plt.figure(figsize=(15, 4))
plt.rcParams.update({
    'axes.grid': False,
    'grid.alpha': 0.3,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'font.size': 13,
    "figure.facecolor": (1,1,1,1)
 
})


fig.subplots_adjust(
    top=0.92,    # 整体顶部边距
    bottom=0.08, # 整体底部边距
    left=0.08,   # 整体左侧左边距
    right=0.92,  # 整体右边距
    hspace=0.45,  # 子图之间的垂直间距
    wspace=0.55   # 子图之间的水平间距
)
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1])

# 绘制区域子图（2行3列）
axes_region = []
colors = ['#FCEC8D', '#FF9830', '#B51209', ]

D = {}
# 0 all 1 baby 2 old 3 v
z = 3
for i in range(6):
    row = i // 3
    col = i % 3
    ax = plt.subplot(gs[row, col])
    axes_region.append(ax)
    region = regions[i]
    ax.set_title(region)
    # 绘制分组柱状图，不同颜色代表不同组成部分（这里简单分三种颜色示例，你可按实际数据调整）
    bar_width = 0.2
    
    d1 = A['SSP1'][all_regions[i]][z]/1e6
    d2 = A['SSP3'][all_regions[i]][z]/1e6
    d3 = A['SSP5'][all_regions[i]][z]/1e6
    ssp_cat  = ['126', '370', '585']
    plt.plot(Years,d1,color = colors[0])
    plt.plot(Years,d2,color = colors[1])
    plt.plot(Years,d3,color = colors[2])
    print(Ar[i])
    
    
    ax.set_xlim([2020,2100])
    #ax.set_ylabel('Y')
    ax_pos = ax.get_position()
    ax.set_ylabel('population,million')
    # 添加顶部紧贴矩形
    rect_top = Rectangle(
        (ax_pos.x0, ax_pos.y0 + ax_pos.height),  # 与子图顶部紧贴
        ax_pos.width,                            # 与子图同宽
        0.07,                                   # 矩形高度（适应子图密度）
        facecolor='none',
        edgecolor='#2C3E50',
        linewidth=1,
        transform=fig.transFigure,
        zorder=10
    )
    fig.patches.append(rect_top)

# 绘制全球子图（第二行第四列位置）
ax_global = plt.subplot(gs[0:2, 3])
ax_global.set_title('Asia')
ax_pos = ax_global.get_position()

# 添加顶部紧贴矩形
rect_top = Rectangle(
    (ax_pos.x0, ax_pos.y0 + ax_pos.height),  # 与子图顶部紧贴
    ax_pos.width,                            # 与子图同宽
    0.07,                                   # 矩形高度（适应子图密度）
    facecolor='none',
    edgecolor='#2C3E50',
    linewidth=1,
    transform=fig.transFigure,
    zorder=10
)
fig.patches.append(rect_top)


d1 = A['SSP1'][all_regions[6]][z]/1e6
d2 = A['SSP3'][all_regions[6]][z]/1e6
d3 = A['SSP5'][all_regions[6]][z]/1e6
ssp_cat  = ['126', '370', '585']
p1, = plt.plot(Years,d1,color = colors[0])
p2, = plt.plot(Years,d2,color = colors[1])
p3, = plt.plot(Years,d3,color = colors[2])
 

ax_global.set_ylabel('population,million')
fig.legend(
    [p1,p2,p3], ['ssp126','ssp370','ssp585'],
    loc='lower center',  # 位置：底部中央
    ncol=3,  # 图例列数
    bbox_to_anchor=(0.5, -0.3),  # 微调位置（相对于图形）
    fontsize=15,
    borderaxespad = 4,
    edgecolor = 'none'
)
plt.show()



