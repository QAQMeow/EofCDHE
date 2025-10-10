
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:06:35 2025

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
# with open('../data/Exposure.pkl', 'rb') as f:
#     Exposure_data = joblib.load(f)   
#     f.close()

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
AreaE = ['East Asia','Cental Asia','Western Asia','Southern Asia','Southeast Asia']
 


fy =  np.arange(2015,2101,1) 
Years = np.arange(2020,2101,5)
Data = {}


Years2 = np.arange(1990,2021,1)
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
    Years3 = np.arange(1981,2015,1)
    for y in range(1995,2015):
        pop = np.squeeze( PopH[Years2==y])
        
        fb = Age_group['baby_h'][y]
        fo = Age_group['old_h'][y]
        
        p1.append(np.nansum(ca*pop))
        p2.append(np.nansum(ca*pop*fb))
        p3.append(np.nansum(ca*pop*fo))
        pwd.append(np.nansum(ca*pop*CDHD1s[Years3==y,:,:])/np.nansum(ca*pop))
        pwf.append(np.nansum(ca*pop*CDHF1s[Years3==y,:,:])/np.nansum(ca*pop))
        pwi.append(np.nansum(ca*pop*CDHI1s[Years3==y,:,:])/np.nansum(ca*pop))
        print(str(cid)+' '+c+' '+str(y))
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
Data['historical'] = {'d':Pwd,'f':Pwf,'i':Pwi,'p':pb,'pt':pa}



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
        cid = data['FID'].values[0]
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
        for i in range(17):
            pop = np.squeeze( PopF[SN1[s]][str(Years[i])])
             
            fb = Age_group['baby_f'][SN1[s]][Years[i]]
            fo = Age_group['old_f'][SN1[s]][Years[i]]
            
            p1.append(np.nansum(ca*pop))
            p2.append(np.nansum(ca*pop*fb))
            p3.append(np.nansum(ca*pop*fo))
            pwd.append(np.nansum(ca*pop*CDHD2s[SN2[s]][fy==Years[i],:,:])/np.nansum(ca*pop))
            pwf.append(np.nansum(ca*pop*CDHF2s[SN2[s]][fy==Years[i],:,:])/np.nansum(ca*pop))
            pwi.append(np.nansum(ca*pop*CDHI2s[SN2[s]][fy==Years[i],:,:])/np.nansum(ca*pop))
            print(str(cid)+' '+SN1[s]+' '+c+' '+str(Years[i]))
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
    
    Data[SN1[s]] = {'d':Pwd,'f':Pwf,'i':Pwi,'p':pb,'pt':pa}












tit  = ['(a)','(b)','(c)']
AreaE2 = ['EA','CA','WA','NA','SA','SEA']
va = ['d','f','i']
xl = ['population weighted CDHD','population weighted CDHF','population weighted CDHI']
colors = ['#222222','#FCEC8D', '#FF9830', '#B51209']
fig, axs = plt.subplots(1, 3,figsize=(16,4),dpi=300)
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
for k in range(3):
    for i in range(4):
        data = []
        ppp = Data[SN1[i]][va[k]]
        for a  in AreaC:
            if SN1[i] == 'historical':
                d = np.nanmean(ppp[BR_Countries[BR_Countries['AREA']==a].ISO.values].values,axis=0)
            else:
                d = np.nanmean(ppp[BR_Countries[BR_Countries['AREA']==a].ISO.values].values[(Years>=2050)&(Years<=2070),:],axis=0)
            d = d[d==d]
            data.append(d)
        pos = np.arange(len(data))
        bplot = axs[k].boxplot(data,positions =3*pos+i*0.6 , vert=False,patch_artist=True,boxprops={'facecolor': colors[i]},
        
                showmeans=True,
            meanprops={'marker':'D',
                       'markerfacecolor':'none', 
                       'markeredgecolor':'#34F6F6',
                       'markersize':'5'})
        for daa in data:
            
            print([k,np.nanmean(daa),np.nanmedian(daa),SN1[i]])
    axs[k].set_yticks(3*pos+0.8,AreaE2)
    handles = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black',  ) for color in colors]
    axs[k].text(0.01,1.02, tit[k], transform=axs[k].transAxes)
    axs[k].set_xlabel(xl[k])

fig.legend(handles, SN2,  fontsize = 12,loc = 'lower center',edgecolor = 'none',facecolor='none',ncols = 4,bbox_to_anchor=(0.5, -0.1))








# def plotArea(Years,data):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from matplotlib.patches import Rectangle
    
#     # 设置全局样式，确保所有子图风格统一
    
#     config = {
#                 "font.family": 'serif',
#                 "font.size": 5,# 相当于小四大小
#                 "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
#                 "font.serif": ['Arial'],#宋体
#                 'axes.unicode_minus': False ,# 处理负号，即-号
#                 "figure.facecolor": (1,1,1,1)
#              }
#     rcParams.update(config) 
    
#     # 创建4×4子图
#     fig, axes = plt.subplots(8, 8, figsize=(16, 15))
#     fig.subplots_adjust(
#         top=0.92,    # 整体顶部边距
#         bottom=0.08, # 整体底部边距
#         left=0.08,   # 整体左侧左边距
#         right=0.92,  # 整体右边距
#         hspace=0.45,  # 子图之间的垂直间距
#         wspace=0.55   # 子图之间的水平间距
#     )
    
    
#     # 生成不同的数据模式
    
#     # 为每个子图绘制内容
#     c = 0
#     for i in range(8):
#         for j in range(8):
#             ax = axes[i, j]
            
#             x = Years
#             # 生成并绘制数据
#             y = po[list(po.keys())[c]]/1e6
#             y[y==0] = np.nan
#             ax.plot(x, y, color='#FB7770', linewidth=1.5)
            
#             # 设置坐标轴范围
#             ax.set_xlim(2020, 2100)
#             # ax.set_ylim(-2, 2)
            
#             # 设置子图标题
          
            
#             # 获取当前子图的位置
#             ax_pos = ax.get_position()
            
#             # 添加顶部紧贴矩形
#             rect_top = Rectangle(
#                 (ax_pos.x0, ax_pos.y0 + ax_pos.height),  # 与子图顶部紧贴
#                 ax_pos.width,                            # 与子图同宽
#                 0.015,                                   # 矩形高度（适应子图密度）
#                 facecolor='none',
#                 edgecolor='#2C3E50',
#                 linewidth=1,
#                 transform=fig.transFigure,
#                 zorder=10
#             )
#             fig.patches.append(rect_top)
            
#             # 在矩形中添加简短标签
            
#             ax.set_title(list(pa.keys())[c],pad = 4)
#             c+=1
#     # 添加整体标题
#     fig.suptitle('SSP585 old(65+) population', fontsize=16, y=0.96)
    
#     plt.show()

# from matplotlib.patches import Rectangle
# import matplotlib.pyplot as plt
# import numpy as np

# # 定义区域和情景
# regions =  ['Europe','East Asia','Cental Asia','Western Asia','Southern Asia','Southeast Asia']
# scenarios = ['SSP126',  'SSP370', 'SSP585']
# # 全球作为单独的子图，这里也列出来方便统一处理
# all_regions = regions + ['BR']  

 
 
# import matplotlib.gridspec as gridspec
# fig = plt.figure(figsize=(15, 4))
# fig.subplots_adjust(
#     top=0.92,    # 整体顶部边距
#     bottom=0.08, # 整体底部边距
#     left=0.08,   # 整体左侧左边距
#     right=0.92,  # 整体右边距
#     hspace=0.45,  # 子图之间的垂直间距
#     wspace=0.55   # 子图之间的水平间距
# )
# gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1])

# # 绘制区域子图（2行3列）
# axes_region = []
# for i in range(6):
#     row = i // 3
#     col = i % 3
#     ax = plt.subplot(gs[row, col])
#     axes_region.append(ax)
#     region = regions[i]
#     ax.set_title(region)
#     # 绘制分组柱状图，不同颜色代表不同组成部分（这里简单分三种颜色示例，你可按实际数据调整）
#     bar_width = 0.2
#     x = np.arange(2020,2100,5)
#     # 假设数据有三个组成部分，这里用随机数模拟占比，实际替换为真实分类数据
#     ax.plot(2050,1)
#     #ax.set_xticks(x)
#     ax.set_xlim([2020,2100])
#     ax.set_ylabel('Y')
#     ax_pos = ax.get_position()

#     # 添加顶部紧贴矩形
#     rect_top = Rectangle(
#         (ax_pos.x0, ax_pos.y0 + ax_pos.height),  # 与子图顶部紧贴
#         ax_pos.width,                            # 与子图同宽
#         0.07,                                   # 矩形高度（适应子图密度）
#         facecolor='none',
#         edgecolor='#2C3E50',
#         linewidth=1,
#         transform=fig.transFigure,
#         zorder=10
#     )
#     fig.patches.append(rect_top)

# # 绘制全球子图（第二行第四列位置）
# ax_global = plt.subplot(gs[0:2, 3])
# ax_global.set_title('BR')
# ax_pos = ax_global.get_position()

# # 添加顶部紧贴矩形
# rect_top = Rectangle(
#     (ax_pos.x0, ax_pos.y0 + ax_pos.height),  # 与子图顶部紧贴
#     ax_pos.width,                            # 与子图同宽
#     0.07,                                   # 矩形高度（适应子图密度）
#     facecolor='none',
#     edgecolor='#2C3E50',
#     linewidth=1,
#     transform=fig.transFigure,
#     zorder=10
# )
# fig.patches.append(rect_top)
# x_global = np.arange(2020,2100,5)
# # 同样假设数据有三个组成部分，用随机数模拟占比示例

# ax_global.plot(2050,1)
# ax_global.set_xlim([2020,2100])

# ax_global.set_ylabel('Y')

# # 调整子图间距
# #plt.tight_layout()
# plt.show()

