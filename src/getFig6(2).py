 

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

# with open('../data/Historical_ASPOP.pkl', 'rb') as f:  # 读取pickle文件
#     PopH = joblib.load(f)
#     f.close()
    
# with open('../data/Future_ASPOP.pkl', 'rb') as f:
#     PopF = joblib.load(f)
#     f.close()

with open('../data/ASPOP_URh.pkl', 'rb') as f:  # 读取pickle文件
    Pop_H = joblib.load(f)
    f.close()

with open('../data/ASPOP_URssps.pkl', 'rb') as f:
    Pop_F = joblib.load(f)
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


prt = ['Rural','Urban','Total',]
urt = prt[1]
#tit  = ['(a) baseline','(b) ssp1-2.6','(c) ssp3-7.0','(d) ssp5-8.5']
tit  = ['(e) baseline','(f) ssp1-2.6','(g) ssp3-7.0','(h) ssp5-8.5']
Ps1 = Pop_F['SSP1'][urt]
Ps3  =Pop_F['SSP3'][urt]
Ps5 = Pop_F['SSP5'][urt]
SN1 = ['historical','SSP1','SSP3','SSP5']
SN2 = ['historical','ssp126','ssp370','ssp585']

AreaC = ['东亚','中亚','西亚','北亚','南亚','东南亚']
AreaE = ['East Asia','Cental Asia','Western Asia','Southern Asia','Southeast Asia']
 


fy =  np.arange(2015,2101,1) 
Years = np.arange(2020,2101,10)
Data = {}


Years2 = np.arange(1990,2021,1)
pa = {}
po = {}
pb = {}
eb = {}
es = {}
ew = {}
eo = {}
Pwd = {}
Pwf = {}
Pwi = {}
for c in BR_Countries['ISO']:
    ca = cg.copy()
    
    data  =  BR_Countries[BR_Countries['ISO']==c]
    cid = data['OBJECTID_1'].values[0]
    ca[cg!=cid] = np.nan
    ca[cg==cid] = 1
     
    p04 = []
    p514 = []
    p1564 = []
    p65 = []
     
    Years3 = np.arange(1981,2015,1)
    y = 2000
    pop = mask*Pop_H[y][urt]
    
    fb = Age_group['baby_h'][y]
    fs =Age_group['student_h'][y]
    fw =Age_group['worker_h'][y]
    fo = Age_group['old_h'][y]
    
    p04.append(np.nansum(mask*ca*pop*fb*np.squeeze(CDHD1s[Years3==2000,:,:])))
    p514.append(np.nansum(mask*ca*pop*fs*np.squeeze(CDHD1s[Years3==2000,:,:])))
    p1564.append(np.nansum(mask*ca*pop*fw*np.squeeze(CDHD1s[Years3==2000,:,:])))
    p65.append(np.nansum(mask*ca*pop*fo*np.squeeze(CDHD1s[Years3==2000,:,:])))
    
    print(str(cid)+' '+c+' '+str(y))
      
    eb[c] = np.array(p04) 
    es[c] = np.array(p514) 
    ew[c] = np.array(p1564) 
    eo[c] = np.array(p65) 
   
    
 
Data['historical'] = { 
                      'eb': pd.DataFrame(eb),
                      'es': pd.DataFrame(es),
                      'ew': pd.DataFrame(ew),
                      'eo': pd.DataFrame(eo)}


for s in range(1,4):
     
    eb = {}
    es = {}
    ew = {}
    eo = {}
     
    for c in BR_Countries['ISO']:
        ca = cg.copy()
        
        data  =  BR_Countries[BR_Countries['ISO']==c]
        cid = data['OBJECTID_1'].values[0]
        ca[cg!=cid] = np.nan
        ca[cg==cid] = 1
         
        p04 = []
        p514 = []
        p1564 = []
        p65 = []
         
        for i in range(len(Years)):
            pop = np.squeeze( mask*Pop_F[SN1[s]][urt][i])
             
            fb = Age_group['baby_f'][SN1[s]][Years[i]]
            fs =Age_group['student_f'][SN1[s]][Years[i]]
            fw =Age_group['worker_f'][SN1[s]][Years[i]]
            fo = Age_group['old_f'][SN1[s]][Years[i]]
            
             
            
            p04.append(np.nansum(mask*ca*pop*fb*CDHD2s[SN2[s]][fy==Years[i],:,:]))
            p514.append(np.nansum(mask*ca*pop*fs*CDHD2s[SN2[s]][fy==Years[i],:,:]))
            p1564.append(np.nansum(mask*ca*pop*fw*CDHD2s[SN2[s]][fy==Years[i],:,:]))
            p65.append(np.nansum(mask*ca*pop*fo*CDHD2s[SN2[s]][fy==Years[i],:,:]))
            
            print(str(cid)+' '+SN1[s]+' '+c+' '+str(Years[i]))
         
        eb[c] = np.array(p04) 
        es[c] = np.array(p514) 
        ew[c] = np.array(p1564) 
        eo[c] = np.array(p65) 
       
    
    
    Data[SN1[s]] = { 
                    'eb': pd.DataFrame(eb),
                    'es': pd.DataFrame(es),
                    'ew': pd.DataFrame(ew),
                    'eo': pd.DataFrame(eo)}











dn = ['eb','es','ew','eo']

AreaE2 = ['EA','CA','WA','NA','SA','SEA']
va = ['d','f','i']
xl = ['exposure (million person·days)','exposure (million person·days)','exposure (million person·days)','exposure ( million person·days)']
colors = ['#222222','#FCEC8D', '#FF9830', '#B51209']
fig, axs = plt.subplots(1, 4,figsize=(16,4),dpi=300)
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
for k in range(4):
    for i in range(4):
        data = []
        std_values = []
        for a  in AreaC:
            
            if SN1[k] == 'historical':
                ppp = Data['historical'][dn[i]]
                d = np.nansum(ppp[BR_Countries[BR_Countries['AREA']==a].ISO.values].values)
                dst = np.std(ppp[BR_Countries[BR_Countries['AREA']==a].ISO.values].values)
            else:
                ppp = Data[SN1[k]][dn[i]]
                d = np.nansum(np.nanmean(ppp[BR_Countries[BR_Countries['AREA']==a].ISO.values].values[(Years>=2080)&(Years<=2100),:],axis=0))
                dst = np.nanstd(np.nanmean(ppp[BR_Countries[BR_Countries['AREA']==a].ISO.values].values[(Years>=2080)&(Years<=2100),:],axis=0))
            d = d[d==d]
            std_values.append(np.sqrt(dst/1e7))
            data.append(d/1e6)
        pos = np.arange(len(data))
        bplot = axs[k].barh(3*pos+i*0.6, np.sqrt(np.squeeze(data)),xerr=np.squeeze(std_values),height = 0.55,color = colors[i])
        # bplot = axs[k].boxplot(data,positions =3*pos+i*0.6 , vert=False,patch_artist=True,boxprops={'facecolor': colors[i]},
        #         showfliers=False,
        #         showmeans=True,
        #     meanprops={'marker':'D',
        #                'markerfacecolor':'none', 
        #                'markeredgecolor':'#34F6F6',
        #                'markersize':'5'})
        for daa in range(len(data)):
            
            print([k,data[daa],std_values[daa],SN1[k],AreaE2[daa]])
    axs[k].set_yticks(3*pos+0.8,AreaE2)
    handles = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black',  ) for color in colors]
    axs[k].text(0.01,1.02, tit[k], transform=axs[k].transAxes)
    axs[k].set_xlabel(xl[k])
    axs[k].set_xlim([0,120])
    axs[k].set_xticks(axs[k].xaxis.get_ticklocs(),np.int16(np.power(axs[k].xaxis.get_ticklocs(),2)))
aggroup = ['0-4','5-14','15-64','65+']
fig.legend(handles, aggroup,  fontsize = 12,loc = 'lower center',edgecolor = 'none',facecolor='none',ncols = 4,bbox_to_anchor=(0.5, -0.1))








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

