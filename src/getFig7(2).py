 

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 17:35:14 2025

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

 


SN1 = ['historical','SSP1','SSP3','SSP5']
SN2 = ['historical','ssp126','ssp370','ssp585']

AreaC = ['东亚','中亚','西亚','北亚','南亚','东南亚']
AreaE = ['East Asia','Cental Asia','Western Asia','Northern Asia','Southern Asia','Southeast Asia']
 


fy =  np.arange(2015,2101,1) 
Years = np.arange(2020,2101,10)
Data = {}


Years2 = np.arange(1990,2021,1)

urt = 'Rural'
urt = 'Urban'
aggr  = ['preschool children','students','worker','elder']
sbt = urt +' ' + aggr[3]

ti = ['e','f']
cpop = []
cbaby = []
cold = []
p1h = []
p2h = []
p3h = []
p4h = []
eh = []
fh = []
Years3 = np.arange(1981,2015,1)
y = 2000
poph = Pop_H[y][urt]
fbh = Age_group['baby_h'][y]
fsh = Age_group['student_h'][y]
fwh = Age_group['worker_h'][y]
foh = Age_group['old_h'][y]
frah = foh 



 
p1h.append(poph) 
 
p4h.append(frah)

fh.append(np.squeeze(CDHD1s[Years3==y])*mask)
eh.append(poph*(frah)*np.squeeze(CDHD1s[Years3==y])*mask)

pah = np.array(p1h)
pfh = np.array(fh)
pvh = np.array(poph*frah/(poph+1))
peh = np.array(eh)
     

Data['historical'] = {'pop_h':np.squeeze(pah)*mask,
                      'pop_v':np.squeeze(pvh),
                      'e':np.squeeze(peh)*mask,
                      'h':np.squeeze(np.array(pfh))}
     

 


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
    for i in range(len(Years)):
        pop = np.squeeze(Pop_F[SN1[s]][urt][i])
        fb = Age_group['baby_f'][SN1[s]][Years[i]]
        fs =Age_group['student_f'][SN1[s]][Years[i]]
        fw =Age_group['worker_f'][SN1[s]][Years[i]]
        fo = Age_group['old_f'][SN1[s]][Years[i]]
        
        fra =  fo
        
        
        p1.append(pop)
         
        p3.append(pop*fra)
        p4.append(pop*fra/(pop+1))
        f.append(np.squeeze(CDHD2s[SN2[s]][fy==Years[i]])*mask)
        e.append(pop*(fra)*np.squeeze(CDHD2s[SN2[s]][fy==Years[i]])*mask)
        
    pa = np.array(p1)
    pf = np.array(f)
    pe = np.array(e)
    pv = np.array(p4)    
    
    Data[SN1[s]] = {'pop_f':np.squeeze(pa)*mask,
                    'pop_v':np.squeeze(pv),
                    'e':np.squeeze(pe)*mask,
                    'h':np.squeeze(np.array(pf))}


A = {}
Ar = ['东亚','中亚','西亚','北亚','南亚','东南亚']
rg = (Years>=2020)&(Years<=2040)


# rg = (Years>=2050)&(Years<=2070)
# sbt = 'Mid-century'
# ti = ['c','d']
# rg = (Years>=2080)&(Years<=2100)
# sbt = 'Late-century'
# ti = ['e','f']
for s in ['SSP1','SSP3','SSP5']:
    E_b = Data['historical']['e']*mask
    E_s = np.nanmean(Data[s]['e'][rg],axis=0)*mask
    
    pop_b = Data['historical']['pop_h']*mask
    pop_s = np.nanmean(Data[s]['pop_f'][rg],axis=0)*mask
    
    vp_b = Data['historical']['pop_v']*mask
    vp_s = np.nanmean(Data[s]['pop_v'][rg],axis=0)*mask
    
    h_b = Data['historical']['h']*mask
    h_s = np.nanmean(Data[s]['h'][rg],axis=0)*mask
    
    E_b1 = pop_b*vp_b*h_b
    
    #E_b1[np.abs(E_b1)<1e-10] = 1e-10
    E_s1 = pop_s*vp_s*h_s
    d_e = (E_s1-E_b1)/(E_b1)
    d_e[E_b1==0] = (E_s1-E_b1)[E_b1==0]
    #pop_b[np.abs(pop_b)<1] = 1E-6
    #pop_s[np.abs(pop_s)<1] =  1E-6
    #vp_b[np.abs(vp_b)<1e-10] = 1e-10
    #h_b[h_b<=1] = 1
    
    
    d_p = (pop_s-pop_b)/(pop_b)
    #d_p[pop_b==0] =(pop_s-pop_b)[pop_b==0]
    d_v = (vp_s-vp_b)/vp_b
    
    d_h = (h_s-h_b)/(h_b)
    #d_h[h_b==0] = (h_s-h_b)[h_b==0]
    
    #c_h = pop_b * vp_b * d_h
    #c_p = h_b * vp_b * d_p
    #c_v = pop_b * h_b * d_v
   

    B = {}
    for a in range(7):
        if a<6:
            ca = ag.copy()
            
             
             
            ca[ca!=a] = 0
            ca[ca==a] = 1
            w = ca*E_s1/np.nansum(ca*E_b1)
            
            re = np.nansum(ca*d_e)
            rh = np.nansum(w*ca*d_h)
            rv = np.nansum(w*ca*d_v)
            rp = np.nansum(w*ca*d_p)
            B[Ar[a]] = np.array([rh,rp,rv,np.abs(rh)+np.abs(rp)+np.abs(rv)])
        else:
            ca = ag.copy()
            
             
             
            ca[ca<0] = 0
            ca[ca>=0] = 1
            w = ca*E_s1/np.nansum(ca*E_b1)
            re = np.nansum(ca*d_e)
            rh = np.nansum(w*ca*d_h)
            rv = np.nansum(w*ca*d_v)
            rp = np.nansum(w*ca*d_p)
            B['Asia'] = np.array([rh,rp,rv,np.abs(rh)+np.abs(rp)+np.abs(rv)])
    A[s] = B
    
    

 


# 定义区域和情景
Ar = ['东亚','中亚','西亚','北亚','南亚','东南亚']
regions =  ['East Asia','Cental Asia','Western Asia','Southern Asia','Northern Asia','Southeast Asia']
scenarios = ['SSP126',  'SSP370', 'SSP585']
# 全球作为单独的子图，这里也列出来方便统一处理
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
colors = ['#b02425', '#Faa419', '#4583B6', ]
cb = ['h','p','v']
D = {}
for i in range(6):
    row = i // 3
    col = i % 3
    ax = plt.subplot(gs[row, col])
    axes_region.append(ax)
    region = regions[i]
    ax.set_title(region)
    # 绘制分组柱状图，不同颜色代表不同组成部分（这里简单分三种颜色示例，你可按实际数据调整）
    bar_width = 0.2
    B = A['SSP1'].copy()
    data = {
                '126':100*A['SSP1'][list(B.keys())[i]][:3]/A['SSP1'][list(B.keys())[i]][3],
                
                '370': 100*A['SSP3'][list(B.keys())[i]][:3]/A['SSP3'][list(B.keys())[i]][3],
                '585':100*A['SSP5'][list(B.keys())[i]][:3]/A['SSP5'][list(B.keys())[i]][3]
            }
    ssp_cat  = ['126', '370', '585']
    D[Ar[i]] = data
    print(Ar[i])
    print(data)
    # 转换数据为 numpy 数组，方便计算堆叠
    data_array = np.array([data[ssp] for ssp in ssp_cat])
    data_array1 = data_array.copy()
    data_array2 = data_array.copy()
    data_array1[data_array1<0] = 0;
    data_array2[data_array2>=0] = 0;
    # 获取分层数量
    num_layers = data_array.shape[1]
    bottom1 = np.zeros(len(ssp_cat ))
    bottom2 = np.zeros(len(ssp_cat ))
    for s in range(num_layers):
    
        ax.bar(
                [1,2,3],  # x 轴类别
                data_array1[:, s],  # 当前层的数值
                bottom=bottom1,  # 堆叠的底部位置
                color=colors[s],  # 当前层的颜色
                label=cb[s]  # 图例标签)
        )
        bottom1 += data_array1[:,s]
        p = ax.bar(
                [1,2,3],  # x 轴类别
                data_array2[:, s],  # 当前层的数值
                bottom=bottom2,  # 堆叠的底部位置
                color=colors[s],  # 当前层的颜色
                label=cb[s]  # 图例标签)
        )
        bottom2 += data_array2[:,s]
    # 获取当前子图的位置
    ax.set_xlim([0.5,3.5])
    ax.set_xticks([1,2,3],ssp_cat)
    # 假设数据有三个组成部分，这里用随机数模拟占比，实际替换为真实分类数据
    #ax.plot(2050,1)
    #ax.set_xticks(x)
    #ax.set_xlim([2020,2100])
    #ax.set_ylabel('Y')
    ax_pos = ax.get_position()

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
 

# fig.legend(
#     [h[0],h[2],h[4]], ['Cimate change','Pop.growth','Vulnerable pop'],
#     loc='lower center',  # 位置：底部中央
#     ncol=3,  # 图例列数
#     bbox_to_anchor=(0.5, -0.02),  # 微调位置（相对于图形）
#     fontsize=15,
#     borderaxespad = 4,
#     edgecolor = 'none'
# )
fig.suptitle(sbt, fontsize=16, y=1.1)
B = A['SSP1']
data = {
            '126':100*A['SSP1'][list(B.keys())[-1]][:3]/A['SSP1'][list(B.keys())[-1]][3],
            
            '370': 100*A['SSP3'][list(B.keys())[-1]][:3]/A['SSP3'][list(B.keys())[-1]][3],
            '585':100*A['SSP5'][list(B.keys())[-1]][:3]/A['SSP5'][list(B.keys())[-1]][3]
        }
D['Asia'] = data
ssp_cat  = ['126', '370', '585']
    
print(data)
# 转换数据为 numpy 数组，方便计算堆叠
data_array = np.array([data[ssp] for ssp in ssp_cat])
data_array1 = data_array.copy()
data_array2 = data_array.copy()
data_array1[data_array1<0] = 0;
data_array2[data_array2>=0] = 0;
# 获取分层数量
num_layers = data_array.shape[1]
bottom1 = np.zeros(len(ssp_cat ))
bottom2 = np.zeros(len(ssp_cat ))
for s in range(num_layers):

    ax_global.bar(
            [1,2,3],  # x 轴类别
            data_array1[:, s],  # 当前层的数值
            bottom=bottom1,  # 堆叠的底部位置
            color=colors[s],  # 当前层的颜色
            label=cb[s]  # 图例标签)
    )
    bottom1 += data_array1[:,s]
    p = ax_global.bar(
            [1,2,3],  # x 轴类别
            data_array2[:, s],  # 当前层的数值
            bottom=bottom2,  # 堆叠的底部位置
            color=colors[s],  # 当前层的颜色
            label=cb[s]  # 图例标签)
    )
    bottom2 += data_array2[:,s]
ax_global.set_xlim([0.5,3.5])
ax_global.set_xticks([1,2,3],ssp_cat)

ax_global.set_ylabel('Relative Contribution(%)')
h, lab = ax_global.get_legend_handles_labels()

# 将图例添加到整个图形（而非单个子图）
fig.legend(
    [h[0],h[2],h[4]], ['Cimate change','Pop.growth','Fraction change'],
    loc='lower center',  # 位置：底部中央
    ncol=3,  # 图例列数
    bbox_to_anchor=(0.5, -0.02),  # 微调位置（相对于图形）
    fontsize=15,
    borderaxespad = -1,
    edgecolor = 'none'
)
fig.text(0.04,0.3,'Relative Contribution(%)',rotation = 'vertical')
fig.text(0.04,1,ti[0],fontsize = 16, fontweight='bold',)
fig.text(0.73,1,ti[1],fontsize = 16, fontweight='bold',)
# 调整子图间距
#plt.tight_layout()
plt.show()



