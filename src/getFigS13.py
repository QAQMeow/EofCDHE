# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:12:13 2025

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
LON,LAT = np.meshgrid(lon,lat)
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
        p4.append(fb+fo)
        f.append(CDHD2s[SN2[s]][fy==Years[i]])
        e.append(pop*(fb+fo)*CDHD2s[SN2[s]][fy==Years[i]])
    pa = np.array(p1)
    pb = np.array(p2)
    po = np.array(p3)
    pe = np.array(e)
    pv = np.array(p4)    
    
    Data[SN1[s]] = {'pop_all':np.squeeze(pa),'pop_old':np.squeeze(po),'pop_baby':np.squeeze(pb),'pop_v':np.squeeze(pv),'e':np.squeeze(pe),'h':np.squeeze(np.array(f))}







# d_e = E_s1-E_b1
# d_e[np.abs(d_e)<1e-10] = 1e-10
# d_p = pop_s-pop_b
# d_v = vp_s-vp_b
# d_h = h_s-h_b

# c_h = pop_b * vp_b * d_h
# c_p = h_b * vp_b * d_p
# c_v = pop_b * h_b * d_v

# c_hp = vp_b * d_h*d_p
# c_hv = pop_b * d_h*d_v
# c_pv = h_b * d_p*d_v

# c_hpv = d_h*d_p*d_v

# c_e =  c_h+c_p+c_v+c_hp+c_hv+c_pv+c_hpv

# ct_h = c_h/c_e
# ct_p = c_p/c_e
# ct_v = c_v/c_e
# ct_hp = c_hp/c_e
# ct_hv = c_hv/c_e
# ct_pv = c_pv/c_e

# ct_hpv = c_hpv/c_e

# for c in BR_Countries['SOC']:
#     ca = cg.copy()
    
#     data  =  BR_Countries[BR_Countries['SOC']==c]
#     cid = data['FID'].values[0]
#     ca[ca!=cid] = np.nan
#     ca[ca==cid] = 1
#     re = np.nansum(ca*c_e)
#     rh = np.nansum(ca*c_h)
#     rv = np.nansum(ca*c_v)
#     rp = np.nansum(ca*c_p)
    
#     rhp = np.nansum(ca*c_hp)
#     rhv = np.nansum(ca*c_hv)
#     rpv = np.nansum(ca*c_pv)
    
#     rhpv = np.nansum(ca*c_hpv)
A = {}
# prd = (Years>=2020)&(Years<=2040)
# tit = 'Early-Century'
# prd = (Years>=2050)&(Years<=2070)
# tit = 'Mid-Century'
prd = (Years>=2080)&(Years<=2100)
tit = 'Late-Century'
for s in ['SSP1','SSP3','SSP5']:
    
    E_b = np.nanmean(Data['historical']['e'],axis=0)
    E_s = np.nanmean(Data[s]['e'][prd],axis=0)
    
    pop_b = np.nanmean(Data['historical']['pop_all'],axis=0)
    pop_s = np.nanmean(Data[s]['pop_all'][prd],axis=0)
    
    vp_b = np.nanmean(Data['historical']['pop_v'],axis=0)
    vp_s = np.nanmean(Data[s]['pop_v'][prd],axis=0)
    
    h_b = np.nanmean(Data['historical']['h'],axis=0)
    h_s = np.nanmean(Data[s]['h'][prd],axis=0)
    
    E_b1 = pop_b*vp_b*h_b
    
    E_b1[np.abs(E_b1)<1e-10] = 1e-10
    
    E_s1 = pop_s*vp_s*h_s
    d_e = (E_s1-E_b1)/E_b1
    pop_b[np.abs(pop_b)<1] = 1
    #vp_b[np.abs(vp_b)<1e-10] = 1e-10
    h_b[np.abs(h_b)<1] = 1
    
    d_p = (pop_s-pop_b)/pop_b
    d_v = (vp_s-vp_b)/vp_b
    d_h = (h_s-h_b)/h_b
    
    
    
    
    c_h = pop_b * vp_b * d_h
    c_p = h_b * vp_b * d_p
    c_v = pop_b * h_b * d_v
   

    B = {}
    for c in BR_Countries['ISO']:
        
        ca = cg.copy()
        
        data  =  BR_Countries[BR_Countries['ISO']==c]
        cid = data['OBJECTID_1'].values[0]
        if not cid in cg:
            print([cid,data['中文名称'].values[0],data['ISO'].values[0]])
        ca[cg!=cid] = np.nan
        ca[cg==cid] = 1
        
        
        
        
        w = ca*E_s1/np.nansum(ca*E_b1)
        #re = np.nansum(ca*d_e)
        rh = np.nansum(w*ca*d_h)
        rv = np.nansum(w*ca*d_v)
        rp = np.nansum(w*ca*d_p)
        B[c] = np.array([rh,rp,rv,np.abs(rh)+np.abs(rp)+np.abs(rv)])
    A[s] = B
    
    




def plotall(A):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    
    # 设置全局样式，确保所有子图风格统一
    plt.rcParams.update({
        'axes.grid': False,
        'grid.alpha': 0.3,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'font.size': 13,
        "figure.facecolor": (1,1,1,1)
     
    })
    
    # 创建4×4子图
    fig, axes = plt.subplots(8, 6, figsize=(16, 15))
    fig.subplots_adjust(
        top=0.92,    # 整体顶部边距
        bottom=0.08, # 整体底部边距
        left=0.08,   # 整体左侧左边距
        right=0.92,  # 整体右边距
        hspace=0.45,  # 子图之间的垂直间距
        wspace=0.55   # 子图之间的水平间距
    )
    

     

    # 用于记录堆叠的底部位置，初始为 0
   

    # 定义颜色（可根据需要调整，模拟类似你图中的颜色分层 ）
    colors =  ['#C2607A', '#E5BF67', '#ADADAB', ]
    cb = ['h','p','v']
    
    # 为每个子图绘制内容
    c = 0
    for i in range(8):
        for j in range(6):
            ax = axes[i, j]
            B = A['SSP1']
            data = {
                '126':100*A['SSP1'][list(B.keys())[c]][:3]/A['SSP1'][list(B.keys())[c]][3],
                
                '370': 100*A['SSP3'][list(B.keys())[c]][:3]/A['SSP3'][list(B.keys())[c]][3],
                '585':100*A['SSP5'][list(B.keys())[c]][:3]/A['SSP5'][list(B.keys())[c]][3]
            }
            ssp_cat  = ['126', '370', '585']
            
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
            ax_pos = ax.get_position()
            
            # 添加顶部紧贴矩形
            rect_top = Rectangle(
                (ax_pos.x0, ax_pos.y0 + ax_pos.height),  # 与子图顶部紧贴
                ax_pos.width,                            # 与子图同宽
                0.015,                                   # 矩形高度（适应子图密度）
                facecolor='none',
                edgecolor='#2C3E50',
                linewidth=1,
                transform=fig.transFigure,
                zorder=10
            )
            fig.patches.append(rect_top)
           
            # 在矩形中添加简短标签
            
            ax.set_title(list(B.keys())[c],pad = 4)
            c+=1
    # 添加整体标题
    h, lab = axes[0, 0].get_legend_handles_labels()
    
    # 将图例添加到整个图形（而非单个子图）
    fig.legend(
        [h[0],h[2],h[4]], ['Cimate change','Pop.growth','Vulnerable pop'],
        loc='lower center',  # 位置：底部中央
        ncol=3,  # 图例列数
        bbox_to_anchor=(0.5, -0.02),  # 微调位置（相对于图形）
        fontsize=15,
        borderaxespad = 4,
        edgecolor = 'none'
    )
    fig.suptitle(tit, fontsize=16, y=0.96)
    
    plt.show()
    return h, lab

h,l = plotall(A)