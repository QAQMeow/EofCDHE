# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 02:49:25 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:45:14 2025

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
from getgridC import getCG
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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

CDHF1s = CDHEs_data['historical_frequency']
CDHD1s = CDHEs_data['historical_days'] 
CDHA1s = CDHEs_data['historical_area'] 
CDHI1s = CDHEs_data['historical_intensity'] 
CDHF2s = CDHEs_data['future_frequency'] 
CDHD2s = CDHEs_data['future_days']  
CDHA2s = CDHEs_data['future_area'] 
CDHI2s = CDHEs_data['future_intensity'] 

hy = np.arange(1981,2015)
hy2 = np.arange(1990,2021)
fy = np.arange(2015,2101)
fy2 = np.arange(2015,2101,5)

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geopandas import GeoDataFrame, points_from_xy
from tqdm import tqdm

def zonal_statistics(shp_path, matrix, lat,lon, stats=['sum', 'mean', 'max', 'min', 'count']):
    """
    对shapefile中的多个面要素进行自定义矩阵的分区统计
    
    参数:
        shp_path: shapefile文件路径
        matrix: 自定义数值矩阵 (二维numpy数组)
        x_range: 矩阵x轴范围 (min_x, max_x)
        y_range: 矩阵y轴范围 (min_y, max_y)
        stats: 需要计算的统计指标列表，可选值: sum, mean, max, min, count
        
    返回:
        GeoDataFrame: 包含原始几何和统计结果的地理数据框
    """
    # 1. 读取shapefile
    gdf = gpd.read_file(shp_path)
    print(f"成功读取shapefile，包含 {len(gdf)} 个面要素")
    
     
    X, Y = np.meshgrid(lon+0.25, lat+0.25)
    
    # 转换为点列表和对应的值
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = matrix.ravel()
    
    # 3. 创建点地理数据框
    gdf_points = GeoDataFrame(
        {'value': values},
        geometry=points_from_xy(points[:, 0], points[:, 1]),
        crs=gdf.crs  # 与面要素保持相同坐标系统
    )
    
    # 4. 对每个面要素进行统计
    # 创建统计结果列
    for stat in stats:
        gdf[f'matrix_{stat}'] = 0
    
    # 遍历每个面要素
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="分区统计中"):
        polygon = row['geometry']
        
        # 快速筛选可能在面内的点（边界框过滤）
        bbox = polygon.bounds
        mask = (
            (gdf_points.geometry.x >= bbox[0]) & 
            (gdf_points.geometry.x <= bbox[2]) & 
            (gdf_points.geometry.y >= bbox[1]) & 
            (gdf_points.geometry.y <= bbox[3])
        )
        candidate_points = gdf_points[mask]
        
        # 精确判断点是否在面内
        in_points = candidate_points[candidate_points.geometry.within(polygon)]
        
        # 计算统计指标
        if len(in_points) > 0:
            if 'sum' in stats:
                gdf.at[idx, 'matrix_sum'] = in_points['value'].sum()
            if 'mean' in stats:
                gdf.at[idx, 'matrix_mean'] = in_points['value'].mean()
            if 'max' in stats:
                gdf.at[idx, 'matrix_max'] = in_points['value'].max()
            if 'min' in stats:
                gdf.at[idx, 'matrix_min'] = in_points['value'].min()
            if 'count' in stats:
                gdf.at[idx, 'matrix_count'] = len(in_points)
    
    return gdf

gdfDf = []
gdfPf = []
for p in [[2020,2040],[2050,2070],[2080,2100]]:

    gdfd = zonal_statistics(
            shp_path='H:/65国家/ASIA/Asia_Sub.shp',
            matrix=np.nanmean(CDHD2s['ssp370'][(fy>=p[0])&(fy<=p[1]),:,:],axis=0),
            lat = lat,
            lon = lon,
            stats=['sum', 'mean'] )
    Pop = []
    for i in range(p[0],p[1]+1,5):
        fb = Age_group['baby_f']['SSP3'][i]
        fo = Age_group['old_f']['SSP3'][i]
        Pop.append(PopF['SSP3'][str(i)]*(fb+fo))
    Pop = np.array(Pop)  
    gdfp = zonal_statistics(
            shp_path='H:/65国家/ASIA/Asia_Sub.shp',
            matrix=np.nanmean(Pop,axis=0),
            lat = lat,
            lon = lon,
            stats=['sum', 'mean'] )
    gdfDf.append(gdfd)
    gdfPf.append(gdfp)
    print(p)

gdfDh = zonal_statistics(
        shp_path='H:/65国家/ASIA/Asia_Sub.shp',
        matrix=np.nanmean(CDHD1s[hy>1994,:,:],axis=0),
        lat = lat,
        lon = lon,
        stats=['sum', 'mean'] )
 
Pop = PopH[(hy2>1994)&(hy2<2015)]
Pop2 = np.zeros_like(Pop)
hy3 = np.arange(1995,2014)
for y in range(len(hy3)):
    fb = Age_group['baby_h'][hy3[y]]
    fo = Age_group['old_h'][hy3[y]]
    Pop2[y] =  Pop[hy3[y]-1995]*(fb+fo)
gdfPh = zonal_statistics(
        shp_path='H:/65国家/ASIA/Asia_Sub.shp',
        matrix=np.nanmean(Pop2,axis=0),
        lat = lat,
        lon = lon,
        stats=['sum', 'mean'] )


with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']
lat = maskdata['lat']
lon = maskdata['lon']

def getCE(gdfd,gdfp):
    A = gdfd['matrix_mean']
    B = gdfp['matrix_sum']
    
    ad = [np.nanmin(A),5,25,50,75]
    bd = [np.nanmin(B),0,1e6,5e6,1e7]
    C = np.zeros_like(A)
    k = 0;
    for i in range(5):
        for j in range(5):
            C[(A>=ad[i])&(B>=bd[j])] = k
            k+=1  
            
    C = np.int16(C)
         
    CE = pd.DataFrame({'NAME':gdfDh['shapeID'],'E':C})
    return CE,ad,bd

CE =  [] 
AD = []
BD = []
ceh,adh,bdh =getCE(gdfDh,gdfPh) 
CE.append(ceh)
AD.append(adh)
BD.append(bdh)

for i in range(3):
    cef,adf,bdf =getCE(gdfDf[i],gdfPf[i]) 
    CE.append(cef)
    AD.append(adf)
    BD.append(bdf)
    
df = pd.read_table("color1.txt",header=None, delim_whitespace=True)
y = 'Test'
c4 = df.values[:,1:4]
c = ListedColormap(c4/255,name = 'test')  



def PLot1(i,j,ax,ax2,mlon,mlat,CE,c,a,b,prd,tit):
    #c = cmaps.sunshine_9lev
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import  BoundaryNorm

    #plt.title(y)
    #m = Basemap(projection='robin',resolution='l',lon_0=0)
    m = Basemap(ax = ax,projection = 'cyl',resolution='l',lon_0=0,llcrnrlon = 0, llcrnrlat = -15, urcrnrlon = 200, urcrnrlat = 90)
    m.fillcontinents(color = '#DDDDDD')
    #plt.text(153,40,y,fontsize = 7)
    LON,LAT = np.meshgrid(mlon,mlat)
    Lon = LON 
    xi, yi = m(Lon, LAT)
    #c =cmaps.sunshine_9lev


    #m.drawcoastlines() 
    #m.readshapefile('H:/65国家/一带一路/Export_Output_2', 'BR_countries',drawbounds=True,linewidth = 0.25,)

    m.readshapefile('H:/65国家/ASIA/Asia_Sub', 'AS_2l',drawbounds=False)
    df_poly = pd.DataFrame({
            'shapes': [Polygon(np.array(shape)) for shape in m.AS_2l],
            'area': [area['shapeID'] for area in m.AS_2l_info]
        })


    colrnum = pd.DataFrame({'area':CE['NAME'],'L':CE['E']})
    df_poly2 = df_poly.merge(colrnum,how='left')
    norm = Normalize(vmin=0,vmax = 25)
    pc = PatchCollection(df_poly2.shapes, zorder=1)
     
#pc.set_facecolor(np.array([0.1,0.2,0.3,0.3]))
    pc.set_facecolor(c(norm(df_poly2['L'].fillna(0).values)))
    pc.set_alpha(0.75)
    pc.set_edgecolor('#FFFFFF')
    pc.set_linewidth(0.15)
    ax.add_collection(pc)
    m.readshapefile('H:/65国家/ASIA/Asia', 'bc',drawbounds=True,linewidth = 0.4,color = '#ffffff')
    m.readshapefile('H:/65国家/ASIA/AsiaSub', 'BR',drawbounds=True,linewidth = 0.45,color = '#000000')
    
    # citys = pd.read_excel('../data/City2.xlsx')
    # ax.scatter(citys.LON,citys.LAT,s=6,linewidth = 0.5,facecolor ='none',edgecolor ='#0C4458',marker = 'o', zorder=2)
    
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    if i==1:
        ax.set_xticks([ 0 ,30, 60, 90, 120, 150, 180],['0°' ,'30° E', '60° E', '90° E', '120° E', '150° E', '180°'])
    if j==0:
        ax.set_yticks([-15,0, 30, 60, 90],['15° S' ,'0°', '30° N', '60° N', '90° N'])
    ax.text(0.09,0.1, prd,fontsize = 11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.01,1.04, tit, transform=ax.transAxes)
    left, bottom, width, height = 0.77, 0.27, 0.1, 0.15
    a = np.array(a)
    b = np.array(b)
    C = np.arange(1,26).reshape(5,5)
    #ax2 = fig.add_axes([left, bottom, width, height])
    ax2.pcolor(C,cmap=c,alpha = 0.75)
    ax2.set_xticks([1,2,3,4],np.int16(b[[1,2,3,4]]/1e6),fontsize = 7)
    ax2.set_yticks([1,2,3,4],a[[1,2,3,4]],fontsize = 7)
    
    ax2.set_xlabel('VP,million',labelpad = 0.08,fontsize = 7)
    ax2.set_ylabel('duration,days',labelpad = 0.08,fontsize = 7)
    
    #ax.text(150,20,[i,j])


       
        
 
   
va = ['d','f','i']
sn = ['ssp126','ssp370','ssp585']
van = ['Duration','Frequency','Intensity']
PRD = ['Baseline','Early-Century','Mid-Century','Late-Century']

width_cm = 20# 设置图形宽度
height_cm = 12 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54

# 使用inch指定图形大小
#fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)

 
fig = plt.figure(figsize=(width_inch, height_inch),dpi=300)

outer_grid = GridSpec(2, 2, figure=fig,wspace=0.02,hspace=0.01)
from matplotlib import rcParams
config = {
           "font.family": 'serif',
           "font.size": 8,# 相当于小四大小
           "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
           "font.serif": ['Arial'],#宋体
           'axes.unicode_minus': False ,# 处理负号，即-号
           'axes.grid': False,
           "figure.facecolor": (1,1,1,1)
           
        }
rcParams.update(config) 
k = 0
Tit = ['(a)','(b)','(c)','(d)']
for i in range(2):
   for j in range(2):

       #ax = fig.add_subplot(gs[i, j])
       inner_grid = GridSpecFromSubplotSpec(
           20, 40,  # 5行5列网格
           subplot_spec=outer_grid[i,j],
           wspace=0.05, hspace=0.05
       )
       ax = fig.add_subplot(inner_grid[:, :]) 
       ax2 = fig.add_subplot(inner_grid[10:15,33:39])
       PLot1(i,j,ax,ax2,lon,lat,CE[k],c,AD[k],BD[k],PRD[k],Tit[k])
       k+=1
       

