# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:42:01 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from PIL import Image
import threading
from datetime import datetime, timedelta
with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']

with open('../data/phatFisk90.pkl', 'rb') as f:  # 读取pickle文件
    Phat = joblib.load(f)
    f.close()
dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')
    
Alpha = Phat[0,mask==1]
Beta = Phat[1,mask==1]
Gamm = Phat[2,mask==1]

mask = maskdata['Mask']

def process(N,Dr,Dm,B):
    S = np.shape(Dr)
    for i in range(S[0]-N):
        Dm[i+N,:] = np.nansum(np.float32(Dr[i:i+N+1,:])*B,axis =0)
        #D[:N,:,:] = 0;


def getsapei(alpha,beta,gamm,x,N):
    
    sapei = np.zeros(len(x)+N)
    Fx = 1/(1+np.power(alpha/(x-gamm),beta))
    P = 1-Fx
    result = np.zeros(len(P))
    W= np.zeros(len(P))
    W[P<=0.5] = np.sqrt(-2*np.log(P[P<=0.5]))
    W[P>0.5] = np.sqrt(-2*np.log(1-P[P>0.5]))
    
    WW = np.power(W,2)
    WWW = np.power(W,3)
    C=[2.515517,0.802853,0.010328]
    d=[1.432788,0.189269,0.001308]
    result= W - (C[0] + C[1]*W + C[2]*WW) / (1 + d[0]*W + d[1]*WW + d[2]*WWW)
    
    result[P >0.5] = -result[P >0.5]
    sapei[N:] = result
    return sapei 



def process2(S,Dr,N,A,B,G):
    
    for i in range(np.shape(Dr)[1]):
        if not np.isnan(Dr[0,i]):
             
            S[:,i] = getsapei(A[i],B[i],G[i], Dr[:,i], N)



dt =  pd.date_range(start='1981-01-01', end='2014-12-31', freq='D')

MODEL = ['GFDL-ESM4']#'GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0'
#MODEL = ['IPSL-CM6A-LR']
#MODEL = ['MPI-ESM1-2-HR']
#MODEL = ['MRI-ESM2-0']

Period1 =['historical'] 
Period2 =['historical']
tp = ['1981_1990','1991_2000','2001_2010','2011_2014']
d1 = 'H:/data'

for model in MODEL:
    for period1 in Period1:
        for period2 in Period2:
            
            for timep in tp:
                
               
                d2pet = model+'/pet/'+period1
                d2pre = model+'/pr/'+period1
                pet_dir = d1+'/'+d2pet+'/'+period2+'_pet_asia_daily_'+timep+'.nc'
                pr_dir = d1+'/'+d2pre+'/'+period2+'_pr_asia_daily_'+timep+'.nc'
                predataset = nc.Dataset(pr_dir)
                petdataset = nc.Dataset(pet_dir)
                prd = predataset['pr'][:]*3600*24
                petd = petdataset['pet'][:]
                
                if timep == tp[0]:
                    pre = prd[:,mask==1] 
                    pet = petd[:,mask==1] 
                else:
                    pre = np.concatenate((pre,prd[:,mask==1]),axis=0,out=None)
                    pet = np.concatenate((pet,petd[:,mask==1]),axis=0,out=None)
                petdataset.close()
                predataset.close()
                
            D = pre-pet
            sha = np.shape(D)
            N = 90
            a = 0.98
            A = np.float32(np.zeros([N+1,np.shape(D)[1]]))
            for i in range(N+1):
                A[i,:] = np.power(a,N-i)
            
                
            Dx = np.float32(np.zeros(sha))
                  
            s = np.array([ 0,0,0,0,0,0,0],dtype = int)
            e = np.array([2000,4000,6000,8000,10000,12000,sha[0]],dtype = int)
            for i in range(len(e)-1):
                s[i+1] = e[i]-N
            for i in range(len(e)-1):
                s[i+1] = e[i]-N
            T = {}
            for i in range(7):
                T[i] = threading.Thread(target=process,args=(N,D[s[i]:e[i],:],Dx[s[i]:e[i],:],A))
            for i in range(7):
                T[i].start()
            for i in range(7):
                T[i].join()
            
            
            Db = Dx[N:,:].copy() 
            SAPEI = np.float32(np.zeros_like(Dx))
            
            
            s1 = np.array([ 0,0,0,0,0,0,0,0,0,0,0],dtype = int)
            e1 = np.array([2000,4000,6000,80000,10000,12000,14000,16000,18000,20000,sha[0]],dtype = int)

            T2 = {}
            for i in range(len(s1)):
                T2[i] = threading.Thread(target=process2,args=(SAPEI[:,s1[i]:e1[i]],Db[:,s1[i]:e1[i]],N,Alpha[s1[i]:e1[i]],
                                                              Beta[s1[i]:e1[i]],Gamm[s1[i]:e1[i]]))
            for i in range(len(s1)):
                T2[i].start()
            for i in range(len(s1)):
                T2[i].join()

            
            sapei = np.zeros([sha[0],210,341])*np.nan
            for i in range(sha[0]):
                sapei[i,mask==1] = SAPEI[i,:]
                
            
                
            for timep in tp:
                ts = np.int16(timep[:4])
                te = np.int16(timep[5:])
                
                sapeif = sapei[(dt.year>=ts) &(dt.year<=te)]
                d2 = model+'/sapei/'+period1
                if not os.path.exists(d1+'/'+d2):
                    os.makedirs(d1+'/'+d2)
                NewData = nc.Dataset(d1+'/'+d2+'/'+period2+'_sapei_asia_daily_'+timep+'.nc', 'w', format='NETCDF4')
                NewData.description = 'the Standardized Antecedent Precipitation Evapotranspiration Index in Asia from '+timep[:4]+' to '+timep[5:]

                time = NewData.createDimension('time', None)
                lat = NewData.createDimension('lat', 210)
                lon = NewData.createDimension('lon', 341)

                times = NewData.createVariable("time", "f8", ("time",))
                times.units = 'days since 1981-1-1 00:00:00'
                times.axis = 'T'
                times.calendar = 'proleptic_gregorian'
                start_date = datetime(ts, 1, 1)
                end_date = datetime(te, 12, 31)
                delta = end_date - start_date
                num_days = delta.days + 1
                dates = [start_date + timedelta(days=i) for i in range(num_days)]
                time_values = nc.date2num(dates, units='days since 1981-1-1 00:00:00', calendar='proleptic_gregorian')
                times[:] = time_values


                latitudes = NewData.createVariable("lat", "f8", ("lat",))
                latitudes.units = dataBR.variables['lat'].units
                latitudes.axis = 'Y'
                latitudes[:] = dataBR.variables['lat'][:]

                longitudes = NewData.createVariable("lon", "f4", ("lon",))
                longitudes.units = dataBR.variables['lon'].units
                longitudes.axis = 'X'
                longitudes[:] = dataBR.variables['lon'][:]

                Gdata = NewData.createVariable('sapei', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                                               least_significant_digit=3)
                Gdata.units = ' '
                Gdata.standard_name = 'the Standardized Antecedent Precipitation Evapotranspiration Index'
                Gdata.missing_value = -9999
                Gdata[:, :, :] = sapeif

                NewData.close()    
                
                print(model+' '+period1+' '+period2+' '+timep)
                
                
                
                
                