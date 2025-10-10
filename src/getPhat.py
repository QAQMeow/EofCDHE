# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:44:35 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
import os
import numpy as np
import joblib
import netCDF4 as nc
import threading 
from FiskFit2 import FiskFit2
from datetime import datetime, timedelta


with open('../data/Maskdata.pkl', 'rb') as f:  # 读取pickle文件
    maskdata = joblib.load(f)
    f.close()
mask = maskdata['Mask']


dataBR = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(1979)+'.nc')

Pre = []
PET = []
for y in range(1981,2021):
    predata = nc.Dataset('H:/BR/Data/Asia/pre/asia_pre_'+str(y)+'.nc')
    pr = np.array(predata.variables['pre'])*3600*24
    petdata = nc.Dataset('H:/BR/Data/Asia/pet/asia_pet_'+str(y)+'.nc')
    pet = np.array(petdata.variables['pet'])*3600*24
    if y == 1981:
        Pre = pr[mask==1,:]
        PET = pet[mask==1,:]
    else:
        Pre = np.concatenate((Pre,pr[mask==1,:]),axis=1,out=None)
        PET = np.concatenate((PET,pet[mask==1,:]),axis=1,out=None)
    print(y)
    predata.close()
    petdata.close()
#D:Pre-PET    

D = Pre-PET
sha = np.shape(D)
# with open('../Data/PRE025/pre.pkl', 'wb') as f: 
#     joblib.dump(Pre,f)
#     f.close()
# with open('../Data/PET025/pet.pkl', 'wb') as f:
#     joblib.dump(PET,f)
#     f.close()
    
#del Pre,PET


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



N = 90
a = 0.98
A = np.float32(np.zeros([np.shape(D)[0],N+1]))
for i in range(N+1):
    A[:,i] = np.power(a,N-i)
    
Dx = np.float32(np.zeros(sha))

def process(N,Dr,Dm,B):
    S = np.shape(Dr)
    for i in range(S[1]-N):
        Dm[:,i+N] = np.nansum(np.float32(Dr[:,i:i+N+1])*B,axis = 1)
        #D[:N,:,:] = 0;
           
s = np.array([ 0,0,0,0,0,0,0],dtype = int)
e = np.array([2000,4000,6000,8000,10000,12000,sha[1]],dtype = int)
for i in range(len(e)-1):
    s[i+1] = e[i]-N
T = {}
for i in range(7):
    T[i] = threading.Thread(target=process,args=(N,D[:,s[i]:e[i]],Dx[:,s[i]:e[i]],A))
for i in range(7):
    T[i].start()
for i in range(7):
    T[i].join()

with open('../data/Dx'+str(N)+'.pkl', 'wb') as f:  # 读取pickle文件
    joblib.dump(Dx,f)
    f.close()
    
    
# with open('../Data/Dx.pkl', 'rb') as f:  # 读取pickle文件
#     Dx = joblib.load(f)
    
#     f.close()



SAPEI = np.float32(np.zeros_like(Dx))

Ph = np.zeros([sha[0],3])

# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'));   
# for i in range(sha[0]):
#     [alpha,beta,gamm] = FiskFit2(Dx[i,N:])
#     Ph[i,:] = [alpha,beta,gamm]
#     #SAPEI[:,i] = getsapei(alpha, beta, gamm, Dx[N:,i], N)  
#     if i%10000==0:
#         print(i)
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'));

 
Db = Dx[:,N:].copy()
del Dx

print('Dx ready')
#[alpha,beta,gamm] = FiskFit2(np.nanmean(np.float32(Db),axis=1))
#[alpha,beta,gamm] = np.load(wr+'abg.npy')




def process2(S,ph,Dr):
    
    for i in range(np.shape(Dr)[0]):
        if not np.isnan(Dr[i,1]):
            [alpha,beta,gamm] = FiskFit2(Dr[i,:])
            ph[i,:] = [alpha,beta,gamm]
            S[i,:] = getsapei(alpha, beta, gamm, Dr[i,:], N)
s1 = np.array([ 0,0,0,0,0,0,0,0,0,0,0],dtype = int)
e1 = np.array([2000,4000,6000,80000,10000,12000,14000,16000,18000,20000,sha[0]],dtype = int)

T = {}
for i in range(len(s1)):
    T[i] = threading.Thread(target=process2,args=(SAPEI[s1[i]:e1[i],:],Ph[s1[i]:e1[i],:],Db[s1[i]:e1[i],:]))
for i in range(len(s1)):
    T[i].start()
for i in range(len(s1)):
    T[i].join()


phat = np.zeros([3,210,341])*np.nan
for i in range(3):
    phat[i,mask==1] = Ph[:,i]


with open('../data/phatFisk'+str(N)+'.pkl', 'wb') as f:  # 读取pickle文件

    joblib.dump(phat,f)
    f.close()


sapei = np.zeros([sha[1],210,341])*np.nan
for i in range(sha[1]):
    sapei[i,mask==1] = SAPEI[:,i]
    
    
NewData = nc.Dataset('H:/BR/Data/Asia/sapei/asia_sapei_1981_2020.nc', 'w', format='NETCDF4')
NewData.description = 'the Standardized Antecedent Precipitation Evapotranspiration Index in Asia from 1981 to 2020'

time = NewData.createDimension('time', None)
lat = NewData.createDimension('lat', 210)
lon = NewData.createDimension('lon', 341)

times = NewData.createVariable("time", "f8", ("time",))
times.units = 'days since 1981-1-1 00:00:00'
times.axis = 'T'
times.calendar = 'proleptic_gregorian'
start_date = datetime(1981, 1, 1)
end_date = datetime(2020, 12, 31)
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
Gdata[:, :, :] = sapei

NewData.close()    








