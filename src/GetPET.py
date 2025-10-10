# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:55:35 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

  
def GetPET(tmax,tmin,rlds,rsds,hurs,wind10,alt):
    """
    该函数利用Penman-Monteith公式计算日潜在蒸散发（mm)
    参考算法见《GBT 20481-2017 气象干旱等级》附录C
    输入值及单位：
    tmax：日最高气温(K)
    tmin：日最低气温(K)
    hurs：近地相对湿度(100%)
    wind10：近地风速(m/s) 10m高度
    alt：海拔(m)
    rlds :向下长波辐射 W/m^2
    rsds：向下短波辐射 W/m^2
    @author: Van M10w
    """
    import numpy as np
    # 开氏温度转换
    tmax = np.float32(tmax -273.16)
    tmin = np.float32(tmin -273.16)
    tas=(tmax+tmin)/2.0
    
    #es,ea  #饱和水汽压和实际水汽压的计算
    etx= np.float32(0.6108*np.exp((17.27*(tmax))/(tmax+237.3)))
    etn= np.float32(0.6108*np.exp((17.27*(tmin))/(tmin+237.3)))
    es=(etx+etn)/2.0
    ea= np.float32(es*hurs/100)  #相对湿度=实际水汽压/饱和水汽压

    #slope饱和水汽压曲线斜率计算
    a = 4098*0.6108*np.exp(17.27*(tas)/(tas+237.3));
    b = np.power(tas+237.3,2);
    slope=a/b;
    
    #Rn净辐射计算
    #净辐射Rn
    Rns= np.float32((1-0.23)*rsds*60*60*24/1e6)# W/m^2 -> MJ/m^2·d
    #0.94为比辐射率
    Rnl =  np.float32(0.94*(4.903e-9*np.power(tas+273.16,4)-rlds*60*60*24/1e6))# W/m^2 -> MJ/m^2·d
    
    Rn=(Rns-Rnl)    # W/m^2 -> MJ/m^2·d
    del rsds,rlds,tmin
    #土壤热通量
    G=0;  #日尺度的土壤热通量相当小，可忽略
    
    #U2的换算（需要2米高处的风速，下式为10米高处风速向2米高处风速的转换），单位 m/s
    U2= np.float32(wind10*4.87/np.log(67.8*10-5.42))

    del wind10
    #psy干湿表常数计算(若有大气压强观测值，可直接通过系数转换）
    #psy=0.000665*pa;
    #pa大气压强
    pa= np.float32((101.3*(np.power(((293-0.0065*alt)/293),5.26))))
    psy=0.000665*pa;
    del alt
    #lamda=2.501-0.002361.*T; %蒸发潜热 MJ/kg
    #psy=0.00101305.*pa/0.622./lamda;
    
    #  PET     
    PET= np.float32((0.408*slope*(Rn-G)+900*psy*U2*(es-ea)/(tas+273.16))/(slope+psy*(1+0.34*U2)))
    
    
    del es,ea,tmax,Rn,U2,pa,psy
    PET[PET<0]=0;  #daily_ETo—小于零的赋值为极小值

    return PET