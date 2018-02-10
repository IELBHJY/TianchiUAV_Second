#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
from matplotlib import pyplot as plt
import skimage.measure
# %pylab inline
import numpy as np
np.set_printoptions(precision=45)

import sys
import time
import pickle as pkl
#%pylab inline
#import tensorflow as tf
import threading

def get_next_points(p):
    next_p=np.array([[p[0],p[1]],[p[0]-1,p[1]],[p[0],p[1]-1],[p[0]+1,p[1]],[p[0],p[1]+1]])
    ind=np.where(next_p[:,0]>547)
    next_p[ind,0]=547
    ind=np.where(next_p[:,1]>420)
    next_p[ind,1]=420
    ind=np.where(next_p[:,0]<0)
    next_p[ind,0]=0
    ind=np.where(next_p[:,1]<0)
    next_p[ind,1]=0
    return (next_p[:,0],next_p[:,1])



def get_rewards(df1,df2,start_id,start_time):

    #risk1=[0.00000001,0.0000001,0.0000002,0.0000004,0.0000008,0.0001,0.002,0.03,0.4,0.5]
    #risk2=[0.00000001,0.0000001,0.0000002,0.0000004,0.0000008,0.0001,0.002,0.03,0.4,0.5]
 
    risk1= [0.00000000001,
            0.0000000001,
            0.00000001,
            0.000001,
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1]
   
    risk2=[ 0.0000000001,
            0.00000001,
            0.000001,
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1]
   
    
    df1['wind']=df1['wind'].map(lambda x : risk1[x])
    df2['rainfall']=df2['rainfall'].map(lambda x : risk2[x])
    reward=[]
    step=0
    for h in range(start_h,end_h):
        step=step+1
        tmp_df1=df1[df1['hour']==h]
        tmp_df2=df2[df2['hour']==h]
        tmp_wind1=tmp_df1.ix[:,'wind'].values.reshape([x_space,y_space])
        tmp_wind2=tmp_df2.ix[:,'rainfall'].values.reshape([x_space,y_space])
        wind=(tmp_wind1+tmp_wind2).tolist()
        reward.extend(wind*30)
    reward=np.array(reward)
    
    reward=reward.reshape([h_space,x_space,y_space])
    reward=reward[start_time:]
    reward[0]=1
    reward[0][start_id[0]][start_id[1]]=-540
    

    reward=-np.array(reward)
    reward=reward[::-1]
    
    return reward 


# 初始价值函数
def get_values(rewards):
    '''
    获取values值
    '''
    
    delta = 0
    step = 1
    zero000=0
    V=np.full_like(rewards,zero000,dtype=np.float64)
    V[-1]=rewards[-1].min()
    V[-1][start_id[0]][start_id[1]]=rewards[-1].max()
    
    
    tmp_v2 = np.full_like(V[1],zero000,dtype=np.float64)
    tmp_v3 = np.full_like(V[1],zero000,dtype=np.float64)
    tmp_v4 = np.full_like(V[1],zero000,dtype=np.float64)
    tmp_v5 = np.full_like(V[1],zero000,dtype=np.float64)
    for i in range(1):
        
        for i in range(V.shape[0]-1,0,-1):
            st=time.time()
            tmp_v2[1:,:]=V[i][:-1,:]
            tmp_v2[0,:]=V[i][0,:]
            tmp_v3[:-1,:]=V[i][1:,:]
            tmp_v3[-1,:]=V[i][-1,:]
            tmp_v4[:,1:]=V[i][:,:-1]
            tmp_v4[:,0]=V[i][:,0]
            tmp_v5[:,:-1]=V[i][:,1:]
            tmp_v5[:,-1]=V[i][:,-1]

            tmp_v=np.array([V[i],tmp_v2,tmp_v3,tmp_v4,tmp_v5]).max(axis=0)

            
            V[i-1]=rewards[i-1]+tmp_v
            print("\rStep {} , delta: {}  spend times：{} ".format(step, delta, time.time() - st), end="")
            step = step + 1
            sys.stdout.flush()
            #if delta < 0.0001:
              #  break
    
    return V


def get_route(V,start_p=[422,265],end_p=[142 - 1, 328 - 1]):
    
    route=[]
    distance=abs(start_p[0]-end_p[0])+abs(start_p[1]-end_p[1])
    if distance>V.shape[0]-1:
        return [[0,0]],0
    start_t=np.argmax(V[:-distance,start_p[0],start_p[1]])
    route_risk=V[start_t,start_p[0],start_p[1]]
 
    tmp_p=start_p
    route.append(np.array(tmp_p).tolist())
    for i in range(start_t+1,h_space):
        #print(tmp_p)
        next_xs,next_ys=get_next_points(tmp_p)
        ind=np.argmax(V[i,next_xs,next_ys])
        
        tmp_p=[next_xs[ind],next_ys[ind]]
        route.append(tmp_p)
        if tmp_p==end_p:
            break
    route=route[::-1]
    ind=route.index(np.array(start_p).tolist())
    #if ind<len(route)-1:
     #   route=route[:ind]
    
    return route,route_risk
def check_route(routes):
    for j in range(10):
        route=routes[j]
        for i in range(1,len(route)):
            d=abs(route[i][0]-route[i-1][0])+abs(route[i][1]-route[i-1][1])
            if d>1:
                print(j)
def show_route(route):
    maps=np.zeros([x_space,y_space])
    maps[np.array(route).T[0],np.array(route).T[1]]=1
    plt.imshow(maps)

def get_ts():
    '''
    所有时刻列表
    '''
    ts=[]
    for i in range(18*30):
        hh=np.linspace(3,20,18)[int(i/30)]
        mm=i%30*2
        hh=str(int(hh))
        mm=str(mm)
        if len(hh)==1:
            hh='0'+hh
        if len(mm)==1:
            mm='0'+mm
        t=hh+':'+mm
        ts.append(t)
    return ts

# 全局变量
end_id = [363 - 1, 237 - 1]
start_id = [142 - 1, 328 - 1]
start_id_true=[142,328]
x_space = 548
y_space = 421

h_space = 540
start_h = 3
end_h=21



date=sys.argv[1]
date=int(date)
citys=pd.read_csv('../data/CityData.csv')
citys=citys.loc[:,['xid','yid']][1:]
citys=np.array(citys)-1
ts=get_ts()
#源文件某天的按x,y排序的风力值
print('读取文件...')
df11=pd.read_csv('../ml_result/%d_wind' %date)
df22=pd.read_csv('../ml_result/%d_rainfall' %date)
df11 = df11[df11['date_id'] == date]
df22 = df22[df11['date_id'] == date]
print('生成奖励列表...')

routes_risk=[]
result_df=[]
for start_time in range(0,95):
    start_time=start_time*5
    print('\n start_time: {}'.format(start_time))
           
    rewards = get_rewards(df11.copy(),df22.copy(),start_id,start_time)
    V=get_values(rewards)


    #print('生成路线...')
    routes=[]
    
    for city_id,ct in enumerate(citys):

        city_id=city_id+1
        r,r_r=get_route(V,ct)
        r=np.array(r)+1
        
        if r[0][0]==start_id_true[0] and r[0][1]==start_id_true[1]:
            routes.append(r)
            routes_risk.append([city_id,start_time,r_r])

    #print('生成结果中...')
    if len(routes)<1:
        continue
    
    for r in routes:
        city_id=(citys+1).tolist().index(r[-1].tolist())
        city_id=city_id+1
        tmp_df=pd.DataFrame([city_id]*len(r))
        tmp_df['date_id']=date
        tmp_df['hour']=ts[start_time:len(r)+start_time]
        tmp_df['x']=r[:,0]
        tmp_df['y']=r[:,1]
        tmp_df['start_time']=start_time
        result_df.append(tmp_df)

result_df=pd.concat(result_df)
print('保存结果中...')
routes_risk_df=pd.DataFrame(routes_risk)
routes_risk_df.to_csv('../routes/%d_risk' %date,index=None)
result_df.to_csv('../routes/%d' %date,index=None)
print('完成！')
