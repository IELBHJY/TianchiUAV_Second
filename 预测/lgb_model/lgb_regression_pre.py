
import pandas as pd
import numpy as np
import os
import gc
import lightgbm as lgb
from scipy.stats import hmean
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
os.chdir("/Users/apple/Desktop/TianchiUAV/train20171205")
df_tr=[]
df_te=[]
for i in [1,2,3,4,5]: 
    df_train=pd.read_csv("trainday"+str(i)+"_windall.csv")
    df_test=pd.read_csv("Trueday"+str(i)+".csv")
    df_tr.append(df_train)
    df_te.append(df_test)
df_train=pd.concat(df_tr,ignore_index=True)
df_test=pd.concat(df_te,ignore_index=True)
df_train=df_train.loc[:,['xid','yid','hour','wind1','wind2','wind3','wind4','wind5','wind6','wind7','wind8','wind9','wind10','wind_avg','wind_max','wind_min']]
#x_test=df_test.loc[:,['wind1','wind2','wind3','wind4','wind5','wind6','wind7','wind8','wind9','wind10','wind_avg','wind_max','wind_min']]
y=df_test['wind']
wind_list=['wind1','wind2','wind3','wind4','wind5','wind6','wind7','wind8','wind9','wind10']
os.chdir("/Users/apple/Desktop/TianchiUAV/test20171205")
df_test1=[]
for i in [10]:
    test_x=pd.read_csv("test"+str(i)+"_windall.csv")
    df_test1.append(test_x)
test_x=pd.concat(df_test1,ignore_index=True)
test_x=test_x.loc[:,['xid','yid','hour','wind1','wind2','wind3','wind4','wind5','wind6','wind7','wind8','wind9','wind10','wind_avg','wind_max','wind_min']]
start_time = time.time()
# 平均数，总位数，最大值，最小值，方差
def get_feature(df):
    df['wind_median']=df[wind_list].median(axis=1)
    df['wind_var']=df[wind_list].var(axis=1)
    # df['wind_hmean']=(df[wind_list]+1).apply(hmean,axis=1)
    # df['wind_mean2']=(df[wind_list].sum(axis=1)-df['wind_max']-df['wind_min'])/8
    # df=df.drop('wind_avg',axis=1)
    return df
X=get_feature(df_train)
gc.collect()

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.14, random_state = 144) 
d_train = lgb.Dataset(train_X, label=train_y)
d_valid = lgb.Dataset(valid_X, label=valid_y)
watchlist = [d_train, d_valid]  
params = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4
}
params2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 130,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 4
}
model = lgb.train(params, train_set=d_train, num_boost_round=7500, valid_sets=watchlist, \
early_stopping_rounds=1000, verbose_eval=1000) 
X_test=get_feature(test_x)
predsL = model.predict(X_test)
print('[{}] Predict lgb 1 completed.'.format(time.time() - start_time))
train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101) 
d_train2 = lgb.Dataset(train_X2, label=train_y2)
d_valid2 = lgb.Dataset(valid_X2, label=valid_y2)
watchlist2 = [d_train2, d_valid2]
model = lgb.train(params2, train_set=d_train2, num_boost_round=6000, valid_sets=watchlist2, \
early_stopping_rounds=500, verbose_eval=500) 
predsL2 = model.predict(X_test)
print('[{}] Predict lgb 2 completed.'.format(time.time() - start_time))
preds = (predsL*0.5 + predsL2*0.5)
X_test['pred']=preds
X_test.to_csv('/Users/apple/Desktop/20180125/pred_day10.csv',index=False)
#submission['price'] = np.expm1(preds)
#submission.to_csv("/Users/apple/Desktop/submission_ridge_2xlgbm.csv", index=False)
'''
#回归问题lgb的metric评价标准 l1,l2,rmse
lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting': 'gbdt',
        'learning_rate': 0.1,  # small learn rate, large number of iterations
        'num_leaves': 2 ** 5,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'feature_fraction': 0.9,
        'max_bin': 100,
        'max_depth': 5,
        'verbose': -1,
    }
ltrain=lgb.Dataset(x_train,y_train)
# cv_results=lgb.cv(lgb_params,ltrain,500,nfold=5,shuffle=True,early_stopping_rounds=50)
# plt.plot(cv_results['binary_error-mean'])
# print("###CV结果:{}".format(cv_results['binary_error-mean'][-1:]))
# num_boost_rounds = len(cv_results['binary_error-mean'])
num_boost_rounds=100
model = lgb.train(lgb_params, ltrain, num_boost_round=num_boost_rounds,verbose_eval=50)
x_test=get_feature(x_test)
y_pred=model.predict(x_test)
x_test['y_pred']=y_pred
x_test.to_csv("E://tianchi//data//test8_lgb.csv",index=False)
'''
'''
y_test['y_pred']=y_pred
count=0
for i in y_test.index:
    if y_test.loc[i,'y_pred']<15 and y_test.loc[i,'wind']<15:
        count+=1
    if y_test.loc[i,'y_pred']>=15 and y_test.loc[i,'wind']>=15:
        count+=1
print(count/len(y_test.index))
        
#df_test[['xid','yid','date_id','hour','y_pred']].to_csv("results/lgb_regression_out.csv.gz",index=False, compression='gzip')

#runfile("/home/viczyf/kaggle/weilaiqixiang/model/lgb_regression_pre.py")
'''








