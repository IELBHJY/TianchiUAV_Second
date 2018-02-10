
# coding: utf-8

# # 1. Packages

# In[1]:

import lightgbm as lgb
import pandas as pd
import numpy as np
import math
import scipy as sp
import scipy.ndimage
from scipy import stats
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
get_ipython().magic(u'matplotlib inline')


# # 2. Preparation 

# In[2]:

def load_source_data(path):
    df_source = pd.read_csv(path)
    return df_source


def deleteBadSample(df_source, df_error, threshold=5):
    df_error = df_error[['xid','yid','date_id','hour']][df_error['error_abs']>=threshold]
    keys = ['xid','yid','date_id','hour']
    index1 = df_source.set_index(keys).index
    index2 = df_error.set_index(keys).index
    df_clear = df_source[~(index1.isin(index2))]
    return df_clear


def split_data_randomly(df_source, train_percent=0.98, dev_percent=0.01, seed=None):
    np.random.seed(seed)
    df_source = df_source.reset_index(drop=True)
    perm = np.random.permutation(df_source.index)
    m = len(df_source)
    train_end = int(train_percent * m)
    dev_end = int(dev_percent * m) + train_end
    train = df_source.iloc[perm[:train_end]]
    dev = df_source.iloc[perm[train_end:dev_end]]
    test = df_source.iloc[perm[dev_end:]]
    return train, dev, test


def prepare_lgb_data(df_source):
    feature_name = ['xid', 'yid', 'hour', 'model_1', 'model_2', 'model_3', 'model_4',
                    'model_5', 'model_6', 'model_7', 'model_8', 'model_9', 'model_10']
    label_name = 'real_wind'
    train_data, dev_data, test_data = split_data_randomly(df_source)
    X_train = train_data[feature_name]
    Y_train = train_data[label_name]
    W_train = None  # train_data[label_name].apply(lambda wind: 1 if wind < 20 else 1)
    X_dev = dev_data[feature_name]
    Y_dev = dev_data[label_name]
    W_dev = None  # test_data[label_name].apply(lambda wind: 1 if wind < 20 else 1)
    X_test = test_data[feature_name]
    Y_test = test_data[label_name]
    lgb_data = {
        'train_data': train_data, 'dev_data': dev_data, 'test_data': test_data,
        'X_train': X_train, 'Y_train': Y_train, 'W_train': W_train,
        'X_dev': X_dev, 'Y_dev': Y_dev, 'W_dev': W_dev,
        'X_test': X_test, 'Y_test': Y_test
    }
    return lgb_data


def initialize_lgb_parameters(boosting_type='gbdt', objective='regression', metric='rmse', learning_rate=0.1):
    boosting_typec_choices = {'gbdt', 'rf'}
    objective_choices = {'regression', 'regression_l1', 'regression_l2', 'binary'}
    metric_choices = {'l1', 'l2', 'l2_root', 'mse', 'mae', 'rmse', 'auc', 'binary_logloss', 'binary_error'}
    if boosting_type not in boosting_typec_choices or objective not in objective_choices or metric not in metric_choices:
        raise ValueError('Wrong objective or metric.')
        return None
    lgb_params = {
        'boosting_type': boosting_type,
        'objective': objective,
        'metric': metric,
        'learning_rate': learning_rate,
        'verbose': 0
    }
    return lgb_params


# # 3. Training

# In[3]:

def train_lgb(lgb_params, lgb_data, lgb_round=50000):
    lgb_train_data = lgb.Dataset(lgb_data['X_train'], lgb_data['Y_train'], weight=lgb_data['W_train'])
    lgb_dev_data = lgb.Dataset(lgb_data['X_dev'], lgb_data['Y_dev'], weight=lgb_data['W_dev'], reference=lgb_train_data)
    lgb_feature_name = list(lgb_data['X_train'].columns.values)
    lgb_model = lgb.train(params=lgb_params,
                          train_set=lgb_train_data,
                          num_boost_round=lgb_round,
                          valid_sets=lgb_dev_data,
                          feature_name=lgb_feature_name,
                          early_stopping_rounds=10)
    lgb.plot_importance(lgb_model, importance_type='gain', ignore_zero=False, figsize=(10, 6))
    model_filename = lgb_params['boosting_type'] + '_' + lgb_params['objective'] + '_' + str(lgb_params['metric'])
    lgb_model.save_model(model_filename)
    return lgb_model


# # 4. Evalutaion

# In[4]:

def print_lgb_error(Y_real, Y_predict):
    df_compare = pd.DataFrame({'Y_real': Y_real, 'Y_predict': Y_predict}).sort_values(by='Y_real')
    df_compare['error']= df_compare['Y_real'] - df_compare['Y_predict']
    df_compare['error'].hist(bins=1000, figsize=(12,8))
    print ('error mean: ', df_compare['error'].mean())
    print ('error var: ', df_compare['error'].var())
    print ('error rmse: ', metrics.mean_squared_error(Y_real, Y_predict)** 0.5)
    return df_compare


def convert_predict_to_prob(Y_predict, threshold=15, mean=0, var=1):
    Y_predict_prob = stats.norm(mean, var).cdf(threshold - Y_predict)
    return Y_predict_prob


def save_prediction_danger(df_source, Y_predict, path, threshold=15):
    df = df_source.copy()
    df['wind_predict'] = pd.Series(Y_predict, index=df.index)
    for date_id in range(1, 6):
        file_name = path + '/In-situMeasurementforTraining_date' + str(date_id) + '.csv'
        df_by_date = df[(df['date_id'] == date_id) & (df['wind_predict'] >= threshold)]
        df_by_date.to_csv(file_name, columns=['xid', 'yid', 'hour'], index=False, sep=',', header=False)


def error_count(error, n=100):
    range_len = (error.max() - error.min()) * 1.0 / n
    error_range = pd.Series(np.linspace(error.min(), error.max(), n))
    count = error_range.apply(lambda x: np.sum((error >= x) & (error < x + range_len)))
    return pd.DataFrame({'error': error_range, 'error_count': count})


def binary_evaluation(Y_real, Y_predict, threshold=15):
    # the sample is a positive sample(1) if it's wind < threshold, and negative sample(0) otherwise
    Y_predict_binary = np.where(Y_predict < threshold, 1, 0)
    Y_real_binary = np.where(Y_real < threshold, 1, 0)

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(0, len(Y_predict_binary)):
        if Y_real_binary[i] == 1:
            if Y_predict_binary[i] == 0:
                FN += 1
            else:
                TP += 1
        else:
            if Y_predict_binary[i] == 0:
                TN += 1
            else:
                FP += 1

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [TPR, FPR]


# # 5. Usage

# In[16]:

# load data set and clear bad sample
#df_source = load_source_data('../Data/DataForTraining.csv')
#df_error = load_source_data('../DataCache/regression_rmse_37505_cache/total_cache.csv')
#df_clear = deleteBadSample(df_source, df_error, threshold=3)
df_source = load_source_data('../Data/DataForTesting.csv')


# In[24]:

#len(df_error),len(df_clear),len(df_error)-len(df_clear)
#df_error.head()
len(df_source[df_source['model_2']==0])


# In[78]:

#282	296	5	15
xid,yid,date_id,hour=282,296,5,15
#df_clear[(df_clear['xid']==xid)&(df_clear['yid']==yid)&(df_clear['date_id']==date_id)&(df_clear['hour']==hour)]
df_error[(df_error['xid']==xid)&(df_error['yid']==yid)&(df_error['date_id']==date_id)&(df_error['hour']==hour)]


# In[ ]:

# train begin
lgb_data = prepare_lgb_data(df_clear)
lgb_params = initialize_lgb_parameters('gbdt', 'regression', 'rmse', 0.5)
lgb_model = train_lgb(lgb_params, lgb_data, lgb_round=20000)


# # 6. Training, Validating and Testing Error, TPR and FPR

# In[71]:

# training data prediction
Y_train_predict = lgb_model.predict(lgb_data['X_train'])

# cache prediction
train_cache = lgb_data['train_data'].copy()
train_cache['Y_train'] = lgb_data['Y_train'].copy()
train_cache['Y_train_predict'] = Y_train_predict
train_cache.to_csv('../DataCache/regression_rmse_20000_cache/train_cache.csv', index=False, sep=',')

# training data prediction error
lgb_train_compare = print_lgb_error(lgb_data['Y_train'], Y_train_predict)


# In[70]:

# validating data prediction
Y_dev_predict = lgb_model.predict(lgb_data['X_dev'])

# cache prediction
dev_cache = lgb_data['dev_data'].copy()
dev_cache['Y_dev'] = lgb_data['Y_dev'].copy()
dev_cache['Y_dev_predict'] = Y_dev_predict
dev_cache.to_csv('../DataCache/regression_rmse_20000_cache/dev_cache.csv', index=False, sep=',')

# validating data prediction error
lgb_dev_compare = print_lgb_error(lgb_data['Y_dev'], Y_dev_predict)


# In[69]:

# testing error
Y_test_predict = lgb_model.predict(lgb_data['X_test'])

# cache prediction
test_cache = lgb_data['test_data'].copy()
test_cache['Y_test'] = lgb_data['Y_test'].copy()
test_cache['Y_test_predict'] = Y_test_predict
test_cache.to_csv('../DataCache/regression_rmse_20000_cache/test_cache.csv', index=False, sep=',')

lgb_test_compare = print_lgb_error(lgb_data['Y_test'], Y_test_predict)


# In[74]:

[TPR, FPR] = binary_evaluation(lgb_data['Y_test'], Y_test_predict, threshold=13)
[TPR, FPR]


# # 7. Error Analysis on Testing Data

# In[30]:

test_cache = pd.read_csv('../DataCache/regression_rmse_37505_cache/test_cache.csv')
#Y_test_predict_prob = convert_predict_to_prob(test_cache['Y_test_predict'], 15, 0.02, 3)
#test_cache['Y_test_predict_prob'] = Y_test_predict_prob
test_cache['Y_error'] = abs(test_cache['Y_test'] - test_cache['Y_test_predict'])


# In[21]:

test_cache[['Y_error', 'Y_test', 'Y_test_predict', 
            'model_7', 'model_3', 'model_1', 'model_5',
            'model_6', 'model_2', 'model_10', 'model_9', 'model_8', 
            'model_4','xid','yid','date_id','hour']].round(3).sort_values(by='Y_error', ascending=False)

#[(test_cache['Y_test_predict']<15)&(test_cache['Y_test']>15)]


# In[29]:

test_cache[['xid','yid']][test_cache['Y_error']>=3].hist()


# In[25]:

test_cache['Y_test_error'] = test_cache['Y_test'] - test_cache['Y_test_predict']
test_cache[['Y_test', 'Y_test_predict', 'Y_test_error', 'Y_test_predict_prob']].sort_values(by='Y_test_error', ascending=False)
#[(test_cache['Y_test_predict_prob']==1)]


# # 8. Plot Total Error

# In[75]:

# combine train, dev, and test error

total = pd.DataFrame()

train_cache = pd.read_csv('../DataCache/regression_rmse_20000_cache/train_cache.csv')
train_cache = train_cache[['xid', 'yid', 'date_id', 'hour', 
                           'model_1', 'model_2', 'model_3', 'model_4','model_5', 'model_6', 'model_7', 
                           'model_8', 'model_9', 'model_10', 'real_wind', 'Y_train_predict']]
train_cache.rename(columns={'Y_train_predict': 'predict_wind',}, inplace=True)
train_cache.round({'predict_wind': 4})

dev_cache = pd.read_csv('../DataCache/regression_rmse_20000_cache/dev_cache.csv')
dev_cache = dev_cache[['xid', 'yid', 'date_id', 'hour', 
                       'model_1', 'model_2', 'model_3', 'model_4','model_5', 'model_6', 'model_7', 
                       'model_8', 'model_9', 'model_10', 'real_wind', 'Y_dev_predict']]
dev_cache.rename(columns={'Y_dev_predict': 'predict_wind',}, inplace=True)
dev_cache.round({'predict_wind': 4})

test_cache = pd.read_csv('../DataCache/regression_rmse_20000_cache/test_cache.csv')
test_cache = test_cache[['xid', 'yid', 'date_id', 'hour', 
                       'model_1', 'model_2', 'model_3', 'model_4','model_5', 'model_6', 'model_7', 
                       'model_8', 'model_9', 'model_10', 'real_wind', 'Y_test_predict']]
test_cache.rename(columns={'Y_test_predict': 'predict_wind',}, inplace=True)
test_cache.round({'predict_wind': 4})

total = total.append(train_cache, ignore_index=True)
total = total.append(dev_cache, ignore_index=True)
total = total.append(test_cache, ignore_index=True)
total['error'] = total['real_wind'] - total['predict_wind']
total['error_abs'] = abs(total['error'])
total = total.sort_values(by=['error_abs'], ascending=False).round(4)


# In[83]:

# cache
#total.to_csv('../DataCache/regression_rmse_20000_cache/total_cache.csv', index=False, sep=',')
ltotal = pd.read_csv('../DataCache/regression_rmse_17575_cache/total_cache.csv')


# In[94]:

len(total[total['error_abs']>=3])
#len(ltotal)+1911,len(total)+41928


# In[34]:

total[['error_abs']][total['error_abs']>=5].hist(figsize=(12,8), bin=500)


# In[11]:

# plot error
data_id = 3
hour = 10
yid_range, xid_range = total['yid'].max(), total['xid'].max()

error = total[['xid', 'yid', 'error']][(total['date_id']== data_id) & (total['hour']== hour)]
error['error'] = abs(error['error'])
error = error.sort_values(by=['yid', 'xid'])
error_matrix = error[['error']].values.reshape(yid_range, xid_range)

plt.figure(figsize=(15, 12))
plt.imshow(error_matrix, cmap=cm.coolwarm)
plt.colorbar()
plt.show()


# In[4]:

Y_predict_cache = pd.read_csv('../DataCache/regression_rmse_21041_cache/Y_predict_cache.csv')
data_for_testing = pd.read_csv('../Data/DataForTesting.csv')
Y_predict_cache.head()


# In[20]:




# In[5]:

data_for_testing['Y_predict'] = Y_predict_cache


# In[6]:

data_for_testing.head()


# In[7]:

xmin,xmax,ymin,ymax = 1,548,1,421
wind_data = data_for_testing
day = 7
hour = 10
wind = wind_data[(wind_data['hour']== hour) & (wind_data['date_id']== day)]
wind = wind.sort_values(by= ['yid','xid'])[['Y_predict']].values.reshape(ymax,xmax)
plt.figure(figsize=(30,10))
plt.imshow(wind, cmap=cm.coolwarm)
plt.colorbar()


# In[32]:

day = 6
hour = 9
wind = wind_data[(wind_data['hour']== hour) & (wind_data['date_id']== day)]
wind = wind.sort_values(by= ['yid','xid'])[['Y_predict']].values.reshape(ymax,xmax)
x_range = np.arange(xmin,xmax+1,1)
y_range = np.arange(ymin,ymax+1,1)
xs , ys = np.meshgrid(x_range,y_range)
fig = plt.figure()
ax = Axes3D(fig)
zs = wind
surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# In[9]:

#wind_smooth = sp.ndimage.filters.gaussian_filter(wind, [0.5,0.5], mode='constant')
wind_smooth = sp.ndimage.filters.convolve(wind, np.ones((3,3))/9, mode='constant')
plt.figure(figsize=(30,10))
plt.imshow(wind_smooth, cmap=cm.coolwarm)
plt.colorbar()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




#  $C_i = \sum_j{I_{i+j-k} W_j}$

# ### （1） 加入平均值特征

# In[3]:

df['model_mean']=df[['model_1','model_2','model_3','model_4','model_5',
                     'model_6','model_7','model_8','model_9','model_10']].mean(1)


# ### （2） 加入方差特征

# In[150]:

df['model_var']=df[['model_1','model_2','model_3','model_4','model_5',
                    'model_6','model_7','model_8','model_9','model_10']].var(axis=1,ddof=0)


# ### （3）加入左右区域各10列模型值特征

# In[18]:

#df_test=df.head(1000)
#new_xid=df_test['xid']-1
df_test['xid']=new_xid


# ### 特征值规范化

# In[3]:

df['xid']=df['xid']/548
df['yid']=df['yid']/421
for i in range(1,11):
    df['model_'+str(i)]=df['model_'+str(i)]/20
#df['model_mean']=df['model_mean']/20
df['hour']=(df['hour']-9)/11
df.head(5)


# ### 划分训练集和测试集

# In[77]:

df_train = df[(df['date_id']==1)|(df['date_id']==3)|(df['date_id']==4)|(df['date_id']==5)]
df_test = df[df['date_id'] == 2]


# ### 划分X矩阵和Y向量，正样本和负样本

# In[99]:

y_train = np.where(df_train['real_wind']>=15,0,1) #df_train['wind'].values/20
x_train = df_train.drop(['real_wind','date_id'], axis=1).values #,'xid','yid','hour'

y_test = np.where(df_test['real_wind']>=15,0,1) #df_test['wind'].values/20
x_test = df_test.drop(['real_wind','date_id'], axis=1).values #,'xid','yid','hour'


# ### 正负样本权重

# In[100]:

w_train = pd.DataFrame( np.where(df_train['real_wind']>=20,1,1) )[0]
w_test = pd.DataFrame( np.where(df_test['real_wind']>=20,1,1) )[0]


# ### 特征名称

# In[101]:

num_train, num_feature = x_train.shape
feature_name = list(df_train.columns.values)
feature_name.remove('real_wind')
feature_name.remove('date_id')
#feature_name.remove('xid')
#feature_name.remove('yid')
#feature_name.remove('hour')


# ### lightGBM 参数设置

# In[102]:

lgb_train = lgb.Dataset(x_train, y_train ,weight=w_train) #free_raw_data=False
lgb_eval = lgb.Dataset(x_test, y_test, weight=w_test, reference=lgb_train) #free_raw_data=False
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',#regression
    'metric': 'binary_logloss',# mse 
    'learning_rate': 0.05, 
    #'num_leaves': 31,
    #'feature_fraction': 0.9,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 5,
    'verbose': 0
}


# ### lightGBM 训练并存储模型

# In[103]:

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=61,
                valid_sets=lgb_eval,
                feature_name=feature_name,
                early_stopping_rounds=10
               )
gbm.save_model('model.txt')


# ### lightGBM 特征信息图

# In[104]:

lgb.plot_importance(gbm, importance_type='gain', ignore_zero=False, figsize=(10, 6))


# ### lightGBM 模型加载，输入测试集进行预测

# In[105]:

#bst = lgb.Booster(model_file='model.txt')

y_predict = gbm.predict(x_test,num_iteration=gbm.best_iteration)


# ### 分析训练效果（将预测的第五天风速y_predict与真实的第五天风速y_test对比）

# In[214]:

-sum(y_test*np.log(y_predict)+(1-y_test)*np.log(1-y_predict))/len(y_test)


# In[106]:

y_test,y_predict


# In[107]:

df_test['y_predict'] = y_predict


# In[109]:

df_test[['real_wind','y_predict']][
     (19<df_test['real_wind'])
    &(df_test['real_wind']<20)
    &(0.8<df_test['y_predict'])
    #&(df_test['y_predict']<20)
]
#df_test.head(15)


# In[110]:

df_source.loc[[12379,14020,14569]]


# In[111]:

df_s=pd.read_csv('./Data/ForecastDataforTraining.csv', header=0, sep=',')


# In[115]:

xid=51
yid=371
date_id=2
hour=14


# In[116]:

df_source[(df_source['xid'] == xid)&(df_source['yid'] == yid)&(df_source['date_id'] == date_id)&(df_source['hour'] == hour)]


# In[118]:

df_s[(df_s['xid'] == xid)&(df_s['yid'] == yid)&(df_s['date_id'] == date_id)&(df_s['hour'] == hour)].sort_values(by='realization', ascending=True)


# In[120]:

df_r=pd.read_csv('./Data/In-situMeasurementforTraining.csv', header=0, sep=',')


# In[121]:

df_r[(df_r['xid'] == xid)&(df_r['yid'] == yid)&(df_r['date_id'] == date_id)&(df_r['hour'] == hour)]


# In[36]:

y_test_r=y_test*20
y_predict_r=y_predict*20
y_test_binary = np.where(y_test_r>=20,1,0) #[1 if x >= 20 else 0 for x in y_test]
y_predict_binary = np.where(y_predict_r>=14,1,0) #[1 if x >= 20 else 0 for x in y_predict]

d_to_s , s_to_d ,d_to_d, s_to_s = 0,0,0,0
for i in range(0, len(y_predict_binary)):   
    if y_test_binary[i] == 1:
        if y_predict_binary[i] == 0:
            d_to_s +=1
        else :
            d_to_d +=1
    else:
        if y_predict_binary[i] == 0:
            s_to_s +=1
        else:
            s_to_d +=1


# In[37]:

d_to_s, s_to_d, d_to_d, s_to_s,len(y_predict_binary), len(y_test_binary[y_test_binary==0]), len(y_test_binary[y_test_binary==1])


# In[ ]:




# In[2]:

df_source = pd.read_csv('../Data/DataForTraining.csv')


# In[3]:

df_source['model_mean'] = df_source[['model_'+str(model_id) for model_id in range(1,11)]].mean(1)


# In[7]:

df_source['error'] = df_source['real_wind'] - df_source['model_mean']


# In[9]:

df = df_source[df_source['date_id']==5].copy()


# In[10]:

df['error'].mean()


# In[11]:

df['error'].var()


# In[12]:

df['error'].hist(bins=1000, figsize=(12,8))


# In[ ]:

df[abs(df['error'])>12].sort_values(by='error')


# In[1]:




# In[ ]:



