
# coding: utf-8

# # 1. Packages

# In[1]:

import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from scipy import stats
import scipy as sp
import scipy.ndimage
get_ipython().magic(u'matplotlib inline')


# # 2. Preparation 

# In[2]:

def load_source_data(path):
    df_source = pd.read_csv(path)
    return df_source


def prepare_lgb_data(df_source):
    feature_name = ['xid', 'yid', 'hour', 'model_1', 'model_2', 'model_3', 'model_4',
                    'model_5', 'model_6', 'model_7', 'model_8', 'model_9', 'model_10']
    X_predict = df_source[feature_name]
    return X_predict


def load_lgb_model(model_file_name):
    if not os.path.exists(model_file_name):
        raise ValueError('lgb model file is not exists.')
    lgb_model = None
    try:
        lgb_model = lgb.Booster(model_file=model_file_name)
    except:
        raise ValueError('lgb model file is not valid.')
    return lgb_model


# # 3. Predicting

# In[ ]:




# # 4. Save predictions

# In[3]:

def convert_predict_to_prob(Y_predict, threshold=15, mean=0, var=1):
    Y_predict_prob = stats.norm(mean, var).cdf(threshold - Y_predict)
    return Y_predict_prob


def cache_Y_predict(Y_predict, path):
    file_name = path + 'Y_predict_cache.csv'
    df_Y_predict = pd.DataFrame(Y_predict, columns=['Y_predict'])
    df_Y_predict.to_csv(file_name, index=False, sep=',')


def save_prediction_danger(df_source, Y_predict, path, threshold=15):
    df = df_source.copy()
    df['wind_predict'] = pd.Series(Y_predict, index=df.index)
    for date_id in range(6, 11):
        file_name = path + '/In-situMeasurementforTraining_date' + str(date_id) + '.csv'
        df_by_date = df[(df['date_id'] == date_id) & (df['wind_predict'] >= threshold)]
        df_by_date.to_csv(file_name, columns=['xid', 'yid', 'hour'], index=False, sep=',', header=False)


def save_prediction_prob(df_source, Y_predict, path, threshold=15, mean=0, var=1):
    Y_predict_prob = convert_predict_to_prob(Y_predict, threshold, mean, var)
    df = df_source.copy()
    df['wind_predict_prob'] = pd.Series(Y_predict_prob, index=df.index)
    for date_id in range(6, 11):
        file_name = path + '/In-situMeasurementforTraining_date' + str(date_id) + '.csv'
        df_by_date = df[(df['date_id'] == date_id)]
        df_by_date.to_csv(file_name, columns=['xid', 'yid', 'hour', 'wind_predict_prob'], index=False, sep=',', header=False)


# # 5. Usage

# In[5]:

# load data and model
df_source = load_source_data('../Data/DataForTesting.csv')
#Y_predict = pd.read_csv('../DataCache/regression_rmse_21041_cache/Y_predict_cache.csv')
#Y_predict = pd.read_csv('../DataCache/xgb/xgb_predict.csv')
#lgb_model = load_lgb_model('gbdt_regression_rmse')

df = df_source.copy()
df['model_mean'] = df[['model_'+str(model_id) for model_id in range(1,11)]].mean(1)


# In[7]:

# predict
X_predict = prepare_lgb_data(df_source)
Y_predict = lgb_model.predict(X_predict)

# cache
path = '../DataCache/regression_rmse_17575_cache/'
cache_Y_predict(Y_predict, path)


# In[8]:

# save prediction
path = '../DataOut/No_training_10_models_mean_14P5/'


save_prediction_danger(df, df['model_mean'], path, threshold=14.5)
#save_prediction_danger(df_source, Y_predict, path, threshold=15)
#save_prediction_danger(df_source, Y_predict['xgb_predict'], path, threshold=15)

threshold = 15
mean = 0.0032
var = 0.6053
#save_prediction_prob(df_source, Y_predict, path, threshold, mean, var)
#save_prediction_prob(df_source, Y_predict['xgb_predict'], path, threshold, mean, var)


# # 6. Smoothing from Cache

# In[7]:

# load Y_predict_cache
#Y_predict_cache = pd.read_csv('../DataCache/regression_rmse_21041_cache/Y_predict_cache.csv')
#data_for_testing = pd.read_csv('../Data/DataForTesting.csv')
#data_for_testing['Y_predict'] = Y_predict_cache
#df_source['Y_predict'] = Y_predict_cache

wind_data = df_source.copy()
wind_data['Y_predict'] = Y_predict['xgb_predict']


threshold = 15
mean = 0.0032
var = 0.6053
xmin,xmax,ymin,ymax = 1,548,1,421
#path = '../DataOut/regression_rmse_17575_'+str(threshold)+'_prob_smooth_Convolve/'
path = '../DataOut/xgb_regression_rmse_15_prob_GaussianSmooth/'

# Gaussian Smoothing
smooth_sigma = 1

# Convolve Smoothing
weight_matrix = np.ones((3,3))/9

for day in range(6,11):
    day_data = pd.DataFrame()   
    for hour in range(3,21):
        wind = wind_data[['xid','yid','hour','Y_predict']][(wind_data['hour']== hour) & (wind_data['date_id']== day)]
        wind = wind.sort_values(by= ['yid','xid'])
        
        
        # Y_predict to matrix
        wind_matrix = wind[['Y_predict']].values.reshape(ymax,xmax)
        
        # matrix by Gaussian smooth 
        wind_matrix_smooth = sp.ndimage.filters.gaussian_filter(wind_matrix, [smooth_sigma, smooth_sigma], mode='constant')
        
        # matrix by Convole smooth
        #wind_matrix_smooth = sp.ndimage.filters.convolve(wind_matrix, weight_matrix, mode='constant')
        
        # matrix back to Y_predict
        Y_predict_smooth = wind_matrix_smooth.reshape(len(wind['Y_predict']), order='C')
        
        
        # prob
        Y_predict_prob = convert_predict_to_prob(Y_predict_smooth, threshold, mean, var)
        wind['Y_predict_prob'] = Y_predict_prob
        day_data = day_data.append(wind[['xid','yid','hour','Y_predict_prob']], ignore_index=True)
        
        #danger
        #wind['Y_predict'] = Y_predict_smooth
        #day_data = day_data.append(wind[['xid','yid','hour','Y_predict']], ignore_index=True)
        
    
    file_name = path + 'In-situMeasurementforTraining_date' + str(day) + '.csv'
    day_data.to_csv(file_name, columns=['xid', 'yid', 'hour', 'Y_predict_prob'], index=False, sep=',', header=False)
    #day_data[(day_data['Y_predict'] >= threshold)].to_csv(file_name, columns=['xid', 'yid', 'hour'], index=False, sep=',', header=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[2]:

df_source = pd.read_csv('./Data/DataForTesting.csv', header=0, sep=',')
df=df_source.copy()


# ### 特征值规范化

# In[3]:

df['xid']=df['xid']/548
df['yid']=df['yid']/421
for i in range(1,11):
    df['model_'+str(i)]=df['model_'+str(i)]/20
df['hour']=(df['hour']-9)/11
df.head(5)


# ### lightGBM 模型加载，输入预测数据，输出预测结果

# In[13]:

x_predict = df.drop(['date_id'], axis=1).values #

bst = lgb.Booster(model_file='model.txt')

y_predict = bst.predict(x_predict)


# ### 重新读入未规范化的原数据集

# In[14]:

#d = df_source.copy()
d = d.drop(['prob'], axis=1)


# ### 把预测结果（风速值）作为新的一列加附到原数据集

# In[15]:

d['prob'] = pd.Series(y_predict, index=df.index)
d.head(15)


# ### 把数据集写入 CSV 文件

# In[16]:

for date_id in range(6, 11):
    df_by_date = d[['xid', 'yid', 'hour','prob']][(d['date_id'] == date_id)] #& (d['wind_predict'] >= 0.75)
    df_by_date.to_csv('./DataOutput/DataOutput-prob-20-date2-Nor/In-situMeasurementforTraining_date' + str(date_id) + '.csv', index=False, sep=',', header=False)


# In[21]:

result=[]
wind_split=20
#df['wind_predict']=df['wind_predict']/20
for date_id in range(6,11):
    result.append([len(df[(df['date_id']==date_id)&(df['wind_predict']<wind_split)]),len(df[(df['date_id']==date_id)&(df['wind_predict']>=wind_split)])])
result


# In[19]:

df_by_date.head(5)


# In[ ]:



