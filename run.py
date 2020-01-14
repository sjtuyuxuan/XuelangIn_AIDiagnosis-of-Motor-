import pandas as pd
import numpy as np
import glob
import os
import lightgbm as lgb
####################


#step1:Yong_train_test_data
from method import GetYongData,lgb_yong
train_df1 = GetYongData('data/Motor_tain/Positive/',1)
train_df0 = GetYongData('data/Motor_tain/Negative/',0)
train = pd.concat([train_df1,train_df0])
train.to_csv('data/train/train_yong.csv',index=None)

test = GetYongData('data/Motor_testP/',0)
test.to_csv('data/test/test_yong.csv',index=None)

#step2:tutu_model_predict
from method import get_unique_ID,get_train_data,get_test_data,lgb_model,data_augmentation
#生成train数据
Motor_train_p_ID = get_unique_ID('data/Motor_tain/Positive')
Motor_train_n_ID = get_unique_ID('data/Motor_tain/Negative')

data_augmentation(15, Motor_train_p_ID, Pos_Neg='Positive')

Motor_train_p_ID = get_unique_ID('data/Motor_tain_tutu/Positive')

X_train_p, y_train_p = get_train_data('data/Motor_tain_tutu/Positive' ,Motor_train_p_ID, 1, 10000, 10000)
X_train_n, y_train_n = get_train_data('data/Motor_tain/Negative' ,Motor_train_n_ID, 0, 10000, 10000)
X_train = np.concatenate((X_train_p ,X_train_n) ,axis=0)
y_train = np.concatenate((y_train_p, y_train_n) ,axis=0)

#生成test数据
test_path = glob.glob('data\\Motor_testP/*.csv')
test_ID = []

for path in test_path:
    test_ID.append(path.split('\\')[-1].split('_')[0])
test_ID = list(set(test_ID))
X_test = get_test_data('data/Motor_testP', test_ID, 10000, 10000)

lgb_parms = {
    "boosting_type": "gbdt",
    "num_leaves": 150,
    "max_depth": 12,
    "learning_rate": 0.01,
    "n_estimators": 1500,
    "max_bin": 425,
    "subsample_for_bin": 2000,
    "objective": 'binary',
    # "metric": 'l1',
    "min_split_gain": 0,
    "min_child_weight": 0.001,
    "min_child_samples": 20,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 3,
    "reg_lambda": 5,
    "seed": 2019,
    "n_jobs": 5,
    "verbose": 1,
    "silent": False,
    #     'scale_pos_weight':20,
}

preds = lgb_model(X_train, y_train, X_test ,lgb_parms ,5)

thred = 0.00965
result = []
count = 0
for pred in preds:
    if pred < thred:
        result.append(0)
    else:
        result.append(1)
        count += 1
df = pd.DataFrame({'idx':test_ID,'result':result})

df.to_csv('data/stack/sub_tutu.csv' ,index=None)


#step3:lidao_model_predict
from method import data_aug,get_feature
fileList = glob.glob("data/Motor_tain/Positive/*")  ## 需要修改路径
data_aug(fileList, flagl=["_2_", "_3_"], rl=[0.01, 0.05])

fileList = glob.glob("data/Motor_tain/*/*.csv")  ## 需要修改路径

if not os.path.exists("data/train/train_feature.csv"):
    fileList = glob.glob("data/Motor_tain/*/*.csv")
    fileList = list(set([f[:-6] for f in fileList]))
    get_feature(fileList, "data/train/train_feature.csv", flag=True)

train_feature = pd.read_csv("data/train/train_feature.csv")
if not os.path.exists("data/test/test_feature.csv"):
    fileList = glob.glob("data/Motor_testP/*.csv")
    fileList = list(set([f[:-6] for f in fileList]))
    get_feature(fileList, "data/test/test_feature.csv", flag=False)

test_feature = pd.read_csv("data/test/test_feature.csv")
fn = train_feature.columns.tolist()
fn.remove("idx")
fn.remove("label")

param = {'num_leaves': 31, 'objective': 'binary',
         'learning_rate': 0.01, "boosting": "gbdt",
         "feature_fraction": 0.85, "bagging_seed": 2019,
         "metric": 'binary_logloss', "lambda_l1": 0.01,
         "verbosity": -1
         }

num_round = 310
trn_data = lgb.Dataset(train_feature[fn], train_feature["label"])
lgb_clf = lgb.train(param, trn_data, num_round, verbose_eval=10)
res_lgb = lgb_clf.predict(test_feature[fn])
test_feature["result"] = res_lgb
res_pd = test_feature[["idx", "result"]].copy()
res_pd = res_pd.sort_values("result").reset_index(drop=True)
res_pd.loc[:4000, "result"] = 0
res_pd.loc[4000:, "result"] = 1
res_pd["result"] = res_pd["result"].astype(int)
res_pd[["idx", "result"]].to_csv("data/stack/sub_li.csv", index=False)


#step4:融合step2、step3结果
sub_tutu=pd.read_csv('data/stack/sub_tutu.csv')
sub_li=pd.read_csv('data/stack/sub_li.csv')
sub_stack= sub_li.merge(sub_tutu,on='idx',how='left')
sub_stack['result']=sub_stack.sum(axis=1)
sub_stack[sub_stack['result']<2]['result']=1
sub_stack[sub_stack['result']!=1]['result']=0
sub_stack=sub_stack[['idx','result']]
sub_stack.to_csv('data/stack/sub_stack.csv',index=None)


#step5：输出最终结果stack_by-yong-feature

train = pd.read_csv('data/train/train_yong.csv')
test = pd.read_csv('data/test/test_yong.csv')

xtrain = train.drop(['ID','label'],axis=1)
ytrain = train['label']
xtest = test.drop(['ID','label'],axis=1)

sub1=pd.read_csv('data/stack/sub_stack.csv')#最高分数816个1
sub1.columns=['ID','label']
#把最高分label结果匹配到test集上
ysub=test[['ID']].merge(sub1,on='ID',how='left')['label']

#合并test-train作为训练集。
train_df=pd.concat([xtrain,xtest])
y_df=pd.concat([ytrain,ysub])

param = {
    'num_leaves': 2,
    'learning_rate': 0.1,
    'feature_fraction': 0.02,
    'max_depth': -1,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    # 'metric': 'Accuracy',
    'num_threads': 4
}

from method import lgb_yong
ypred = lgb_yong(train_df,y_df,xtest,param,5)

sub=pd.DataFrame({'idx':test['ID'],'result':ypred})
sub=sub.sort_values(by='result').reset_index(drop=True)
sub['result'][:4700]=0
sub['result'][4700:]=1
sub['result']=sub['result'].astype('int')
sub.to_csv('data/stack/stack4700.csv',index=None)

#step6：final
sub_stack=pd.read_csv('data/stack/sub_stack.csv')
sub_stack4700=pd.read_csv('data/stack/stack4700.csv')
sub_final= sub_stack.merge(sub_stack4700,on='idx',how='left')
sub_final['result']=sub_final.sum(axis=1)
sub_final[sub_final['result']<2]=1
sub_final[sub_final['result']!=1]=0
sub_final.to_csv('data/sub_final.csv',index=None)