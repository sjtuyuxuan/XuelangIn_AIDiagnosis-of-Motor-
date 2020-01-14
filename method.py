import numpy as np
import pandas as pd
import time
import os
import gc
import glob
from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb
from scipy.fftpack import fft, fftshift, ifft
import tsfresh.feature_extraction.feature_calculators as fc
from sklearn.preprocessing import PolynomialFeatures
from multiprocessing import Pool
from sklearn.preprocessing import PolynomialFeatures
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from sklearn.model_selection import KFold, StratifiedKFold


#yong_method
def get_new_columns(name,aggs):
    return [str(k) + '_' + name for k in aggs]

def get_new_columns2(name,aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

def GetYongData(path,label):
    w = GD(path,label)
    Pcol = list(set([i.split('_')[0] for i in os.listdir(path)]))
    paths = [i for i in Pcol]
    #多进程读取数据
    all_df = []
    pool= Pool()
    all_df = pool.map(w.get_data, paths)
    train_df = pd.concat(all_df)
    pool.close()
    return train_df


class GD:

    def __init__(self, path, label):

        self.path = path
        self.label = label

    def get_data(self, d):
        col = [k for k in os.listdir(self.path) if d in k]
        x = pd.DataFrame()

        for j in col:
            x1 = pd.read_csv(self.path + j)
            x1.columns = [j.split('_')[1][0], j.split('_')[1][0]] + x1.columns
            x = pd.concat([x, x1], axis=1)
        x['ID'] = d
        x['label'] = self.label

        poly = PolynomialFeatures(degree=2)
        X_poly = pd.DataFrame(poly.fit_transform(x[['Bai1', 'Bai2', 'Fai1', 'Fai2']]))
        X_poly.columns = get_new_columns('BF', X_poly.columns)
        train1 = pd.concat([x[['ID', 'label']], X_poly], axis=1)

        col = [i for i in train1.columns if 'BF' in i]
        for i in [50, 100, 200, 300]:
            q = train1[col].rolling(window=i).mean()
            train1[get_new_columns('win_mean%d' % i, q.columns)] = q

            q = train1[col].rolling(window=i).max()
            train1[get_new_columns('win_max%d' % i, q.columns)] = q

            q = train1[col].rolling(window=i).min()
            train1[get_new_columns('win_min%d' % i, q.columns)] = q

            q = train1[col].rolling(window=i).min()
            train1[get_new_columns('win_std%d' % i, q.columns)] = q

            q = train1[col].diff(i)
            train1[get_new_columns('diff%d' % i, q.columns)] = q

        aggs = {}
        col = [i for i in train1.columns if 'win_' not in i and i not in ['ID', 'label']]
        for i in col:
            aggs[i] = ['std', 'max', 'min', 'sum', 'mean']

        col1 = [i for i in train1.columns if 'win_' in i]

        for i in col1:
            aggs[i] = ['std', 'sum', 'mean']

        w = train1.groupby(['ID', 'label']).agg(aggs)
        w.columns = get_new_columns2("", aggs)
        print("Done%s"%d)
        return w.reset_index()

def lgb_yong(xtrain, ytrain, xtest, param, fk=5):
    folds = StratifiedKFold(n_splits=fk, shuffle=True, random_state=4590)
    ypred = np.zeros(len(xtest))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain, ytrain)):
        print("fold {}".format(fold_))

        trn_data = lgb.Dataset(xtrain.iloc[trn_idx], ytrain.iloc[trn_idx])
        val_data = lgb.Dataset(xtrain.iloc[val_idx], ytrain.iloc[val_idx])

        num_round = 30000
        model = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data],
                          verbose_eval=100, early_stopping_rounds=300)
        ypred += model.predict(xtest) / fk


#lidao_method
sampleRate = 51200
def data_aug(fileList,flagl=["_2_","_3_"],rl=[0.01,0.05]):
    # 单个特征增强
    for f in tqdm(fileList):
        for ii in range(len(flagl)):
            flag = flagl[ii]
            r = rl[ii]
            file = pd.read_csv(f)
            file_ai1 = file.copy()
            file_ai1["ai1"] = file_ai1["ai1"].values + file_ai1["ai1"].max() * r * np.random.random(size=(file_ai1["ai1"].shape[0]))
            b1,fa = f[:-5],f[-5:]
            fn = b1+flag+"a1_"+fa
            file_ai1.to_csv(fn,index=False)

            file_ai2 = file.copy()
            file_ai2["ai2"] = file_ai2["ai2"].values + file_ai2["ai2"].max() * r * np.random.random(size=(file_ai2["ai2"].shape[0]))
            b1,fa = f[:-5],f[-5:]
            fn = b1+flag+"a2_"+fa
            file_ai2.to_csv(fn,index=False)

            ai1_ai2 = file.copy()
            ai1_ai2["ai1"] = ai1_ai2["ai1"].values + ai1_ai2["ai1"].max() * r * np.random.random(size=(ai1_ai2["ai1"].shape[0]))
            ai1_ai2["ai2"] = ai1_ai2["ai2"].values + ai1_ai2["ai2"].max() * r * np.random.random(size=(ai1_ai2["ai2"].shape[0]))
            b1,fa = f[:-5],f[-5:]
            fn = b1+flag+"a12_"+fa
            ai1_ai2.to_csv(fn,index=False)
def get_feature(fileList,saveFile,flag=False):
    nuf = 0
    for file in tqdm(fileList):
        fList = []
        vvF = []
        idxF = file.split("\\")[-1]
        fList.append(idxF)
        for fc in ["_B.csv","_F.csv"]:
            vv = []
            extract_file =  pd.read_csv(file+fc)
            for exf in ["ai1","ai2"]:
                v = extract_file[exf].values
                vv.append(v)
                vvF.append(v)
                numWin = int(v.shape[0] / (sampleRate*0.25))
                for win in range(numWin):
                    winV = v[int(win*sampleRate*0.25):int((win+1)*sampleRate*0.25)]
                    stdF = winV.std()
                    meanF = winV.mean()
                    maxF = winV.max()
                    minF = winV.min()
                    fList.extend([stdF,meanF,maxF,minF])
                fList.extend([v.std(),v.mean(),v.max(),v.min()])
                maoci = np.abs(v - v.mean()) / (v.std()*2) *np.abs(v)
                maoci = maoci.sum()
                fList.append(maoci)
            difv = vv[0] - vv[1]
            fList.extend([difv.std(),difv.mean(),difv.max(),difv.min()])
        diffai1 = vvF[0] - vvF[2]
        fList.extend([diffai1.std(),diffai1.mean(),diffai1.max(),diffai1.min()])
        diffai2 = vvF[1] - vvF[3]
        fList.extend([diffai2.std(),diffai2.mean(),diffai2.max(),diffai2.min()])
        diffai1_ai2 = vvF[0] - vvF[3]
        fList.extend([diffai1_ai2.std(),diffai1_ai2.mean(),diffai1_ai2.max(),diffai1_ai2.min()])
        diffai2_ai1 = vvF[1] - vvF[2]
        fList.extend([diffai2_ai1.std(),diffai2_ai1.mean(),diffai2_ai1.max(),diffai2_ai1.min()])
        if flag:
            if "Negative" in file:
                fList.append(0)
            elif "Positive" in file:
                fList.append(1)
        else:
            fList.append("None")
        if nuf == 0:
            col = ["f_"+str(ii) for ii in range(len(fList))]
            col[0] = "idx"
            col[-1] = "label"
            feature = pd.DataFrame(columns=col)
            feature.loc[nuf,:] = np.array(fList)
        else:
            feature.loc[nuf,:] = np.array(fList)
        nuf += 1
    feature.to_csv(saveFile,index=False)
    print("done")


#tutu_method

def data_augmentation(num,ID,Pos_Neg='Positive'):
    for j in tqdm(range(num)):
        i = 0
        for path_ID in ID:
            df = pd.read_csv('data/Motor_tain/'+Pos_Neg+'/'+ path_ID +'_B.csv')
            df2 = df.copy()

            std = np.random.choice([0.0001,0.00012,0.00014,0.00016,0.00018,0.0002,0.00022,0.00024,0.00026,0.00028,0.0003,\
                                    0.00032,0.00034,0.00036,0.00038,0.0004])

            noise1 = np.random.normal(0, std,79999)
            df2['ai1'] = df2['ai1'] + noise1

            noise2 = np.random.normal(0, std,79999)
            df2['ai2'] = df2['ai2'] + noise2

            df2.to_csv('data/Motor_tain_tutu/'+Pos_Neg+'/'+'%s-%s_B.csv'%(i,j),index=None)
            df.to_csv('data/Motor_tain_tutu/'+Pos_Neg+'/'+ path_ID +'_B.csv',index=None)


            df = pd.read_csv('data/Motor_tain/'+Pos_Neg+'/'+ path_ID +'_F.csv')
            df2 = df.copy()

            noise1 = np.random.normal(0, std,79999)
            df2['ai1'] = df2['ai1'] + noise1
            noise2 = np.random.normal(0, std,79999)
            df2['ai2'] = df2['ai2'] + noise2

            df2.to_csv('data/Motor_tain_tutu/'+Pos_Neg+'/'+'%s-%s_F.csv'%(i,j),index=None)
            df.to_csv('data/Motor_tain_tutu/' + Pos_Neg+ '/' + path_ID + '_F.csv', index=None)
            i+=1

def get_unique_ID(path):
    file_paths = glob.glob(path + '/*.csv')
    unique_ID = []
    for file_path in file_paths:
        unique_ID.append(file_path.split('\\')[-1].split('_')[0])
    return list(set(unique_ID))


def get_stastic_features(df_B, df_F, win_size, win_skip):
    # 构建多项式特征
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

    X_B = df_B[['ai1', 'ai2']]
    X_ploly_B = poly.fit_transform(X_B)
    X_ploly_df_B = pd.DataFrame(X_ploly_B, columns=poly.get_feature_names())

    X_F = df_F[['ai1', 'ai2']]
    X_ploly_F = poly.fit_transform(X_F)
    X_ploly_df_F = pd.DataFrame(X_ploly_F, columns=poly.get_feature_names())

    win_size = [10000, 20000, 40000, 80000]
    features = []

    for df in [X_ploly_df_B, X_ploly_df_F]:
        for fe in ['x0', 'x1']:
            for size in win_size:
                for i in range(0, len(df), size):

                    #                     features.append(df[fe][i:i+size].max())
                    features.append(df[fe][i:i + size].max() - df[fe][i:i + size].min())
                    #                     features.append(df[fe][i:i+size].min())
                    features.append(df[fe][i:i + size].std())
                    features.append(df[fe][i:i + size].median())
                    features.append(df[fe][i:i + size].mean())
                    features.append(df[fe][i:i + size].mad())
                    features.append(df[fe][i:i + size].skew())
                    features.append(df[fe][i:i + size].kurtosis())
                    features.append(fc.autocorrelation(df[fe][i:i + size], 100))
                    features.append(fc.binned_entropy(df[fe][i:i + size], 100))
                    features.append(fc.abs_energy(df[fe][i:i + size]))

                    if size == 10000 or size == 20000:

                        fs = fc.fft_coefficient(df[fe][i:i + size],
                                                param=[{"coeff": 10, 'attr': 'real'}, {"coeff": 10, "attr": "imag"},
                                                       {"coeff": 10, "attr": "abs"}, {"coeff": 10, "attr": "angle"}])
                        for item in fs:
                            features.append(item[1])
        for fe in ['x0^2', 'x0 x1', 'x1^2']:
            for size in win_size:
                for i in range(0, len(df), size):
                    #                     features.append(df[fe][i:i+size].max())
                    features.append(df[fe][i:i + size].max() - df[fe][i:i + size].min())
                    #                     features.append(df[fe][i:i+size].min())
                    features.append(df[fe][i:i + size].std())
                    features.append(df[fe][i:i + size].median())
                    features.append(df[fe][i:i + size].mean())
                    features.append(df[fe][i:i + size].mad())
                    features.append(df[fe][i:i + size].skew())
                    features.append(df[fe][i:i + size].kurtosis())
    return features


def get_train_data(path, ID, label, win_size, win_skip):
    df_B = pd.read_csv(path + '/' + ID[0] + '_B.csv')
    df_F = pd.read_csv(path + '/' + ID[0] + '_F.csv')
    features = get_stastic_features(df_B, df_F, win_size, win_skip)

    X_train = np.zeros((len(ID), len(features)))
    y_train = np.zeros((len(ID),))
    i = 0
    for id_ in tqdm(ID):
        df_B = pd.read_csv(path + '/' + id_ + '_B.csv')
        df_F = pd.read_csv(path + '/' + id_ + '_F.csv')
        features = get_stastic_features(df_B, df_F, win_size, win_skip)
        X_train[i, :] = features
        y_train[i] = label
        i += 1
    return X_train, y_train


def lgb_model(train_features, train_labels, test_features, params, nflod):
    kfolder = KFold(n_splits=nflod, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_features, train_labels)
    print('start lightgbm train..')
    preds_list = list()
    for train_index, test_index in kfold:
        k_x_train = train_features[train_index]
        k_y_train = train_labels[train_index]
        k_x_test = train_features[test_index]
        k_y_test = train_labels[test_index]

        gbm = lgb.LGBMRegressor(**params)
        gbm = gbm.fit(k_x_train, k_y_train,
                      eval_metric="logloss",
                      eval_set=[(k_x_train, k_y_train),
                                (k_x_test, k_y_test)],
                      eval_names=["train", "valid"],
                      early_stopping_rounds=200,
                      verbose=True)

        preds = gbm.predict(test_features, num_iteration=gbm.best_iteration_)

        preds_list.append(preds)

    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_list = list(preds_df.mean(axis=1))

    return preds_list

def get_test_data(path, ID, win_size, win_skip):

    df_B = pd.read_csv(path + '/' + ID[0] + '_B.csv')
    df_F = pd.read_csv(path + '/' + ID[0] + '_F.csv')
    features = get_stastic_features(df_B, df_F, win_size, win_skip)

    X_test = np.zeros((len(ID), len(features)))
    i = 0
    for id_ in tqdm(ID):
        df_B = pd.read_csv(path + '/' + id_ + '_B.csv')
        df_F = pd.read_csv(path + '/' + id_ + '_F.csv')
        features = get_stastic_features(df_B, df_F, win_size, win_skip)
        X_test[i, :] = features

        i += 1
    return X_test


