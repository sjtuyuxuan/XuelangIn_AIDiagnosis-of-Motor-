import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb
# from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.fftpack import fft, fftshift, ifft
import tsfresh.feature_extraction.feature_calculators as fc
from sklearn.preprocessing import PolynomialFeatures

def data_augmentation(num,ID,Pos_Neg='Positive'):
	for j in tqdm(range(num)):
		i = 0
		for path_ID in ID:
			df = pd.read_csv('Motor_tain/'+Pos_Neg+'/'+ path_ID +'_B.csv')
			df2 = df.copy()

			std = np.random.choice([0.0001,0.00012,0.00014,0.00016,0.00018,0.0002,0.00022,0.00024,0.00026,0.00028,0.0003,\
									0.00032,0.00034,0.00036,0.00038,0.0004])

			noise1 = np.random.normal(0, std,79999)
			df2['ai1'] = df2['ai1'] + noise1

			noise2 = np.random.normal(0, std,79999)
			df2['ai2'] = df2['ai2'] + noise2

			df2.to_csv('Motor_tain/'+Pos_Neg+'/'+'%s-%s_B.csv'%(i,j),index=None)

			df = pd.read_csv('Motor_tain/'+Pos_Neg+'/'+ path_ID +'_F.csv')
			df2 = df.copy()

			noise1 = np.random.normal(0, std,79999)
			df2['ai1'] = df2['ai1'] + noise1
			noise2 = np.random.normal(0, std,79999)
			df2['ai2'] = df2['ai2'] + noise2

			df2.to_csv('Motor_tain/'+Pos_Neg+'/'+'%s-%s_F.csv'%(i,j),index=None)

			i+=1

def get_unique_ID(path):
    file_paths = glob.glob(path+'/*.csv')
    unique_ID = []
    for file_path in file_paths:
        unique_ID.append(file_path.split('\\')[-1].split('_')[0])
    return list(set(unique_ID))

def get_stastic_features(df_B, df_F, win_size, win_skip):

    #构建多项式特征
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    
    X_B = df_B[['ai1','ai2']]
    X_ploly_B = poly.fit_transform(X_B)
    X_ploly_df_B = pd.DataFrame(X_ploly_B, columns=poly.get_feature_names())
    
    X_F = df_F[['ai1','ai2']]
    X_ploly_F = poly.fit_transform(X_F)
    X_ploly_df_F = pd.DataFrame(X_ploly_F, columns=poly.get_feature_names())
    
    win_size = [10000,20000,40000,80000]
    features = []

    for df in [X_ploly_df_B, X_ploly_df_F]:
        for fe in ['x0','x1']:
            for size in win_size:
                for i in range(0,len(df),size):

#                     features.append(df[fe][i:i+size].max())
                    features.append(df[fe][i:i+size].max()-df[fe][i:i+size].min())
#                     features.append(df[fe][i:i+size].min())
                    features.append(df[fe][i:i+size].std())
                    features.append(df[fe][i:i+size].median())
                    features.append(df[fe][i:i+size].mean())
                    features.append(df[fe][i:i+size].mad())
                    features.append(df[fe][i:i+size].skew())
                    features.append(df[fe][i:i+size].kurtosis())
                    features.append(fc.autocorrelation(df[fe][i:i+size], 100))
                    features.append(fc.binned_entropy(df[fe][i:i+size], 100))
                    features.append(fc.abs_energy(df[fe][i:i+size]))

                    if size == 10000 or size == 20000:
                        
                        fs = fc.fft_coefficient(df[fe][i:i+size], param=[{"coeff":10, 'attr':'real'}, {"coeff":10, "attr":"imag"}, {"coeff":10, "attr":"abs"}, {"coeff":10, "attr":"angle"}])
                        for item in fs:
                            features.append(item[1])
        for fe in ['x0^2','x0 x1','x1^2']:
            for size in win_size:
                for i in range(0,len(df),size):

#                     features.append(df[fe][i:i+size].max())
                    features.append(df[fe][i:i+size].max()-df[fe][i:i+size].min())
#                     features.append(df[fe][i:i+size].min())
                    features.append(df[fe][i:i+size].std())
                    features.append(df[fe][i:i+size].median())
                    features.append(df[fe][i:i+size].mean())
                    features.append(df[fe][i:i+size].mad())
                    features.append(df[fe][i:i+size].skew())
                    features.append(df[fe][i:i+size].kurtosis())
    return features

def get_train_data(path, ID, label, win_size, win_skip):
    
    df_B = pd.read_csv(path +'/'+ ID[0] +'_B.csv')
    df_F = pd.read_csv(path +'/'+ ID[0] +'_F.csv')
    features = get_stastic_features(df_B, df_F,win_size,win_skip)
    
    X_train = np.zeros((len(ID),len(features)))
    y_train = np.zeros((len(ID),))
    i = 0
    for id_ in tqdm(ID):
        df_B = pd.read_csv(path +'/'+ id_ +'_B.csv')
        df_F = pd.read_csv(path +'/'+ id_ +'_F.csv')
        features = get_stastic_features(df_B, df_F,win_size,win_skip)
        X_train[i,:] = features
        y_train[i] = label
        i += 1
    return X_train, y_train

def lgb_model(train_features,train_labels, test_features, params, nflod):

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
    
    df_B = pd.read_csv(path +'/'+ ID[0] +'_B.csv')
    df_F = pd.read_csv(path +'/'+ ID[0] +'_F.csv')
    features = get_stastic_features(df_B, df_F,win_size, win_skip)
    
    X_test = np.zeros((len(ID),len(features)))
    i = 0
    for id_ in tqdm(ID):
        df_B = pd.read_csv(path +'/'+ id_ +'_B.csv')
        df_F = pd.read_csv(path +'/'+ id_ +'_F.csv')
        features = get_stastic_features(df_B, df_F,win_size, win_skip)
        X_test[i,:] = features

        i += 1
    return X_test

if __name__ == '__main__':

	Motor_train_p_ID = get_unique_ID('Motor_tain/Positive')
	Motor_train_n_ID = get_unique_ID('Motor_tain/Negative')

	data_augmentation(15, Motor_train_p_ID, Pos_Neg='Positive')

	X_train_p, y_train_p = get_train_data('Motor_tain/Positive',Motor_train_p_ID, 1, 10000, 10000)
	X_train_n, y_train_n = get_train_data('Motor_tain/Negative',Motor_train_n_ID, 0, 10000, 10000)
	X_train = np.concatenate((X_train_p,X_train_n),axis=0)
	y_train = np.concatenate((y_train_p, y_train_n),axis=0)

	test_path = glob.glob('Motor_testP/*.csv')
	test_ID = []
	for path in test_path:
		test_ID.append(path.split('\\')[-1].split('_')[0])
	test_ID = list(set(test_ID))
	X_test = get_test_data('Motor_testP', test_ID, 10000, 10000)

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

	preds = lgb_model(X_train, y_train, X_test,lgb_parms,5)

	thred = 0.00965
	result = []
	count = 0
	for pred in preds:
		if pred < thred:
			result.append(0)
		else:
			result.append(1)
			count += 1
	df = pd.read_csv('Submission.csv')
	df['idx'] = test_ID
	df['result'] = result
	df.to_csv('submit_666.csv',index=None)