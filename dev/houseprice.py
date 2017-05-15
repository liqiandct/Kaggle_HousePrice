import pickle 
import pandas as pd 
import numpy as np 
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
%pylab inline

SHARED_FOLDER = '/Users/liqian/git/Kaggle_HousePrice/data'

with open ('{}/train.csv'.format(SHARED_FOLDER), 'rb') as f:
    train_df = pd.read_csv(f)

with open ('{}/test.csv'.format(SHARED_FOLDER), 'rb') as f:
    test_df = pd.read_csv(f)

def change_value(target_df,column, value_old, value_new):
    target_df.loc[target_df[column] == value_old,column] = value_new

def remplir_null(target_df, column, value_new):
	target_df.loc[target_df[column].isnull(), column] = value_new


##############MasVnrType, MasVnrArea##################################
sns.pairplot(train_df[['MasVnrType','MasVnrArea']].dropna())

MasVnr_NoValue_df = train_df[train_df['MasVnrType'].isnull() == True][['Id','MasVnrType','MasVnrArea','SalePrice']]
MasVnr_WithValue_df = train_df[train_df['MasVnrType'].isnull() == False][['Id','MasVnrType','MasVnrArea','SalePrice']]
MasVnr_WithValue_df['MasVnrType'].corr(MasVnr_WithValue_df['SalePrice'], method='kendall') #spearman

MasVnr_test_df = MasVnr_WithValue_df
MasVnr_test_df.loc[MasVnr_test_df['MasVnrType'] == 'None', 'MasVnrType'] = 0
MasVnr_test_df.loc[MasVnr_test_df['MasVnrType'] == 'BrkCmn', 'MasVnrType'] = 1
MasVnr_test_df.loc[MasVnr_test_df['MasVnrType'] == 'BrkFace', 'MasVnrType'] = 2
MasVnr_test_df.loc[MasVnr_test_df['MasVnrType'] == 'CBlock', 'MasVnrType'] = 3
MasVnr_test_df.loc[MasVnr_test_df['MasVnrType'] == 'Stone', 'MasVnrType'] = 4
#MasVnr_test_df['MasVnrType'] = MasVnr_test_df['MasVnrType'].astype(int)
MasVnr_test_df['MasVnrType'].corr(MasVnr_test_df['SalePrice'], method='pearson') #0.43002815239002551
MasVnr_test_df['MasVnrType'].corr(MasVnr_test_df['SalePrice'], method='spearman') #0.42527411818259886

MasVnr_Alltest_df = train_df[['Id','MasVnrType','MasVnrArea','SalePrice']]
MasVnr_Alltest_df.loc[MasVnr_Alltest_df['MasVnrType'] == 'None', 'MasVnrType'] = 0
MasVnr_Alltest_df.loc[MasVnr_Alltest_df['MasVnrType'] == 'BrkCmn', 'MasVnrType'] = 1
MasVnr_Alltest_df.loc[MasVnr_Alltest_df['MasVnrType'] == 'BrkFace', 'MasVnrType'] = 2
MasVnr_Alltest_df.loc[MasVnr_Alltest_df['MasVnrType'] == 'CBlock', 'MasVnrType'] = 3
MasVnr_Alltest_df.loc[MasVnr_Alltest_df['MasVnrType'] == 'Stone', 'MasVnrType'] = 4
MasVnr_Alltest_df.loc[MasVnr_Alltest_df['MasVnrType'].isnull(), 'MasVnrType'] = 9
MasVnr_Alltest_df['MasVnrType'].corr(MasVnr_Alltest_df['SalePrice'], method='spearman') #0.4289946550895265

remplir_null(train_df, 'MasVnrType', 'None')
remplir_null(train_df, 'MasVnrArea', 0.0)


