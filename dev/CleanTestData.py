import pickle 
import pandas as pd 
import numpy as np 
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
#%pylab inline

SHARED_FOLDER = '/Users/liqian/git/Kaggle_HousePrice/data'

#with open ('{}/train.csv'.format(SHARED_FOLDER), 'rb') as f:
    #train = pd.read_csv(f)

with open ('{}/test.csv'.format(SHARED_FOLDER), 'rb') as f:
    train = pd.read_csv(f)

train["MSZoning"]=train['MSZoning'].fillna('NoMSZ')

train["Utilities"]=train['Utilities'].fillna('NoUti')

train["Exterior1st"]=train['Exterior1st'].fillna('Other')

train["Exterior2nd"]=train['Exterior2nd'].fillna('Other')

train["BsmtFinSF1"]=train['BsmtFinSF1'].fillna(0.0)

train["BsmtFinSF2"]=train['BsmtFinSF2'].fillna(0.0)

train["BsmtUnfSF"]=train['BsmtUnfSF'].fillna(0.0)

train["TotalBsmtSF"]=train['TotalBsmtSF'].fillna(0.0)

train["BsmtFullBath"]=train['BsmtFullBath'].fillna(0)

train["BsmtHalfBath"]=train['BsmtHalfBath'].fillna(0)

train["KitchenQual"]=train['KitchenQual'].fillna('NoQual')

train["Functional"]=train['Functional'].fillna('NoType')

#garagetype = train[train['GarageCars'].isnull]['Garagetype'].iloc[0]
train["GarageCars"]=train['GarageCars'].fillna(train[train['GarageType'] == 'Detchd']['GarageCars'].mean())

train["GarageArea"]=train['GarageArea'].fillna(train[train['GarageType'] == 'Detchd']['GarageArea'].mean())

train["SaleType"]=train['SaleType'].fillna('Oth')

Alley2=train['Alley'].fillna('NoAlley')
train["Alley"]=Alley2

MiscFeature2=train['MiscFeature'].fillna('NoMiscFeature')
train["MiscFeature"]=MiscFeature2


Fence2=train['Fence'].fillna('NoFence')
train["Fence"]=Fence2


PoolQC2=train['PoolQC'].fillna('NoPool')
train["PoolQC"]=PoolQC2

GarageCond2=train['GarageCond'].fillna('NoGarage')
train["GarageCond"]=GarageCond2

GarageQual2=train['GarageQual'].fillna('NoGarage')
train["GarageQual"]=GarageCond2

GarageFinish2=train['GarageFinish'].fillna('NoGarage')
train["GarageFinish"]=GarageFinish2

GarageType2=train['GarageType'].fillna('NoGarage')
train["GarageType"]=GarageType2

FireplaceQu2=train['FireplaceQu'].fillna('NoFireplace')
train["FireplaceQu"]=FireplaceQu2

BsmtFinType22=train['BsmtFinType2'].fillna('NoBasement')
train["BsmtFinType2"]=BsmtFinType22

BsmtFinType12=train['BsmtFinType1'].fillna('NoBasement')
train["BsmtFinType1"]=BsmtFinType12

BsmtCond2=train['BsmtCond'].fillna('NoBasement')
train["BsmtCond"]=BsmtCond2
  
BsmtQual2=train['BsmtQual'].fillna('NoBasement')
train["BsmtQual"]=BsmtQual2

LFm=train["LotFrontage"].mean()
LotFrontage2=train['LotFrontage'].fillna(LFm)
train["LotFrontage"]=LotFrontage2

GarageYrBlt2=train['GarageYrBlt'].fillna(0)
train["GarageYrBlt2"]=GarageYrBlt2

GarageYrBlt2=train['GarageYrBlt'].fillna(train.YearBuilt)
train["GarageYrBlt"]=GarageYrBlt2

MasVnrType2=train['MasVnrType'].fillna("none")
train["MasVnrType"]=MasVnrType2

MasVnrArea2=train['MasVnrArea'].fillna(0.0)
train["MasVnrArea"]=MasVnrArea2

train["BsmtExposure"]=train['BsmtExposure'].fillna("NoBasement")

#train=train.drop(train.index[948])

#del train['Unnamed: 0']
del train['GarageYrBlt2']

train.to_csv("/Users/liqian/git//Kaggle_HousePrice/datatest_clean_v2.csv")
