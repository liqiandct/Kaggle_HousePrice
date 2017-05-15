import pickle 
import pandas as pd 
import numpy as np 
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

import seaborn as sns

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


SHARED_FOLDER = '/Users/liqian/git/Kaggle_HousePrice/data'

with open ('{}/train.csv'.format(SHARED_FOLDER), 'rb') as f:
    train1 = pd.read_csv(f)

with open ('{}/test.csv'.format(SHARED_FOLDER), 'rb') as f:
    test1 = pd.read_csv(f)

with open ('{}/train_cleaned_testv3.csv'.format(SHARED_FOLDER), 'rb') as f:
    train = pd.read_csv(f)

with open ('{}/test_cleaned_testv3.csv'.format(SHARED_FOLDER), 'rb') as f:
    test = pd.read_csv(f)

with open ('{}/train_saleprice.csv'.format(SHARED_FOLDER), 'rb') as f:
    price = pd.read_csv(f)


del train['Unnamed: 0'] 
#del train['Fireplaces'] 
del test['Unnamed: 0'] 
#del test['Fireplaces'] 

pricetest = pd.DataFrame(index = train.index, columns=["SalePrice"])
pricetest["SalePrice"] = price

all_data = pd.concat((train.loc[:,'LotFrontage':'NeighborhoodBin'], test.loc[:,'LotFrontage':'NeighborhoodBin']))
y = pricetest.SalePrice
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y)

#####Ridge regression####################
model_ridge = Ridge(alpha = alphas).fit(X_train, y)
rmse_cv(model_ridge).mean()

alphas = [0.05, 0.1, 0.5, 1, 5, 10, 15, 30, 50, 70]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.min() ###0.140714 #15
cv_ridge.plot(title = "RMSE for Ridge Regression")
plt.xlabel("alpha")
plt.ylabel("rmse")

#plt.draw()
#plt.show()
#plt.close()

#####LASSO regression#######
model_lasso = LassoCV(alphas=[0.1, 0.05, 0.01, 0.001, 0.0005, 0.0001]).fit(X_train1, y_train1)
cv_lasso = rmse_cv(model_lasso).mean() #0.114526(after delete 3 variables)

######graphe de mse path pour choisir meuilleur alphas######
m_log_alphas = -np.log10(model_lasso.alphas_)

plt.figure()
ymin, ymax = 2300, 3800
plt.plot(m_log_alphas, model_lasso.mse_path_, ':')
plt.plot(m_log_alphas, model_lasso.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model_lasso.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)
##get the coeff for each column
coef = pd.Series(model_lasso.coef_, index = X_train1.columns).sort_values(ascending=False) 

plt.figure(figsize=(10, 5))
coef.head(20).plot(kind='bar')
plt.title('Feature Importance in Lasso Model')
plt.tight_layout()

imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
plt.figure(figsize=(8, 10))
imp_coef.plot(kind = "barh")
plt.title("Coefficients in Lasso Model")
plt.tight_layout()

preds_lasso = pd.DataFrame({"preds":model_lasso.predict(X_test1), "true":y_test1})
preds_lasso["residuals"] = preds_lasso["true"] - preds_lasso["preds"]
preds_lasso["residuals"].abs().mean() #0.0807606
preds_lasso.plot(x = "preds", y = "residuals",kind = "scatter")

plt.figure(figsize=(10, 5))
plt.scatter(y_test1, preds_lasso["preds"], s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)])
plt.tight_layout()

#####xgboost model##########
import xgboost as xgb

dtrain = xgb.DMatrix(X_train1, label = y_train1)
dtest = xgb.DMatrix(X_test1, label = y_test1)
###for rmse 
param = {'bst:max_depth':4, 'bst:eta':0.1}
cv_xgb = xgb.cv(param, dtrain,  num_boost_round=500, early_stopping_rounds=100)
cv_xgb[["test-rmse-mean", "train-rmse-mean"]].plot()
###scikit-learn API#####
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, max_depth=4, min_child_weight=1.5, learning_rate=0.01, reg_alpha=0.9, 
								reg_lambda=0.6, subsample=0.2, seed=42, silent=1, n_estimators = 7200)

#model_xgb = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators = 400)
model_xgb.fit(X_train1, y_train1)

preds_xgb = pd.DataFrame({"preds":model_xgb.predict(X_test1), "true":y_test1})
preds_xgb["residuals"] = preds_xgb["true"] - preds_xgb["preds"] 
##preds_xgb["residuals"].abs().mean() #0.07744729
preds_xgb.plot(x = "preds", y = "residuals",kind = "scatter")

plt.figure(figsize=(10, 5))
plt.scatter(y_test1, preds_xgb["preds"], s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)])
plt.tight_layout()

############Plot features importance of XGB#######################
plt.figure(figsize=(10, 5))
xgb.plot_importance(model_xgb,max_num_features=20)
plt.tight_layout()

plt.figure(figsize=(10, 5))
xgb.plot_tree(model_xgb, num_trees=2)
plt.title('Trees of XGB Model')
plt.tight_layout()


xgb.to_graphviz(model_xgb, num_trees=2)
plt.title('Graphviz XGB Model')
plt.tight_layout()
#########Lasso + XGB with train data###########
predictions = pd.DataFrame({"predsx": preds_lasso["preds"], "true":y_test1})
predictions["preds"] = 0.4*predictions["predsx"] + 0.6*preds_xgb["preds"]
predictions["residuals"] = predictions["true"] - predictions["preds"]
predictions["residuals"].abs().mean() 
#0.0758648(0.5) #0.0755988802(0.4)

predictions.plot(x = "preds", y = "residuals",kind = "scatter")

plt.figure(figsize=(10, 5))
plt.scatter(y_test1, predictions["preds"], s=20)
plt.title('Predicted vs. Actual')3
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)])
plt.tight_layout()

##########submission############################
predst_xgb = model_xgb.predict(X_test)
predst_lasso = model_lasso.predict(X_test)
predst = 0.4*predst_lasso + 0.6*predst_xgb
result=np.expm1(predst)
solution = pd.DataFrame({"id":test1.Id, "SalePrice":result})
solution.to_csv("/Users/liqian/git/Kaggle_HousePrice/data／prediction_0.4lasso_cdv3_v2.csv", index = False)