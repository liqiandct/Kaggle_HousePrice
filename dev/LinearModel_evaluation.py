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


with open ('{}/test.csv'.format(SHARED_FOLDER), 'rb') as f:
    test1 = pd.read_csv(f)

with open ('{}/train_cleaned_testv3.csv'.format(SHARED_FOLDER), 'rb') as f:
    train = pd.read_csv(f)

with open ('{}/test_cleaned_testv3.csv'.format(SHARED_FOLDER), 'rb') as f:
    test = pd.read_csv(f)

with open ('{}/train_saleprice.csv'.format(SHARED_FOLDER), 'rb') as f:
    price = pd.read_csv(f)

del train['Unnamed: 0']
del test['Unnamed: 0'] 
del price['Unnamed: 0']

pricetest = pd.DataFrame(index = train.index, columns=["SalePrice"])
pricetest["SalePrice"] = price

all_data = pd.concat((train.loc[:,'LotFrontage':'NeighborhoodBin'], test.loc[:,'LotFrontage':'NeighborhoodBin']))
y = pricetest.SalePrice
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y)


#####LASSO regression#######
model_lasso = LassoCV(alphas=[0.1, 0.05, 0.01, 0.001, 0.0005, 0.0001]).fit(X_train1, y_train1)
#model_lasso = Lasso(alpha=0.00099, max_iter=50000).fit(X_train1, y_train1)
cv_lasso = rmse_cv(model_lasso).mean() ###0.12323089650508814  # the value will change with the changing of "alphas"

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
preds_lasso["residuals"].abs().mean()
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
###for rmse , pour avoir the best num_boost_round(where it stops)
param = {'bst:max_depth':4, 'bst:eta':0.1}
cv_xgb = xgb.cv(param, dtrain,  num_boost_round=8000, early_stopping_rounds=1000)
cv_xgb[["test-rmse-mean", "train-rmse-mean"]].plot()

from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBRegressor

param_test1 = {
 'max_depth':[3,4,5,6,7,8,9],
 'min_child_weight':[1,3,5,6]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=54, max_depth=4, min_child_weight=1, 
	gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
###scikit-learn API#####
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, max_depth=4, min_child_weight=1.5, learning_rate=0.01, reg_alpha=0.9, 
								reg_lambda=0.6, subsample=0.2, seed=42, silent=1, n_estimators = 7200)

model_xgb = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators = 400)
model_xgb.fit(X_train1, y_train1)

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

#####train residual###########
preds_xgb = pd.DataFrame({"preds":model_xgb.predict(X_test1), "true":y_test1})
preds_xgb["residuals"] = preds_xgb["true"] - preds_xgb["preds"] ##preds_xgb["residuals"].abs().mean() = 0.0797241
preds_xgb.plot(x = "preds", y = "residuals",kind = "scatter")

plt.figure(figsize=(10, 5))
plt.scatter(y_test1, preds_xgb["preds"], s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)])
plt.tight_layout()

#########Lasso + XGB with train data###########
predictions = pd.DataFrame({"predsx": preds_lasso["preds"], "true":y_test1})
predictions["preds"] = 0.3*predictions["predsx"] + 0.7*preds_xgb["preds"]
predictions["residuals"] = predictions["true"] - predictions["preds"]
predictions["residuals"].abs().mean()

predictions.plot(x = "preds lasso+xgb", y = "residuals",kind = "scatter")

plt.figure(figsize=(10, 5))
plt.scatter(y_test1, predictions["preds"], s=20)
plt.title('Predicted Lasso+XGB vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)])
plt.tight_layout()

predst_xgb = model_xgb.predict(X_test)
predst_lasso = model_lasso.predict(X_test)
predst = 0.5*predst_lasso + 0.5*predst_xgb
result=np.expm1(predst)
solution = pd.DataFrame({"id":test1.Id, "SalePrice":result})
solution.to_csv("prediction_0.5lasso_cdv3_v1.csv", index = False) ### dernier submission

##################Random Forest(Just for looking perforanc)##################################
from sklearn.cross_validation import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y)
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)
clf.fit(X_train1, y_train1)
y_pred = clf.predict(X_test1)

preds_rf = pd.DataFrame({"preds": y_pred, "true":y_test1})
preds_rf["residuals"] = preds_rf["true"] - preds_rf["preds"] ##preds_rf["residuals"].abs().mean() = 0.0901161
preds_rf.plot(x = "preds", y = "residuals",kind = "scatter")

plt.figure(figsize=(10, 5))
plt.scatter(y_test1, y_pred, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')

plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)])
plt.tight_layout()

predictions = pd.DataFrame({"predsx": preds_lasso["preds"], "true":y_test1})
predictions["preds"] = 0.8*predictions["predsx"] + 0.2*preds_rf["preds"]
predictions["residuals"] = predictions["true"] - predictions["preds"]
predictions["residuals"].abs().mean()

predst_rf = model_clf.predict(X_test)
result=np.expm1(predst_rf)
solution = pd.DataFrame({"id":test.Id, "SalePrice":result})
solution.to_csv("prediction_rf.csv", index = False)
