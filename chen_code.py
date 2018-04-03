# GetYourGuide data science take home task
#
# Author: Kai Chen
# Date: Mar, 2018
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
#%matplotlib inline

from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error



TRAIN_FILE = 'ds_dp_assessment/train.csv'
TEST_FILE = 'ds_dp_assessment/prediction.csv'


#--------------------
# 1. Data preparation
#
train_df = pd.read_csv(TRAIN_FILE)
train_df['RPC'] = train_df['Revenue']/train_df['Clicks']
test_df = pd.read_csv(TEST_FILE)

#print(train_df.head())
#print(train_df.describe())

print('----------------')
print('train data')
print(train_df.describe())
print('test data')
print(test_df.describe())
print('----------------')

feature_column_names = ['Keyword_ID', 'Ad_group_ID', 'Campaign_ID', 'Account_ID', 'Device_ID', 'Match_type_ID']
X_train = train_df[feature_column_names]
Y_train = train_df['RPC']
X_test = test_df[feature_column_names]

print('----------------')
print('x train shape')
print(X_train.shape)
print('y train shape')
print(Y_train.shape)
print('x test shape')
print(X_test.shape)
print('----------------')

#--------------------
# 2. Feature engineering
#
# hasher = FeatureHasher(n_features=5,
#             non_negative=True,
#             input_type='string')
# ## Keyword_ID
# tmp_features = hasher.transform(X_train['Keyword_ID'].astype(str).tolist()).toarray()
# for i in range(5):
#     X_train['keyword_'+str(i)] = tmp_features[:, i]
# X_train.drop(['Keyword_ID'], axis=1, inplace=True)
# print(X_train.describe())


#--------------------
# 3. Model training
#

# examples of using xgboost for regression
#xgb_clf = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
#                           colsample_bytree=1, max_depth=7)
#xgb_clf.fit(X_train_sub, Y_train_sub)
#predictions = xgb_clf.predict(X_val)
#print(predictions)
#print(explained_variance_score(predictions,Y_val))

# xgb_model = xgb.train(
#     xgb_params,
#     dtrain_mat,
#     num_boost_round=100,
#     #evals=[(dval_mat, "Test")],
#     #early_stopping_rounds=10
# )

# cv_results = xgb.cv(
#     xgb_params,
#     dtrain_mat,
#     num_boost_round=10,
#     seed=42,
#     nfold=2,
#     metrics={'rmse'},
#     early_stopping_rounds=10
# )
# print(cv_results)


# ----------------------
# prepare datasets for xgb
#
X_train_sub, X_val, Y_train_sub, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

dtrain_mat = xgb.DMatrix(X_train, Y_train)
dtrain_sub_mat = xgb.DMatrix(X_train_sub, Y_train_sub)
dval_mat = xgb.DMatrix(X_val, Y_val)
dtest_mat = xgb.DMatrix(X_test)

# details of xgboost parameters
# http://xgboost.readthedocs.io/en/latest/parameter.html
xgb_params = {'eta':0.1,
              'seed':42,
              'subsample':0.8,
              'colsample_bytree':0.8,
              'objective':'reg:linear',
              #'objective':'binary:logistic',
              'max_depth':6,
              'min_child_weight':1,
              #'metrics':['auc'],
              #'metrics':['mae'],
              'metrics':['rmse'],
              'eval_metric':['rmse'],
              # 'nthread':1
              }

# -------------------------------
# hyperparameter tuning with CV
# take the idea from: https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html
n_fold = 2
num_boost_round = 200
early_stopping_rounds = 10

# tune 'max_depth' and 'min_child_weight'
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(4, 5)
    for min_child_weight in range(4, 5)
]

min_rmse = float("Inf")
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))

    xgb_params['max_depth'] = max_depth
    xgb_params['min_child_weight'] = min_child_weight

    # TODO: change metrics
    cv_results = xgb.cv(
        xgb_params,
        dtrain_sub_mat,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=n_fold,
        # metrics={'mae'},
        metrics={'rmse'},
        early_stopping_rounds=early_stopping_rounds
    )

    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth, min_child_weight)

print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))

xgb_params['max_depth'] = best_params[0]
xgb_params['min_child_weight'] = best_params[1]


# tune 'subsample' and 'colsample'
# subsample corresponds the fraction of observations (the rows) to subsample at each step.
# By default it is set to 1 meaning that we use all rows.
# colsample_bytree corresponds to the fraction of features (the columns) to use.
# By default it is set to 1 meaning that we will use all features.

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(10, 11)]
    for colsample in [i/10. for i in range(10, 11)]
]

min_rmse = float("Inf")
best_params = None

for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))

    xgb_params['subsample'] = subsample
    xgb_params['colsample_bytree'] = colsample

    cv_results = xgb.cv(
        xgb_params,
        dtrain_sub_mat,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=n_fold,
        # metrics={'mae'},
        metrics={'rmse'},
        early_stopping_rounds=early_stopping_rounds
    )

    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (subsample, colsample)

print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], mean_rmse))

xgb_params['max_depth'] = best_params[0]
xgb_params['min_child_weight'] = best_params[1]

# ----------------------
# tune learning rate
#
min_rmse = float("Inf")
best_learning_rate = None
# learning_rate_range = [0.1, 0.05, 0.01, 0.005]
learning_rate_range = [0.1]

for eta in learning_rate_range:
    print("CV with eta={}".format(eta))

    xgb_params['eta'] = eta

    cv_results = xgb.cv(
            xgb_params,
            dtrain_sub_mat,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=n_fold,
            # metrics=['mae'],
            metrics={'rmse'},
            early_stopping_rounds=early_stopping_rounds
          )

    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_learning_rate = eta

print("Best params: {}, RMSE: {}".format(best_learning_rate, mean_rmse))

xgb_params['eta'] = best_learning_rate

# show parameters
print('XBG parameters')
print(xgb_params)

# ---------------
# find the best number of boost round on the validation set
#
xgb_model = xgb.train(
    xgb_params,
    dtrain_sub_mat,
    num_boost_round=10,
    evals=[(dval_mat, "val")],
    early_stopping_rounds=early_stopping_rounds
)

# ---------
# train the final model with the tuned parameters on all the training set
#
num_boost_round = xgb_model.best_iteration + 1

best_model = xgb.train(
    xgb_params,
    dtrain_mat,
    num_boost_round=num_boost_round,
)

# ---------
# save the model
xgb_model_path = "xgb-{}-{}-{}.model".format(n_fold, num_boost_round, early_stopping_rounds)
best_model.save_model(xgb_model_path)
print('save xgb model to {}'.format(xgb_model_path))


# -------------
# prediction
# loaded_model = xgb.Booster()
# loaded_model.load_model("my_model.model")
#
# loaded_model.predict(dtest)
y_prediction = best_model.predict(dtest_mat)

print('y prediction')
print(y_prediction)

# -------------
# evaluation
# mean_squared_error(y_prediction, y_test)



# cv_results = xgb.cv(
#     xgb_params,
#     dtrain_mat,
#     num_boost_round=100,
#     seed=42,
#     nfold=5,
#     #metrics={'mae'},
#     metrics={'mse'},
#     early_stopping_rounds=10
# )
# print(cvresult)
#
# y_pred_val = xgb_clf.predict(xgb_val_mat)
# print(y_pred_val)
#
# print("Mean squared error: %.2f" % mean_squared_error(Y_val, y_pred_val))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(Y_val, y_pred_val))


