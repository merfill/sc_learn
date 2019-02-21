
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Gradient Boosting
#import lightgbm as lgb
import xgboost as xgb

# Scikit-learn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


data = pd.read_csv('data/train_normalized.csv', sep=',', encoding='utf8')
print(data.shape)

train = data.drop(['ID_code', 'target'], axis=1)
y = data['target']

# Transforming the problem into a classification (unbalanced)
y_bin = (y > np.percentile(y, 90)).astype(int)
print(y_bin.shape)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)

MAX_ROUNDS = 5000
sklearn_gbm_iter1 = []
xgb_gbm_iter1 = []

sklearn_gbm_ap1 = []
xgb_gbm_ap1 = []

# Set up the classifier with standard configuration
# Later we will more performing parameters with Bayesian Optimization
params = {
    'learning_rate':  0.06, 
    'max_depth': 6, 
    #'lambda_l1': 16.7,
    'min_data_in_leaf':5,
    'boosting': 'gbdt', 
    'objective': 'binary', 
    'metric': 'auc',
    'feature_fraction': .9,
    'is_training_metric': False, 
    'seed': 1
}

for i, (train_index, test_index) in enumerate(skf.split(train, y)):
    # Create data for this fold
    y_train, y_valid = y[train_index], y[test_index]
    X_train, X_valid = train.iloc[train_index,:], train.iloc[test_index,:]

    print '\nFold ', i

    # Running models for this fold

    # ->Scikit-learn GBM
    #sklearn_gbm = GradientBoostingClassifier(n_estimators=MAX_ROUNDS,
    #                                learning_rate = 0.06,
    #                                max_features=2, 
    #                                max_depth = 6, 
    #                                n_iter_no_change=50, 
    #                                tol=0.01,
    #                                random_state = 0)

    #sklearn_gbm.fit(X_train, y_train)
    #print( " Best iteration sklearn_gbm = ", sklearn_gbm.n_estimators_)

    xgb_gbm = xgb.XGBClassifier(max_depth=6, 
                                n_estimators=MAX_ROUNDS,
                                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                                learning_rate=0.06,
                                early_stopping_rounds=50)

    xgb_gbm.fit(X_train, y_train)
    print 'Best iteration xgboost_gbm = ', xgb_gbm.get_booster().best_iteration

    # Storing and reporting results of the fold
    #sklearn_gbm_iter1 = np.append(sklearn_gbm_iter1, sklearn_gbm.n_estimators_)
    xgb_gbm_iter1 = np.append(xgb_gbm_iter1, xgb_gbm.get_booster().best_iteration)

    #pred = sklearn_gbm.predict(X_valid)
    #ap = average_precision_score(y_valid, pred, average='macro', pos_label=1, sample_weight=None)
    #print('sklearn_gbn ', ap)
    #sklearn_gbm_ap1 = np.append(sklearn_gbm_ap1, ap)

    pred  = xgb_gbm.predict(X_valid)
    ap = average_precision_score(y_valid, pred, average='macro', pos_label=1, sample_weight=None)
    print 'xgboost ', ap
    xgb_gbm_ap1 = np.append(xgb_gbm_ap1, ap)

