
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Gradient Boosting
#import lightgbm as lgb
import xgboost as xgb

# Scikit-learn
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


data = pd.read_csv('data/train_normalized.csv', sep=',', encoding='utf8')
print(data.shape)
d1 = data[data['target'] == 1]
d0 = data[data['target'] == 0].sample(int(len(d1)*.5))
d = pd.concat([d1, d0])

train = d.drop(['ID_code', 'target'], axis=1)
y = d['target']
print(train.shape)

# Transforming the problem into a classification (unbalanced)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)

MAX_ROUNDS = 1000
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

#for i, (train_index, test_index) in enumerate(skf.split(train, y)):
# Create data for this fold
X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.33, random_state=42)
#y_train, y_valid = y[train_index], y[test_index]
#X_train, X_valid = train.iloc[train_index,:], train.iloc[test_index,:]

#print '\nFold ', i

xgb_gbm = xgb.XGBClassifier(max_depth=7, n_estimators=MAX_ROUNDS, learning_rate=0.06)
eval_set=[(X_train, y_train), (X_valid, y_valid)]
xgb_gbm.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, eval_metric='logloss', verbose=True)
print 'Best iteration xgboost_gbm = ', xgb_gbm.get_booster().best_iteration

# Storing and reporting results of the fold
xgb_gbm_iter1 = np.append(xgb_gbm_iter1, xgb_gbm.get_booster().best_iteration)

pred = xgb_gbm.predict(X_valid)
ap = accuracy_score(y_valid, pred)
print 'xgboost ', ap
xgb_gbm_ap1 = np.append(xgb_gbm_ap1, ap)

xgb_gbm.save_model('model.xgb')

t = pd.read_csv('data/test_normalized.csv', sep=',', encoding='utf8')
print('test: ', t.shape)
x = t.drop(['ID_code'], axis=1)

p = xgb_gbm.predict(x)
pp = pd.DataFrame(p)
print pp.head()
print pp.iloc[:,0]
ppp = pd.DataFrame(columns=['ID_code', 'target'])
ppp['ID_code'] = t['ID_code']
ppp['target'] = pp.iloc[:,0]
print ppp.shape
ppp.to_csv('xgb_submission.csv', index=False)

