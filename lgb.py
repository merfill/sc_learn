

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Gradient Boosting
import lightgbm as lgb

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


params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'dart',
    'learning_rate': 0.01,
    'max_bin': 15,
    'max_depth': 17,
    'num_leaves': 63,
    'subsample': 0.8,
    'subsample_freq': 5,
    'colsample_bytree': 0.8,
    'reg_lambda': 7,
    'n_jobs': 10
}

X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=42)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=50)

p0 = gbm.predict(X_valid)
pred = np.where(p0 > 0.5, 1, 0)
ap = accuracy_score(y_valid, pred)
print 'accuracy: ', ap

gbm.save_model('model.xgb')

t = pd.read_csv('data/test_normalized.csv', sep=',', encoding='utf8')
print('test: ', t.shape)
x = t.drop(['ID_code'], axis=1)

p_ = gbm.predict(x)
p = np.where(p_ > 0.5, 1, 0)
pp = pd.DataFrame(p)
ppp = pd.DataFrame(columns=['ID_code', 'target'])
ppp['ID_code'] = t['ID_code']
ppp['target'] = pp.iloc[:,0]
print ppp.shape
ppp.to_csv('lgb_submission.csv', index=False)


