
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

# Scikit-learn
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

train_data = pd.read_csv('data/train_normalized.csv', sep=',', encoding='utf8')
x_train = train_data.drop(['ID_code', 'target'], axis=1)
y_train = train_data['target']

data = pd.read_csv('data/test_normalized.csv', sep=',', encoding='utf8')
print(data.shape)

x = data.drop(['ID_code'], axis=1)
xgb_gbm = xgb.XGBClassifier()
xgb_gbm.load_model('model_0.xgb')

xgb_gbm.fit(x_train, y_train)

pred = xgb_gbm.predict(x)
pp = pd.DataFrame(pred)
print pp.head()
print pp.iloc[:,0]
ppp = pd.DataFrame(columns=['ID_code', 'target'])
ppp['ID_code'] = data['ID_code']
ppp['target'] = pp.iloc[:,0]
print ppp.shape
ppp.to_csv('submission.csv', index=False)


