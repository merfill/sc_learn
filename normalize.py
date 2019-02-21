
import pandas as pd
from sklearn import preprocessing

prefix = 'train'
work_on_target = True

data = pd.read_csv('data/{}_data_cut.csv'.format(prefix), sep=',', encoding='utf8')
print data.head(3)
d = data.drop(['ID_code'], axis=1)
if work_on_target:
    d = d.drop(['target'], axis=1)
x = d.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns = d.columns
df['ID_code'] = data['ID_code']
if work_on_target:
    df['target'] = data['target']
print df.head(3)
df.to_csv('data/{}_normalized.csv'.format(prefix), index=False)

