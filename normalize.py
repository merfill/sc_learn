
import pandas as pd
from sklearn import preprocessing

# prepare train data
train = pd.read_csv('data/train_data.csv', sep=',', encoding='utf8')
c_train = train.drop(['ID_code'], axis=1)
c_train = c_train.drop(['target'], axis=1)
x_train = c_train.values

# prepare test data
test = pd.read_csv('data/test_data.csv', sep=',', encoding='utf8')
c_test = test.drop(['ID_code'], axis=1)
x_test = c_test.values

#min_max_scaler = preprocessing.MinMaxScaler()
#scaler = preprocessing.StandardScaler().fit(x_train)
scaler = preprocessing.RobustScaler().fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

def write(x_scaled, c, d, path, work_on_target):
    df = pd.DataFrame(x_scaled)
    df.columns = c.columns
    df['ID_code'] = d['ID_code']
    if work_on_target:
        df['target'] = d['target']
    df.to_csv(path, index=False)

write(x_train_scaled, c_train, train, 'data/train_normalized.csv', True)
write(x_test_scaled, c_test, test, 'data/test_normalized.csv', False)
