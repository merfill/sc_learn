
import pandas as pd

data = pd.read_csv('data/train_data.csv', sep=',', encoding='utf8')
d1 = data[data['target'] == 1]
d0 = data[data['target'] == 0].sample(int(len(d1) * 5))
d = pd.concat([d1, d0])
print len(d1)
print len(d0)
print len(d)

df = d.sample(frac=1)
df.to_csv('data/train_data_cut.csv', index=False)
