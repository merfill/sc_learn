
import pandas as pd

data = pd.read_csv('data/train_normalized.csv', sep=',', encoding='utf8')
d1 = data[data['target'] == 1]
d0 = data[data['target'] == 0]

n=1
l = int(len(d1)/n)
for i in range(0, n):
    d_1 = d1[l*i:l*(i+1)]
    d_0 = data[data['target'] == 0].sample(int(l*.7))
    d = pd.concat([d_1, d_0])
    print len(d_1)
    print len(d)

    df = d.sample(frac=1)
    df.to_csv('data/train_normalized_{}.csv'.format(i), index=False)
