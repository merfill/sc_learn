
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train_data_cut.csv', sep=',', encoding='utf8')

train, dev = train_test_split(df)

print df.shape[0]
print train.shape[0]
print dev.shape[0]

train.to_csv('data/train.csv', index=False)
dev.to_csv('data/dev.csv', index=False)
