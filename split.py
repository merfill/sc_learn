
import pandas as pd

df = pd.read_csv('data/train_normalized.csv', sep=',', encoding='utf8')

l = df.shape[0]
tens = int(l * .1)

train = df[ : (l - tens)]
dev = df[(l - tens) : ]

print train.shape[0]
print dev.shape[0]

train.to_csv('data/train.csv', index=False)
dev.to_csv('data/dev.csv', index=False)
