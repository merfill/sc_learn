
import functools
import json
import os
import csv
import numpy as np
import tensorflow as tf

from rnn import model_fn, input_fn

DATADIR = './data'
PARAMS = './rnn_results/params.json'
MODELDIR = './rnn_results/model'


if __name__ == '__main__':
    with open(PARAMS) as f:
        params = json.load(f)

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    path = os.path.join(DATADIR, '{}.csv'.format('test_data'))
    print('\n\n------------- start prediction on {}...\n'.format(path))
    test_inpf = functools.partial(input_fn, path)
    preds_gen = estimator.predict(test_inpf)

    with open(path, 'rb') as f, open('submission.csv', 'w') as ff:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        ff.write('ID_code,target\n')
        for row, preds in zip(reader, preds_gen):
            ff.write('{},{}\n'.format(row['ID_code'], preds['classes']))

