
from __future__ import print_function
import functools
import json
import logging
import os
import sys
import csv
import errno
import random

import numpy as np
import tensorflow as tf


DATADIR = './data'
RESULTSDIR = './rnn_results'


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


# Logging
tf.logging.set_verbosity(logging.INFO)

mkdir(RESULTSDIR)
handlers = [
    logging.FileHandler(RESULTSDIR + '/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(row, dim):
    target = 0
    if 'target' in row:
        target = int(row['target'])
    source = []
    for idx in range(0, 200):
        source += [float(row['var_' + str(idx)])]
    return (source, len(source)), (target)


def generator_fn(data_file, dim):
    with open(data_file, 'rb') as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            yield parse_fn(row, dim)


def input_fn(data_file, params):
    shapes = (([None], ()), ())
    types = ((tf.float32, tf.int32), (tf.int32))
    defaults = ((0., 0), (0))

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data_file, params['dim']), output_shapes=shapes, output_types=types)
    dataset = dataset.repeat(params['epochs'])
    return (dataset.padded_batch(params.get('batch_size', 50), shapes, defaults).prefetch(1))

def model_fn(features, labels, mode, params):
    # Read vocabs and inputs
    _features, _ = features
    _source = tf.reshape(_features, [-1, 50, 4])
    source = tf.layers.batch_normalization(_source, training=mode == tf.estimator.ModeKeys.TRAIN)
    batch_size = tf.shape(source)[0]
    source_length = tf.fill([batch_size], 50)

    d1 = tf.layers.dense(source, params['ext_dim'])
    #print_op = tf.print(tf.shape(d1))
    #with tf.control_dependencies(print_op):
    d2 = tf.layers.dense(d1, params['ext_dim'] * 2)
    d3 = tf.layers.dense(d2, params['ext_dim'])

    # --- RNN ---

    cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])
    cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])
    _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, source, sequence_length=source_length, dtype=tf.float32)

    # concatenate states and produce projection level
    state_fw, state_bw = states
    cells = []
    for fw, bw in zip(state_fw, state_bw):
        cells += [tf.concat([fw, bw], axis=-1)]

    dense_layer_1 = tf.layers.dense(tf.concat(cells, axis=-1), params['dense_dim'])
    dropout1 = tf.layers.dropout(inputs=dense_layer_1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense_layer_2 = tf.layers.dense(dropout1, params['dense_dim'] * 2)
    dropout2 = tf.layers.dropout(inputs=dense_layer_2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense_layer_3 = tf.layers.dense(dropout2, params['dense_dim'] * 3)
    dropout3 = tf.layers.dropout(inputs=dense_layer_3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense_layer_4 = tf.layers.dense(dropout3, params['dense_dim'] * 2)
    dropout4 = tf.layers.dropout(inputs=dense_layer_4, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense_layer_5 = tf.layers.dense(dropout4, params['dense_dim'])
    dropout5 = tf.layers.dropout(inputs=dense_layer_5, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(dropout5, 2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probs": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.get('lr', .001))
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "acc": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 256,
        'dense_dim': 512,
        'ext_dim': 128,
        'lr': .001,
        'layers': 5,
        'epochs': 5,
        'batch_size': 50,
    }
    with open('{}/params.json'.format(RESULTSDIR), 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, os.path.join(DATADIR, 'train.csv'), params)
    eval_inpf = functools.partial(input_fn, os.path.join(DATADIR, 'dev.csv'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, RESULTSDIR + '/model', cfg, params)
    mkdir(estimator.eval_dir())
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'acc', 1000, min_steps=20000, run_every_secs=300)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=300)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        path = os.path.join(DATADIR, '{}.csv'.format(name))
        print('\n\n------------- start prediction on {}...\n'.format(path))
        test_inpf = functools.partial(input_fn, path)
        golds_gen = generator_fn(path, params['dim'])
        preds_gen = estimator.predict(test_inpf)
        err = 0
        alls = 0
        for golds, preds in zip(golds_gen, preds_gen):
            ((_, _), (target)) = golds
            alls += 1
            if preds['classes'] != target:
                err += 1
        print('alls: ', alls)
        print('errs: ', err)
        print('acc: ', 1. - (float(err) / alls))

    for name in ['dev']:
        write_predictions(name)

