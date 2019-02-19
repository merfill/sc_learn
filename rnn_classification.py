
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
RESULTSDIR = './results'


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
    target = [0, 0]
    target[int(row['target'])] = 1
    source = []
    for mul in range(0, 200//dim):
        source_row = []
        for idx in range(0, dim):
            source_row += [float(row['var_' + str(idx*mul)])]
        source += [source_row]
    return (source, len(source)), (target)


def generator_fn(data_file, dim):
    with open(data_file, 'rb') as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            yield parse_fn(row, dim)


def input_fn(data_file, params):
    shapes = (([None, params['dim']], ()), ([2]))
    types = ((tf.float32, tf.int32), (tf.int32))
    defaults = ((0., 0), (0))

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data_file, params['dim']), output_shapes=shapes, output_types=types)
    dataset = dataset.repeat(params['epochs'])
    return (dataset.padded_batch(params.get('batch_size', 50), shapes, defaults).prefetch(1))

def model_fn(features, labels, mode, params):
    # Read vocabs and inputs
    dropout = params['dropout']
    source, source_length = features

    # --- RNN ---

    cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])
    cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])
    _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, source, sequence_length=source_length, dtype=tf.float32)

    # concatenate states and produce projection level
    state_fw, state_bw = states
    cells = []
    for fw, bw in zip(state_fw, state_bw):
        cells += [tf.concat([fw, bw], axis=-1)]
    dense_layer = tf.layers.dense(tf.concat(cells, axis=-1), 256)
    logit = tf.layers.dense(dense_layer, 2)

    prediction = tf.nn.softmax(logit)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred': prediction,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=labels))

        if mode == tf.estimator.ModeKeys.EVAL:
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            metrics = {
                'acc': accuracy,
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op)
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params.get('lr', .001))
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, params.get('clip', .5))
            train_op = optimizer.apply_gradients(zip(grads, vs), global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 10,
        'lr': .001,
        'clip': .5,
        'embedding_size': 100,
        'max_iters': 50,
        'dropout': 0.5,
        'layers': 3,
        'num_oov_buckets': 3,
        'epochs': 1,
        'batch_size': 50,
        'source_vocab_file': os.path.join(DATADIR, 'vocab.source.txt'),
        'target_vocab_file': os.path.join(DATADIR, 'vocab.target.txt'),
    }
    with open('{}/params.json'.format(RESULTSDIR), 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, os.path.join(DATADIR, 'train.csv'), params)
    eval_inpf = functools.partial(input_fn, os.path.join(DATADIR, 'dev.csv'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, RESULTSDIR + '/model', cfg, params)
    mkdir(estimator.eval_dir())
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'acc', 1000, min_steps=20000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        path = os.path.join(DATADIR, '{}.csv'.format(name))
        print('\n\n------------- start prediction on {}...\n'.format(path))
        test_inpf = functools.partial(input_fn, path)
        golds_gen = generator_fn(path, params['dim'])
        preds_gen = estimator.predict(test_inpf)
        for golds, preds in zip(golds_gen, preds_gen):
            ((_, _), (target)) = golds
            print(preds)
            print(target)

    for name in ['test', 'dev']:
        write_predictions(name)

