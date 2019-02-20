
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
RESULTSDIR = './cnn_results'


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


def parse_fn(row):
    target = 0
    if 'target' in row:
        target = int(row['target'])
    source = []
    for idx in range(0, 200):
        source += [float(row['var_' + str(idx)])]
    return (source), (target)


def generator_fn(data_file):
    with open(data_file, 'rb') as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            yield parse_fn(row)


def input_fn(data_file, params):
    shapes = (([None]), ())
    types = ((tf.float32), (tf.int32))
    defaults = ((0.), (0))

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data_file), output_shapes=shapes, output_types=types)
    dataset = dataset.repeat(params['epochs'])
    return (dataset.padded_batch(params.get('batch_size', 50), shapes, defaults).prefetch(1))

def model_fn(features, labels, mode, params):
    # intputs
    source = tf.reshape(features, [-1, 10, 20, 1])

    # convolutional layers
    conv1 = tf.layers.conv2d(inputs=source, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # dense layer
    pool2_flat = tf.reshape(pool1, [-1, 5 * 10 * 32])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probs": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=params.get('lr', .001))
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "acc": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':
    # Params
    params = {
        'lr': .001,
        'epochs': 10,
        'batch_size': 100,
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
        golds_gen = generator_fn(path)
        preds_gen = estimator.predict(test_inpf)
        err = 0
        alls = 0
        for golds, preds in zip(golds_gen, preds_gen):
            ((_), (target)) = golds
            alls += 1
            if np.argmax(preds['probs']) != np.argmax(target):
                err += 1
        print('alls: ', alls)
        print('errs: ', err)
        print('acc: ', 1. - (float(err) / alls))

    for name in ['test', 'dev']:
        write_predictions(name)

