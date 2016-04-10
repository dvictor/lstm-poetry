#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf

import os
import sys
import cPickle

from model import Model
import config as c


START = '''A butterfly in '''  # start from this text - can be multiline
COUNT = 400       # number of characters to generate
TEMPERATURE = .5  # try values between 0.1 and 1.5


if __name__ == '__main__':

    print('Sampling %s chars at diversity %s\n' % (COUNT, TEMPERATURE))

    c.load(c.work_dir)
    with open(os.path.join(c.work_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(c.rnn_size, c.num_layers, len(vocab), c.grad_clip)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(c.work_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        sys.stdout.write(START)
        for ch in model.sample(sess, chars, vocab, COUNT, START, TEMPERATURE):
            sys.stdout.write(ch)
            sys.stdout.flush()
        sys.stdout.write('\n')


