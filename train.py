#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function
import tensorflow as tf

import time
import os
import cPickle

from text_loader import TextLoader
from model import Model

import config as c
import cleanup


def train():
    cleanup.cleanup()
    c.save(c.work_dir)

    data_loader = TextLoader(c.work_dir, c.batch_size, c.seq_length)
    with open(os.path.join(c.work_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(c.rnn_size, c.num_layers, len(data_loader.chars), c.grad_clip, c.batch_size, c.seq_length)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        for e in range(c.num_epochs):
            sess.run(tf.assign(model.lr, c.learning_rate * (c.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(e * data_loader.num_batches + b,
                            c.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))
                if (e * data_loader.num_batches + b) % c.save_every == 0:
                    checkpoint_path = os.path.join(c.work_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    train()
