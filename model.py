import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import numpy as np


class Model:
    def __init__(self, rnn_size, num_layers, vocab_size, grad_clip, batch_size=1, seq_length=1):

        cell = rnn_cell.BasicLSTMCell(rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * num_layers)

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable('softmax_w', [rnn_size, vocab_size])
            softmax_b = tf.get_variable('softmax_b', [vocab_size])
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding', [vocab_size, rnn_size])
                inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        train = batch_size == 1 and seq_length == 1
        loop_fn = loop if train else None

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                  loop_function=loop_fn, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * seq_length])],
                vocab_size)
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num, prime, temperature):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(a):
            a = a.astype(np.float64)
            a = a.clip(min=1e-20)
            a = np.log(a) / temperature
            a = np.exp(a) / (np.sum(np.exp(a)))
            return np.argmax(np.random.multinomial(1, a, 1))

        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]
            sample = weighted_pick(p)
            char = chars[sample]
            yield char


