# -*- coding = utf-8 -*-

import json
import sys
import os

work_dir = 'data/example'
#work_dir = 'data/poet'
rnn_size = 512
num_layers = 3   # number of LSTM cells
grad_clip = 5.   # clip gradients at this value
batch_size = 50
seq_length = 50
num_epochs = 50
learning_rate = .002
decay_rate = .97  # decay rate for rmsprop
save_every = 1000

_saves = ['rnn_size', 'num_layers', 'grad_clip']
module = sys.modules[__name__]


def save(path):
    d = dict((name, getattr(module, name)) for name in _saves)
    with open(os.path.join(path, 'config.json'), 'wb') as fh:
        json.dump(d, fh)


def load(path):
    with open(os.path.join(path, 'config.json'), 'rb')as fh:
        for name, value in json.load(fh).iteritems():
            setattr(module, name, value)
