#!/usr/bin/python
# -*- coding: utf-8 -*-

from glob import glob
import os

import config as c


def cleanup():
    for f in glob(os.path.join(c.work_dir, '*')):
        if f.endswith('input.txt'):
            continue
        os.remove(f)

if __name__ == '__main__':
    cleanup()