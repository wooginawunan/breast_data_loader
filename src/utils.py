"""
Minor utilities
"""

import sys
from functools import reduce, partial

import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import traceback
import logging
import argparse
import optparse
import datetime
import sys
import pprint
import types
import time
import copy
import subprocess
import glob
from collections import OrderedDict
import os
import signal
import atexit
import json
import inspect

from logging import handlers

import argh
import gin
from gin.config import _OPERATIVE_CONFIG

import torch
from torch.nn.modules.module import _addindent

logger = logging.getLogger(__name__)

class logger_breast_ori:
    def __init__(self, log_file_directory, log_file_name, buffer_size=100000):

        self.buffer_size = buffer_size
        self.text_to_log = ''   

        os.makedirs(log_file_directory, exist_ok=True)

        log_file_path = os.path.join(log_file_directory, log_file_name)

        if not os.path.exists(log_file_path):
            print('Creating the log file at:', log_file_path)
        else:
            print('Log file at:', log_file_path, 'already exists. Appending to the existing file.')

        self.log_file = open(log_file_path, 'a')

    def __del__(self):

        if not self.log_file.closed:
            self.log_file.write(self.text_to_log)
            self.log_file.close()

    def log(self, string_to_log="", log_log=True, print_log=True, no_enter=False):

        end_of_line = '' if no_enter else '\n'

        if log_log:
            self.text_to_log += string_to_log + end_of_line

            if len(self.text_to_log) > self.buffer_size:
                self.log_file.write(self.text_to_log)
                self.text_to_log = ''

        if print_log:
            print(string_to_log, end=end_of_line)

    def flush(self):
        self.log_file.write(self.text_to_log)
        self.text_to_log = ''
        self.log_file.flush()
        os.fsync(self.log_file.fileno())

    def close(self):

        self.log_file.write(self.text_to_log)
        self.log_file.close()

def x_to_tensor(data_batch_x):
    tensor_list = [torch.Tensor(convert_img(data_batch_x[i])) for i in range(4)]
    cpu_x = dict(zip(["L-CC", "R-CC", "L-MLO", "R-MLO"], tensor_list))
    return cpu_x

def x_to_device(tensor_x, run_container):
    # Weird hack - multi-GPU can be left on CPU, , but single-device must be on GPU
    if not run_container.is_multi_gpu:
        tensor_x = {k: v.to(run_container.base_device) for k, v in tensor_x.items()}
    return tensor_x

def convert_img(img):
    """
    Convert images from
        [batch_size, H, W, chan]
    to
        [batch_size, chan, H, W
    """
    # return np.expand_dims(np.squeeze(img, 3), 1)
    return np.moveaxis(img, 3, 1)

def x_to_tensor_joint(data_batch_x):
    tensor_list = [torch.Tensor(convert_img(data_batch_x[i])) for i in range(2)]
    cpu_x = dict(zip(["cc", "mlo"], tensor_list))
    return cpu_x

def x_to_tensor_single(data_batch_x, view):
    tensor_list = torch.Tensor(convert_img(data_batch_x[0]))
    cpu_x = {view: tensor_list}
    return cpu_x

def unpickle_from_file(file_name):
    with open(file_name, 'rb') as handle:
        try:
            return pickle.load(handle)
        except ImportError:
            return pd.read_pickle(file_name)

def class_count_breast(y_true, key):
    index_table = {'benign': 0, 'malignant': 1}
    if len(y_true)==0:
        return 0, 0
    else:
        return sum(y_true[:, index_table[key]]), y_true.shape[0]

from contextlib import contextmanager


class Fork(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        self.file1.flush()
        self.file2.flush()


@contextmanager
def replace_logging_stream(file_):
    root = logging.getLogger()
    if len(root.handlers) != 1:
        print(root.handlers)
        raise ValueError("Don't know what to do with many handlers")
    if not isinstance(root.handlers[0], logging.StreamHandler):
        raise ValueError
    stream = root.handlers[0].stream
    root.handlers[0].stream = file_
    try:
        yield
    finally:
        root.handlers[0].stream = stream


@contextmanager
def replace_standard_stream(stream_name, file_):
    stream = getattr(sys, stream_name)
    setattr(sys, stream_name, file_)
    try:
        yield
    finally:
        setattr(sys, stream_name, stream)

def gin_wrap(fnc):
    def main(save_path, config, bindings=""):
        # You can pass many configs (think of them as mixins), and many bindings. Both ";" separated.
        gin.parse_config_files_and_bindings(config.split("#"), bindings.replace("#", "\n"))
        if not os.path.exists(save_path):
            logger.info("Creating folder " + save_path)
            os.system("mkdir -p " + save_path)
        run_with_redirection(os.path.join(save_path, "stdout.txt"),
                             os.path.join(save_path, "stderr.txt"),
                             fnc)(save_path)
    argh.dispatch_command(main)

def run_with_redirection(stdout_path, stderr_path, func):
    print(stdout_path, stderr_path)
    def func_wrapper(*args, **kwargs):
        with open(stdout_path, 'a', 1) as out_dst:
            with open(stderr_path, 'a', 1) as err_dst:
                print(stdout_path)
                print(stderr_path)
                out_fork = Fork(sys.stdout, out_dst)
                err_fork = Fork(sys.stderr, err_dst)
                with replace_standard_stream('stderr', err_fork):
                    with replace_standard_stream('stdout', out_fork):
                        with replace_logging_stream(err_fork):
                            func(*args, **kwargs)

    return func_wrapper