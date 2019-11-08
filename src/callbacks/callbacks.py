# -*- coding: utf-8 -*-
"""
Callbacks implementation. Inspired by Keras.
"""
import gin
import sys
import os
import timeit
import pickle
import logging
import time
import datetime
import json
import copy

from gin.config import _OPERATIVE_CONFIG
from src.utils import class_count_breast


logger = logging.getLogger(__name__)

class CallbackList:
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_forward_begin(self, batch, data):
        for callback in self.callbacks:
            callback.on_forward_begin(batch, data)

    def on_backward_end(self, batch):
        for callback in self.callbacks:
            callback.on_backward_end(batch)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_train_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_epoch_begin'):
                callback.on_train_epoch_begin(epoch, logs)

    def on_val_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_val_epoch_begin'):
                callback.on_val_epoch_begin(epoch, logs)

    def on_test_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_test_epoch_begin'):
                callback.on_test_epoch_begin(epoch, logs)

    def on_val_batch_end(self, batch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_val_batch_end(batch, logs)

    def __iter__(self):
        return iter(self.callbacks)

class Callback(object):
    def __init__(self):
        pass

    def set_config(self, config):
        self.config = config

    def set_meta_data(self, meta_data):
        self.meta_data = meta_data

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_params(self, params):
        self.params = params

    def set_dataloader(self, data):
        self.data = data

    def get_dataloader(self):
        return self.data

    def get_config(self):
        return self.config

    def get_meta_data(self):
        return self.meta_data

    def get_params(self):
        return self.params

    def get_save_path(self):
        return self.save_path

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_forward_begin(self, batch, data):
        pass

    def on_backward_end(self, batch):
        pass

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass

    def on_train_epoch_begin(self, epoch, logs):
        pass

    def on_val_epoch_begin(self, epoch, logs):
        pass

    def on_test_epoch_begin(self, epoch, logs):
        pass

    def on_val_batch_end(self, batch, logs):
        pass
            
class History(Callback):
    """
    History callback.

    By default saves history every epoch, can be configured to save also every k examples
    """
    def __init__(self, save_every_k_examples=-1):
        self.examples_seen = 0
        self.save_every_k_examples = save_every_k_examples
        self.examples_seen_since_last_population = 0
        super(History, self).__init__()

    def on_train_begin(self, logs=None):
        # self.epoch = []
        self.history = {}
        self.history_batch = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # self.epoch.append(epoch)
        for k, v in logs.items():
            #self.history.setdefault(k, []).append(v)

            if k.endswith("labels"):# and (k not in self.history):
                # we don't need to save labels every epoch.
                #self.history[k] = v
                pass
            else:
                self.history.setdefault(k, []).append(v)

        if self.save_path is not None:
            pickle.dump(self.history, open(os.path.join(self.save_path, "history.pkl"), "wb"))
            if self.save_every_k_examples != -1:
                pickle.dump(self.history_batch, open(os.path.join(self.save_path, "history_batch.pkl"), "wb"))

    def on_batch_end(self, epoch, logs=None):
        # Batches starts from 1
        if self.save_every_k_examples != -1:
            # if getattr(self.model, "history_batch", None) is None:
            #     setattr(self.model, "history_batch", self)
            assert "size" in logs
            self.examples_seen += logs['size']
            logs['examples_seen'] = self.examples_seen
            self.examples_seen_since_last_population += logs['size']

            if self.examples_seen_since_last_population > self.save_every_k_examples:
                for k, v in logs.items():
                    self.history_batch.setdefault(k, []).append(v)
                self.examples_seen_since_last_population = 0

class LambdaCallback(Callback):
    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None):
        super(LambdaCallback, self).__init__()
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda epoch, logs: None
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda epoch, logs: None
        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda batch, logs: None
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda batch, logs: None
        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None
        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None

@gin.configurable
class ProgressionCallback(Callback):
    
    def on_train_begin(self, logs):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

    def on_train_end(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        self.step_times_sum = 0.
        self.epoch = epoch
        sys.stdout.write("\rEpoch %d/%d" % (self.epoch, self.epochs))
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs):
        epoch_total_time = logs['time']

        metrics_str =''
        iol_str = ''
        if self.steps is not None:
            print("\rEpoch %d/%d %.2fs Step %d/%d: %s. %s" %
                  (self.epoch, self.epochs, epoch_total_time, self.steps, self.steps, metrics_str, iol_str))

        else:
            print("\rEpoch %d/%d %.2fs: Step %d/%d: %s. %s" %
                  (self.epoch, self.epochs, epoch_total_time, self.last_step, self.last_step, metrics_str, iol_str))

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        self.step_times_sum += logs['time']

        metrics_str = ''
        iol_str = ''
        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch)
            
            sys.stdout.write("\rEpoch %d/%d ETA %.0fs Step %d/%d: %s. %s" %
                             (self.epoch, self.epochs, remaining_time, batch, self.steps, metrics_str, iol_str))
            if 'cumsum_iol' in iol_str: sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("\rEpoch %d/%d %.2fs/step Step %d: %s. %s" %
                             (self.epoch, self.epochs, times_mean, batch, metrics_str, iol_str))
            sys.stdout.flush()
            self.last_step = batch

class ValidationProgressionCallback(Callback):
    def __init__(self, 
                 steps):
        self.params = {}
        self.params['steps'] = steps

        super(ValidationProgressionCallback, self).__init__()

    def on_batch_begin(self, batch, logs):
        if batch==1:
            self.step_times_sum = 0.
        
        self.batch_begin_time = timeit.default_timer()
        
        self.steps = self.params['steps']

    def on_batch_end(self, batch, logs):
        self.step_times_sum += (timeit.default_timer() - self.batch_begin_time)

        if batch==1:
            sys.stdout.write('\n')
        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch)

            sys.stdout.write("\rvalidation ETA %.0fs Step %d/%d" %
                             (remaining_time, batch, self.steps))
            sys.stdout.flush()
        else:
            sys.stdout.write("\rvalidation %.2fs/step Step %d" %
                             (times_mean, batch))
            sys.stdout.flush()
            self.last_step = batch
              
@gin.configurable
class BreastDataLoader(Callback):
    def __init__(self, 
                 mode="multiclass_cancer_sides",
                 ):
        #self.view_weights = view_weights
        
        super(BreastDataLoader, self).__init__()
        self.mode =  mode

    def on_train_epoch_begin(self, epoch, logs):
        current_random_seed = self.data.seed_shifter.get_seed(phase='training', epoch_number=epoch)
        self.data.start_training_epoch(random_seed=current_random_seed, mode=self.mode)

    def on_val_epoch_begin(self, epoch, logs):
        current_random_seed = self.data.seed_shifter.get_seed(phase='validation', epoch_number=epoch)
        self.data.start_validation_epoch(random_seed=current_random_seed, mode=self.mode)

    def on_test_epoch_begin(self, epoch, logs):
        current_random_seed = self.data.seed_shifter.get_seed(phase='test', epoch_number=epoch)
        self.data.start_test_epoch(random_seed=current_random_seed, mode=self.mode)
