import timeit
import itertools
import numpy as np

from src.callbacks import ValidationProgressionCallback, ProgressionCallback

class Step:
    def __init__(self, number):
        self.number = number
        self.size = None

def cycle(iterable):  # Equivalent to itertools cycle, without any extra memory requirement
    while True:
        for x in iterable:
            yield x

def _get_step_iterator(steps, generator):
    count_iterator = range(1, steps + 1) if steps is not None else itertools.count(1)
    generator = cycle(generator) if steps is not None else generator
    return zip(count_iterator, generator)

class StepIterator:
    def __init__(self, generator, steps_per_epoch, callback):
        self.generator = generator
        self.steps_per_epoch = steps_per_epoch
        self.callback = callback

        self.losses_sum = 0.
        self.sizes_sum = 0.

    def __iter__(self):
        for step, data in _get_step_iterator(self.steps_per_epoch, self.generator):
            self.callback.on_batch_begin(step, {})
        
            batch_begin_time = timeit.default_timer()

            self.callback.on_forward_begin(step, data) 

            step_data = Step(step)

            yield step_data, data # after we get back from yield, step_data has information recorded

            batch_total_time = timeit.default_timer() - batch_begin_time
            batch_logs = {'batch': step, 'size': step_data.size, 'time': batch_total_time}
            self.callback.on_batch_end(step, batch_logs)

class EpochIterator:
    def __init__(self, train_generator, valid_generator, *,
                 epochs, steps_per_epoch, validation_steps,
                 initial_epoch=1, callback):
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.epochs = epochs
        self._init_steps(train_generator, valid_generator, steps_per_epoch, validation_steps)

        self.initial_epoch = initial_epoch
        self.callback = callback
        self.epoch_logs = []
        self.stop_training = False

        params = {'epochs': self.epochs, 'steps': self.steps_per_epoch}
        self.callback.set_params(params)

    def _init_steps(self, train_generator, valid_generator, steps_per_epoch, validation_steps):
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        if valid_generator is not None:
            if validation_steps is None:
                if hasattr(valid_generator, '__len__'):
                    self.validation_steps = len(valid_generator)
                elif steps_per_epoch is not None:
                    self.validation_steps = steps_per_epoch
        if steps_per_epoch is None and hasattr(train_generator, '__len__'):
            self.steps_per_epoch = len(train_generator)

    def __iter__(self):
        self.callback.on_train_begin({})
        for epoch in range(self.initial_epoch, self.epochs + 1):
            self.callback.on_epoch_begin(epoch, {})
            
            epoch_begin_time = timeit.default_timer()
           
            train_step_iterator = StepIterator(self.train_generator,
                                               self.steps_per_epoch,
                                               self.callback)

            valid_step_iterator = None
            if self.valid_generator is not None:
                valid_step_iterator = StepIterator(self.valid_generator,
                                                   self.validation_steps,
                                                   ValidationProgressionCallback(self.validation_steps))

            yield train_step_iterator, valid_step_iterator

            epoch_total_time = timeit.default_timer() - epoch_begin_time

            epoch_log = {'epoch': epoch, 'time': epoch_total_time}
            self.callback.on_epoch_end(epoch, epoch_log)

            self.epoch_logs.append(epoch_log)

            if self.stop_training:
                break

        self.callback.on_train_end({})
