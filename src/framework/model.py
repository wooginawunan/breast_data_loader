####design a callback to return/save prediction of validation set for the best epoch.
############especially for single model to do ensemble
import numpy as np

from src.callbacks import CallbackList, ProgressionCallback
from .iterators import *

class Model_Breast:
    def __init__(self, label_mode = 'multiclass_cancer_sides'):
       self.label_mode = label_mode

    def _validate(self, step_iterator, data_loader):

        for step, (indices, raw_data_batch) in step_iterator:
            
            data_batch = data_loader.process_data_batch(raw_data_batch)
            cancer_label_minibatch = data_loader.get_cancer_label_minibatch(indices, 'validation', self.label_mode)

    def fit_generator(self, data_loader, 
                      *,
                      validation_per_epoch=1,
                      epochs=1000, steps_per_epoch=None, validation_steps=None,
                      initial_epoch=1, verbose=True, callbacks=[]):
        if verbose: 
            callbacks =  callbacks + [ProgressionCallback()]

        callback_list = CallbackList(callbacks) 

        epoch_iterator = EpochIterator(data_loader.give_training_minibatch(), 
                                       data_loader.give_validation_minibatch(),
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch  , #have to get this number before and pass it in
                                       validation_steps=validation_steps,
                                       initial_epoch=initial_epoch,
                                       callback=callback_list)

        self.steps_per_epoch = steps_per_epoch

        for train_step_iterator, valid_step_iterator in epoch_iterator:

            epoch = epoch_iterator.epoch_logs[-1]['epoch']+1 if len(epoch_iterator.epoch_logs)>0 else 1

            callback_list.on_train_epoch_begin(epoch, epoch_iterator.epoch_logs)

            for step, (indices, raw_data_batch) in train_step_iterator:

                data_batch = data_loader.process_data_batch(raw_data_batch)
                cancer_label_minibatch = data_loader.get_cancer_label_minibatch(indices, 'training', self.label_mode)
            
                step.size = len(indices)
            
            if (valid_step_iterator is not None and epoch%validation_per_epoch==0):

                callback_list.on_val_epoch_begin(epoch, epoch_iterator.epoch_logs)

                self._validate(valid_step_iterator, data_loader)

        return epoch_iterator.epoch_logs


