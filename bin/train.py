#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script. Example run command: bin/train.py save_to_folder configs/cnn.gin.
"""
import sys
import gin
from gin.config import _CONFIG
import logging
from functools import partial

#sys.path.append('../src')
from src import data
from src.training_loop import training_loop
from src.callbacks import get_callback
from src.utils import gin_wrap, logger_breast_ori

logger = logging.getLogger(__name__)

@gin.configurable
def train(save_path, data_class, label_mode = 'multiclass_cancer_sides', batch_size=128, callbacks=['BreastDataLoader']):
    '''
    data_class: 'data_with_segmentations_gin' or 'data_gin'
    '''

    # Create dynamically dataset generators
    data_loader = data.__dict__[data_class](logger_breast_ori(save_path, 'output_log.log'), 
                                            minibatch_size=batch_size)

    # Create dynamically callbacks
    callbacks_constructed = []
    
    for name in callbacks:

        clbk = get_callback(name, verbose=0)
        if clbk is not None:
            callbacks_constructed.append(clbk)

    
    if data_loader.parameters['train_sampling_mode'] == 'normal':
        training_oversampled_indices = data_loader.data_list_training
    else:
        training_oversampled_indices = data_loader.train_sampler.sample_indices(data_loader.get_train_labels_cancer('multiclass_cancer_sides'), random_seed=0)
    
    steps_per_epoch = (len(training_oversampled_indices) - 1) // batch_size + 1
    validation_steps = (len(data_loader.data_list_validation) - 1) // batch_size + 1 
    
    logger.info('samples_per_training_epoch=%d; steps_per_epoch=%d'%(len(training_oversampled_indices), steps_per_epoch))
    logger.info('samples_per_evaluation_epoch=%d; validation_steps=%d'%(len(data_loader.data_list_validation), validation_steps))

    training_loop(meta_data=None, 
                  label_mode=label_mode,
                  steps_per_epoch=steps_per_epoch, 
                  validation_steps=validation_steps, 
                  data_loader=data_loader,
                  save_path=save_path, config=_CONFIG,
                  custom_callbacks=callbacks_constructed)

if __name__ == "__main__":
    gin_wrap(train)
