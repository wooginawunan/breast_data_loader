# -*- coding: utf-8 -*-
"""
A gorgeous, self-contained, training loop. Uses Poutyne implementation, but this can be swapped later.
"""

import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import gin

from src.callbacks.callbacks import LambdaCallback, History
from src.framework import Model_Breast

logger = logging.getLogger(__name__)

types_of_instance_to_save_in_csv = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.float128, str)
types_of_instance_to_save_in_history = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.ndarray, np.float128,str)

def _construct_default_callbacks(H, H_batch, save_path, save_freq, custom_callbacks):
    
    history_batch = os.path.join(save_path, "history_batch")
    if not os.path.exists(history_batch):
        os.mkdir(history_batch)

    callbacks = []
    callbacks.append(LambdaCallback(on_epoch_end=partial(_append_to_history_csv, H=H), 
                                    on_batch_end=partial(_append_to_history_csv_batch, H=H_batch)
                               )
    )

    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv, save_path=save_path, H=H, save_with_structure=save_with_structure)))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv_batch, save_path=history_batch, H=H_batch, save_with_structure=save_with_structure)))
    callbacks.append(History())

    return callbacks

def _save_history_csv_batch(epoch, logs, save_path, H):
    # out = ""
    # for key, value in logs.items():
    #     if isinstance(value, (int, float, complex, np.float32, np.float64)):
    #         out += "{key}={value}\t".format(key=key, value=value)

    '''
    saving seperately in files with epoch as names in folder - history_batch 
    '''

    H_tosave = {}
    for key, value in H.items():
        
        if isinstance(value[-1], types_of_instance_to_save_in_csv):
            H_tosave[key] = value
    
    pd.DataFrame(H_tosave).to_csv(os.path.join(save_path, "epoch_%d.csv"%epoch), index=False)


    for key in H.keys():
        H[key] = []

def _save_history_csv(epoch, logs, save_path, H):
    out = ""
    for key, value in logs.items():
        if isinstance(value, types_of_instance_to_save_in_csv):
            out += "{key}={value}\t".format(key=key, value=value)
    logger.info(out)
    
    
    logger.info("Saving history to " + os.path.join(save_path, "history.csv"))
    H_tosave = {}
    for key, value in H.items():
        if isinstance(value[-1], types_of_instance_to_save_in_csv):
            
            H_tosave[key] = value
    pd.DataFrame(H_tosave).to_csv(os.path.join(save_path, "history.csv"), index=False)

def _append_to_history_csv_batch(batch, logs, H):
    if len(H)==0 or len(H)==len(logs) :
        for key, value in logs.items():
            if isinstance(value, (types_of_instance_to_save_in_history)):
                if key not in H:
                    H[key] = [value]
                else:
                    H[key].append(value)

            else:
                pass

def _append_to_history_csv(epoch, logs, H):
    for key, value in logs.items():
        if isinstance(value, types_of_instance_to_save_in_history):
            if key not in H:
                H[key] = [value]
            else:
                H[key].append(value)
        else:
            pass

@gin.configurable
def training_loop(meta_data, config, 
                  save_path,  steps_per_epoch, 
                  train=None, valid=None, validation_per_epoch=1,
                  data_loader=None, validation_steps=None,
                  device_numbers = [0], 
                  custom_callbacks=[], 
                  n_epochs=100, save_freq=1):
    
    assert data_loader is not None and validation_steps is not None, 'Inconsistent setting for breast model'
    
    callbacks = list(custom_callbacks)

    history_csv_path = os.path.join(save_path, "history.csv")
    logger.info("Removing {}".format(history_csv_path))
    os.system("rm " + history_csv_path)
    H, epoch_start = {}, 0
    H_batch = {}

    callbacks += _construct_default_callbacks(H, H_batch, save_path,
                                              save_freq, custom_callbacks)

    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_meta_data(meta_data)
        clbk.set_config(config)
        clbk.set_dataloader(data_loader)
    
    model = Model_Breast()
    
    model.get_x_to_tensor_func(is_multi_gpu)
    _ = model.fit_generator(data_loader,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        initial_epoch=epoch_start,
                        validation_per_epoch=validation_per_epoch,
                        epochs=n_epochs - 1,  # Weird convention
                        verbose=1,
                        callbacks=callbacks)
