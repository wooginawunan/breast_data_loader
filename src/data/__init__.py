# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid.
"""
import logging
 
from .breast_data import data_gin, data_with_segmentations_gin

logger = logging.getLogger(__name__)
