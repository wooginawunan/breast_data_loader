# -*- coding: utf-8 -*-
"""
Callback module (inspired by Keras).
"""
from .callbacks import *

logger = logging.getLogger(__name__)


def get_callback(clb_name, verbose=1, **kwargs):
    if clb_name in callbacks.__dict__:
        return callbacks.__dict__[clb_name](**kwargs)
    else:
        if verbose:
            logger.warning("Couldn't find {} callback. Skipping.".format(clb_name))
        return None
