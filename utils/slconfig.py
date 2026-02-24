# ==========================================================
# Modified from mmcv
# ==========================================================
import os
import sys
import ast
import json
import shutil
import logging
import tempfile

from importlib import import_module
from argparse import Action
from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text', 'get', 'dump', 'merge_from_dict']


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val.lower() in ['none', 'null']:
            return None
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def log_args(args, logger=logging.getLogger("base")):
    """
    Log all arguments in a neat format.
    """
    logger.info("========== ARGS ==========")
    for key, value in sorted(vars(args).items()):
        logger.info(f"{key:<22}: {value}")
    logger.info("==========================\n")


