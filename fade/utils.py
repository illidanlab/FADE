import os
import numpy as np
from contextlib import contextmanager
import time
import logging
from datetime import timedelta
import pandas as pd
from hashlib import sha1
from omegaconf import OmegaConf, DictConfig

timer_logger = logging.getLogger("TIME")
timer_logger.setLevel("DEBUG")


def set_coloredlogs_env():
    os.environ['COLOREDLOGS_LOG_FORMAT'] = "%(name)s[%(process)d] %(levelname)s %(message)s"
    os.environ['COLOREDLOGS_LEVEL_STYLES'] = \
        'spam=22;debug=28;verbose=34;notice=220;warning=184;' \
        'info=101;success=118,bold;error=161,bold;critical=background=red'


def flatten_dict(d: dict, sep=".", handle_list=True):
    if handle_list:
        return _flatten(d, sep)
    else:
        df = pd.json_normalize(d, sep=sep)
        return df.to_dict(orient="records")[0]


def _flatten(input_dict, separator='_', prefix=''):
    """Flatten a dict including nested list.
    Ref: https://stackoverflow.com/a/55834113/3503604
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict) and value:
            deeper = _flatten(value, separator, prefix+key+separator)
            output_dict.update({key2: val2 for key2, val2 in deeper.items()})
        elif isinstance(value, list) and value:
            for index, sublist in enumerate(value):
                if isinstance(sublist, dict) and sublist:
                    deeper = _flatten(sublist, separator, prefix+key+separator+str(index)+separator)
                    output_dict.update({key2: val2 for key2, val2 in deeper.items()})
                else:
                    output_dict[prefix+key+separator+str(index)] = value
        else:
            output_dict[prefix+key] = value
    return output_dict


def hash_config(cfg, select_keys=[], exclude_keys=[]):
    if len(exclude_keys) > 0:
        for k in exclude_keys:
            if k in select_keys:
                raise ValueError(f"Try to exclude key {k} which is the selected keys:"
                                 f"{select_keys}")
        cfg = OmegaConf.masked_copy(cfg, [k for k in cfg if k not in exclude_keys])
    if len(select_keys) > 0:
        cfg = OmegaConf.masked_copy(cfg, select_keys)
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    hash_hex = sha1(cfg_str.encode('utf-8')).hexdigest()
    return hash_hex


@contextmanager
def _log_time_usage(prefix="", debug_only=True):
    '''log the time usage in a code block
    prefix: the prefix text to show

    Refer: https://stackoverflow.com/a/37429875/3503604
    '''
    start = time.time()
    try:
        info = f"=== {prefix} time block ==="
        if debug_only:
            timer_logger.debug(info)
        else:
            print(info)
        yield
    finally:
        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        info = f"=== {prefix} elapsed: {elapsed} ==="
        if debug_only:
            timer_logger.debug(info)
        else:
            print(info)


def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len < 3:
        return data
    for i in range(len(data)):
        x = data[i]
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        results.append(y[window_len - 1:])
    return np.array(results)


def get_file_date(fname):
    import pathlib
    import datetime
    file = pathlib.Path(fname)
    mtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
    return mtime
