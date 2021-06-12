"""Utility for accessing files."""
# from __future__ import annotations
import os
from pathlib import Path
import shutil
from enum import Enum, auto
from datetime import datetime
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

import fade
_absolute_global_root_path = Path(os.path.realpath(fade.__file__)).parents[1]
global_root_path = _absolute_global_root_path  # absolute path.


class FileManager(object):
    """All the path is based on the `src` folder."""
    def __init__(self):
        pass

    task_path = ""  # TODO set the task path in experiments.

    class CleanMode(Enum):
        all = auto()
        log = auto()
        out = auto()
        none = auto()
        # old = auto()  # TODO only keep the lastest

    @staticmethod
    def clean_dir(dir_to_clean: str):
        if os.path.exists(dir_to_clean):
            logger.info(f"Recursively removing dir: {dir_to_clean}")

            def rm_error_handler(function, path, excinfo):
                e = excinfo[1]
                if isinstance(e, OSError) and e.errno == 39:  # 39: e.strerror == "Directory not empty":
                    # print(f"Delete non-empty dir: {path}")
                    # shutil.rmtree(path, ignore_errors=True)
                    logger.error(f"Fail to delete folder '{path}' due to some ramianing files. "
                                 f"Try to close the tensorboard or any occupying process.")
                raise e
            try:
                shutil.rmtree(dir_to_clean, onerror=rm_error_handler)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")

    @classmethod
    def generate_path_with_root(cls, root, subdir, is_dir=False, create_new=True,
                                overwrite=True, verbose=False, return_date=False):
        """Generate path given root and sub-dir.

        :param root: The root of the path, e.g., 'out', 'data'. This is used for specify the function.
        :param subdir: (or filename) Usually, this presents the unique name of the storage, e.g., data name.
        :param is_dir: True if the `subdir` is a dir, else it is treated as a file.
        :param create_new: Create new folder and sub-folders if not exists.
        :param overwrite: Overwrite folders if exists.
        :param verbose: Print info when overwriting or creating.
        :param return_date: Return the date when the file/dir was modified.
        :return: Generated path str. str of modification time of the file (if required).
        """
        path = os.path.join(global_root_path, root, cls.task_path, subdir)
        if create_new:
            if is_dir:
                fld = path
            else:
                fld = os.path.dirname(path)
            if os.path.exists(path):  # os.path.dirname(path)):
                if overwrite:
                    pass
                    # if verbose:
                    #     logger.warning("'{}' already exists. Overwrite...".format(fld))
                else:
                    return path
            elif verbose:
                logger.debug("Creating dir: {}".format(fld))
            os.makedirs(fld, exist_ok=overwrite)
        if return_date:
            if os.path.exists(path):
                import pathlib
                pp = pathlib.Path(path)
                mtime = datetime.fromtimestamp(pp.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            else:
                mtime = "n/a"
            return path, mtime
        return path

    @classmethod
    def out(cls, filename, **kwargs):
        """Generate path given folder.

        :param filename: This presents the unique name of the storage, e.g., data name.
        :param create_new: Create new folder and sub-folders if not exists.
        :param overwrite: Overwrite folders if exists.
        :param verbose: Print info when overwriting or creating.
        :return: Generated path str.
        """
        return cls.generate_path_with_root("out", filename, **kwargs)

    @classmethod
    def log(cls, filename, **kwargs):
        """Generate path given folder.

        :param filename: This presents the unique name of the storage, e.g., data name.
        :param create_new: Create new folder and sub-folders if not exists.
        :param overwrite: Overwrite folders if exists.
        :param verbose: Print info when overwriting or creating.
        :return: Generated path str.
        """
        return cls.generate_path_with_root("log", filename, **kwargs)

    @classmethod
    def data(cls, filename, **kwargs):
        """Generate path given folder.

        :param filename: data filename.
        :param create_new: Create new folder and sub-folders if not exists.
        :param overwrite: Overwrite folders if exists.
        :param verbose: Print info when overwriting or creating.
        :return: Generated path str.
        """
        return cls.generate_path_with_root("data", filename, **kwargs)

    hpcc_jobid = ""
    __logid = None  # type: str
    __logdir = None  # type: str

    @classmethod
    def get_logid(cls):
        """Return a unique id for the current log."""
        if cls.__logid is None:
            cls.__logid = datetime.now().strftime("%Y%m%d-%H%M%S")
            if len(cls.hpcc_jobid) > 0:
                cls.__logid = cls.hpcc_jobid + "@" + cls.__logid
        return cls.__logid

    @classmethod
    def get_logdir(cls, new=True):
        """Get the log path. Set `new` False if an existing path is expected."""
        if cls.__logdir is None:
            cls.__logdir = cls.log(cls.get_logid(), is_dir=True, create_new=new,
                                   overwrite=new, verbose=True)
        return cls.__logdir


def save_current_fig(name="untitled"):
    file_name = FileManager.out(f"./fig/{name}.pdf",
                                create_new=True, overwrite=True)
    # file_name = FileManager.out(f"fig/{name}.pdf", create_new=True, overwrite=True)
    plt.savefig(file_name, bbox_inches="tight")
    print(f"save figure => {file_name}")
