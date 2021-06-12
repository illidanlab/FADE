from __future__ import annotations
import torch
import os
import sys
import numpy as np
import pickle as pk
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import wandb
from omegaconf import ListConfig
import logging

from fade.model import get_model
import copy
from typing import Type, List, TYPE_CHECKING

from fade.file import FileManager
from fade.data.utils import basic_stat, basic_stat_fed_dict
from fade.data.utils import read_fed_dataset, extract_user_data

from ..utils import hash_config

if TYPE_CHECKING:
    from fade.user.base import User


def import_fed_server(algorithm) -> Type[Server]:
    # These has to be loaded for access the `if_personal_local_adaptation()`.
    import importlib
    module = importlib.import_module(f"fade.server.{algorithm}")
    return getattr(module, algorithm)


class ServerAgent:
    """A mock class which has no functionality only for the convenience of access data/model.
    After training, the class can be used to re-locate the stored information.

    you can pass `model = (None, model_name)` to only create an agenet.
    """

    def _init_by_args(self, args):
        self.full_config = args
        # Set up the main attributes
        self.dataset = args.dataset.name
        # To load data on need. Formed as tuple (clients, groups, train_data, test_data)
        self.preloaded_data = None

        self.algorithm = args.server.alg
        self.beta = args.server.beta
        self.privacy_cfg = args.server.privacy
        self.num_users = args.server.num_users

        self.num_glob_iters = args.num_glob_iters
        self.partial_eval = args.partial_eval

        # group config
        self.user_cfg = args.user
        self.model_cfg = args.model
        self.dataset_cfg = args.dataset

        self.model = None

        self.model_name = args.model.name
        self.loss = args.user.loss

    def __init__(self, dataset=None, model=None, times=10, args=None, **ignored):
        if args is not None:
            self._init_by_args(args)
        else:
            raise RuntimeError(f"args is None.")
        if len(ignored) > 0:
            print(f"Ignore kwargs: {ignored.keys()}")

        # Set up the main attributes
        if isinstance(dataset, str):
            assert not hasattr(self,
                               'dataset'), f"dataset name has been initialized as {self.dataset}. Try to init again with {dataset}"
            self.dataset = dataset
            # To load data on need. Formed as tuple (clients, groups, train_data, test_data)
            self.preloaded_data = None
        elif isinstance(dataset, tuple):
            self.dataset, self.preloaded_data = dataset
            if isinstance(self.preloaded_data, dict):
                self.groups = self.preloaded_data["train"]["hierarchies"]
                self.tr_num_samples, self.total_num_users = basic_stat_fed_dict(
                    self.preloaded_data["train"])
                self.te_num_samples, _ = basic_stat_fed_dict(self.preloaded_data['test'])
            else:
                clients, groups, train_data, test_data = self.preloaded_data
                self.groups = groups
                self.tr_num_samples, self.total_num_users = basic_stat(train_data)
                self.te_num_samples, _ = basic_stat(test_data)
            if self.initiated_by_hydra_config():
                if self.full_config.server.num_users < 0:
                    self.full_config.server.num_users = self.total_num_users
                elif 0 < self.full_config.server.num_users < 1:
                    self.full_config.server.num_users = int(np.ceil(self.total_num_users * self.full_config.server.num_users))
                self.num_users = self.full_config.server.num_users
                if self.full_config.user.local_epochs < 0 and hasattr(self.full_config.user,
                                                                      "total_local_epochs"):
                    assert self.full_config.user.total_local_epochs >= self.num_users, \
                        f"self.full_config.user.total_local_epochs " \
                        f"({self.full_config.user.total_local_epochs}) is less than self.num_users " \
                        f"({self.num_users})"
                    self.full_config.user.local_epochs = int(
                        self.full_config.user.total_local_epochs / self.num_users)

        if isinstance(model, str):
            assert not hasattr(self,
                               'model_name'), f"dataset name has been initialized as {self.model_name}. Try to init again with {model}"
            self.model = None
            self.model_name = model
        elif isinstance(model, tuple):
            self.model = copy.deepcopy(model[0]) if model[0] is not None else None
            self.model_name = model[1]

        self.times = times  # NOTE: This remark the repetition time instead of total # of time.
        self.total_train_samples = 0
        self.users: List[User] = []
        self.selected_users: List[User] = []
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []

        self.logger = self.getLogger()

    def preload_data(self, print_stat=False):
        """Load dataset which will be used to initialize servers later."""
        print(f"Pre-loading data: {self.dataset}")
        if self.initiated_by_hydra_config():
            dataset_config = copy.deepcopy(self.full_config.dataset)
            if self.full_config.action == 'eval' and hasattr(self.full_config, "eval_config") and hasattr(self.full_config.eval_config, "dataset"):
                from fade.utils import flatten_dict
                print("Update dataset config for evaluation:")
                for k, v in flatten_dict(OmegaConf.to_container(self.full_config.eval_config.dataset, resolve=True)).items():
                    print(f"  {k}={OmegaConf.select(dataset_config, k)} => {v}")
                    OmegaConf.update(dataset_config, k, v, merge=False)
            self.preloaded_data = read_fed_dataset(dataset_config)
        else:
            from fade.data.utils import read_data
            self.preloaded_data = read_data(self.dataset)
        if isinstance(self.preloaded_data, dict):
            self.groups = self.preloaded_data["train"]["hierarchies"]
            if print_stat: print("//// STAT: Train data ////")
            self.tr_num_samples, self.total_num_users = basic_stat_fed_dict(
                self.preloaded_data["train"], verbose=print_stat)
            if print_stat: print("//// STAT: Test data ////")
            self.te_num_samples, _ = basic_stat_fed_dict(self.preloaded_data["test"],
                                                         verbose=print_stat)
        else:
            clients, groups, train_data, test_data = self.preloaded_data
            self.groups = groups
            if print_stat: print("//// STAT: Train data ////")
            self.tr_num_samples, self.total_num_users = basic_stat(train_data, verbose=print_stat)
            if print_stat: print("//// STAT: Test data ////")
            self.te_num_samples, _ = basic_stat(test_data, verbose=print_stat)

    def create_server(self, times, device="cpu"):
        """Create the functional sever (instance) which should sub-class this class.
        Returns:
            server (ServerAgent)
        """
        if self.initiated_by_hydra_config():
            model = get_model(dataset=self.full_config.dataset.name,
                              **self.full_config.model)
            model.to(device)

            ServerClass = import_fed_server(self.algorithm)  # type: Type[Server]
            server = ServerClass(args=self.full_config,
                                 model=(model, self.model_name),
                                 dataset=(self.dataset, self.preloaded_data),
                                 times=times)
        else:
            raise NotImplementedError(f"Only support hydra config.")
        return server

    def initiated_by_hydra_config(self):
        return hasattr(self, "full_config") and self.full_config is not None and \
               isinstance(self.full_config, DictConfig)

    def get_hash_name(self, personal=False, time=None, include_rep=True):
        """Get the unique name of the server.
        Args:
            personal: If true, name is for personal models.
            time (int|str): If is None, then the object variable will be used.
                The `time` could be str('avg') if not int.
            include_rep: True to include repetition time. If true, 'p_{rep_time}' or 'g_{rep_time}'
                will be added as the last directory. Use this for non-repeat/personal settings.
        """
        if self.initiated_by_hydra_config():
            # use hash code as name
            hash_hex = hash_config(self.full_config, select_keys=self.full_config.hash_keys)
            path = os.path.join(self.dataset, self.full_config.server.name, self.model_name,
                                hash_hex)
            if include_rep:
                run_hash = "p" if personal else "g"
                run_hash += f"_{self.times}" if time is None else f"_{time}"
                path = os.path.join(path, run_hash)
            return path
        else:
            raise NotImplementedError(f"Only support hydra config.")

    if_personal_local_adaptation = False
    """Return True if the algorithm has a personalization (or local training) to adap process
    after gaining a global model. The parameter K and personal_learning_rate are used for the
    adaptation.

    NOTE: Every subclass with personalization has to override this to return True.
    """

    def print_config(self):
        print(OmegaConf.to_yaml(OmegaConf.masked_copy(self.full_config, self.full_config.hash_keys),
                                resolve=True, sort_keys=True))

    def check_files(self, keys=None, verbose=False, personal=False, times=None):
        if keys is None:
            keys = ['config', 'model', 'model_user', 'results', 'user_eval', 'full_user_eval']
        if times is None:
            times = list(range(self.times))
        for key in keys:
            if key in ["config"]:
                path = self.get_path(key, create_new=False)
                exist = os.path.exists(path)
                if verbose:
                    print(f" [{'OK' if exist else '  '}] {key:20s}     @ {path}")
            else:
                for time in times:
                    path = self.get_path(key, personal=personal, time=time, create_new=False)
                    exist = os.path.exists(path)
                    if verbose:
                        print(
                            f" [{'OK' if exist else '  '}] {key:20s} {'p' if personal else 'g'}_{time}"
                            f" @ {path}")

    def get_path(self, key, hash_name=None, personal=False, time=None, snap_shot_idx=-1,
                 **file_kwargs):
        if hash_name is None:
            if key in ["config"]:
                hash_name = self.get_hash_name(personal, include_rep=False)
            else:
                hash_name = self.get_hash_name(personal,
                                               time=time)  # TODO also save personal model.
        else:
            assert isinstance(hash_name, str) and hash_name != "null"
            if hash_name[-1] == "/":
                last_fld = os.path.split(hash_name)[1]
            else:
                last_fld = os.path.split(hash_name)[1]
            if last_fld.startswith("g_") or last_fld.startswith("p_"):
                pass
            else:
                if time is None:
                    time = self.times
                hash_name += f"/g_{time}" if not personal else f"/p_{time}"
        if key == "model":  # server model
            path = FileManager.out(os.path.join("models", hash_name), is_dir=True, **file_kwargs)
            path = os.path.join(path, "server.pt")
        elif key == "model_user":
            path = FileManager.out(os.path.join("models", hash_name, "users"), is_dir=True,
                                   **file_kwargs)
            # TODO return each user's file name.
        elif key == 'results':
            # TODO add config and other results files....
            path = FileManager.out(os.path.join("results", hash_name + '.pk'), **file_kwargs)
        elif key in ("user_eval", "full_user_eval"):
            if snap_shot_idx >= 0:
                key += f"_{snap_shot_idx:05d}"
                path = FileManager.out(os.path.join("results", hash_name, 'snapshot', f'{key}.pk'),
                                       **file_kwargs)
            else:
                path = FileManager.out(os.path.join("results", hash_name, f'{key}.pk'),
                                       **file_kwargs)
        elif key == "config":
            path = FileManager.out(os.path.join("results", hash_name, 'config.yaml'), **file_kwargs)
        elif key == "fig":
            if snap_shot_idx >= 0:
                path = FileManager.out(os.path.join("results", hash_name, 'snapshot', 'fig'),
                                       **file_kwargs)
            else:
                path = FileManager.out(os.path.join("results", hash_name, 'fig'), **file_kwargs)
        else:
            raise ValueError(f"Wrong key: {key}")
        return path

    # Access models
    def save_model(self):
        model_path = self.get_path('model', create_new=True)
        print(f"Saving server model => {model_path}")
        torch.save(self.model.state_dict(), model_path)

        if len(self.users) >= 50 or (hasattr(self.full_config, "disable_save_user_model")
                                     and self.full_config.disable_save_user_model):
            print(f"Not save user models because #user ({len(self.users)}) > 50.")
        else:
            user_path = self.get_path('model_user', create_new=True)
            print(f"Saving user models => {user_path}")
            for user in tqdm(self.users, file=sys.stdout):
                user.save_model(user_path)

    def load_model(self, hash_name=None, to_load=('server', 'user')):
        # NOTE: for hash_name starts with `/`, the hash_name is treated as absolute path.
        if hash_name is not None:
            print(f"Loading by hash name: {hash_name}")
        if 'server' in to_load:
            model_path = self.get_path('model', hash_name=hash_name, create_new=False)
            print(f"Loading SERVER model <= {model_path}")
            assert (os.path.exists(model_path)), f"Server model NOT exist: {model_path}"
            loaded = torch.load(model_path)
            self.model.load_state_dict(loaded)
            # print(f"Loaded server model with structure: {self.model}")

        if 'user' in to_load:
            # NOTE: when training, send_params may override 'partial' params. But the local part
            #   will not change.
            user_path = self.get_path('model_user', hash_name=hash_name, create_new=False)
            print(f"Loading USER models <= {user_path}")
            for user in tqdm(self.users):
                user.load_model(user_path)

    # Save loss, accurancy to h5 file
    def save_results(self):
        """TODO remove?"""
        # store global performance value
        self._write_result(self.get_path("results"),
                           self.rs_glob_acc, self.rs_train_acc, self.rs_train_loss)
        # save config
        self._write_result_(self.get_path("config"),
                            OmegaConf.to_yaml(self.full_config, resolve=True, sort_keys=True),
                            _format="yaml")

    def save_figure(self, fig_name, snap_shot_idx=-1):
        import matplotlib.pyplot as plt
        if snap_shot_idx >= 0:
            fig_name += f"_{snap_shot_idx}"
        file_name = os.path.join(self.get_path("fig", personal=False, is_dir=True,
                                               snap_shot_idx=snap_shot_idx
                                               ), fig_name + ".pdf")
        plt.savefig(file_name, bbox_inches="tight")
        print(f"save figure => {file_name}")

    def dump_to_file(self, personal, key, obj, snap_shot_idx=-1):
        _fname = self.get_path(key, personal=personal, snap_shot_idx=snap_shot_idx)
        with open(_fname, "wb") as f:
            pk.dump(obj, f)
            print(f" {key} ==> {_fname}")

    def load_from_file(self, personal, key, return_date=False, time=None):
        _fname = self.get_path(key, personal=personal, time=time, return_date=return_date)
        if return_date:
            _fname, mod_time = _fname
        with open(_fname, "rb") as f:
            obj = pk.load(f)
            print(f" {key} <== {_fname}")
            if return_date:
                return obj, mod_time
            else:
                return obj

    def _write_result(self, fname, rs_glob_acc, rs_train_acc, rs_train_loss):
        if len(rs_glob_acc) != 0 & len(rs_train_acc) & len(rs_train_loss):
            self._write_result_(fname,
                                data_dict=dict(rs_glob_acc=np.array(rs_glob_acc),
                                               rs_train_acc=np.array(rs_train_acc),
                                               rs_train_loss=np.array(rs_train_loss),
                                               ),
                                _format="pk")

    @staticmethod
    def _write_result_(fname, data_dict, _format="pk", alert_on_empty=False):
        """Generic func for writing dict data.
        Args:
            data_dict (str | dict)
        """
        assert fname.endswith("." + _format), f"Format ({_format}) is inconsistent with the " \
                                              f"file name: {fname}"
        # _fname = FileManager.out(os.path.join("results", f'{fname}.' + _format))
        if alert_on_empty and any([v is None or 0 == len(v) for _, v in data_dict.items()]):
            print(f"Found empty data_dict:", end="")
            for k, arr in data_dict.items():
                print(f"{k} ({None if arr is None else len(arr)})")
            print()
        print(f"Write result => {fname}")
        if _format == "pk":
            with open(fname, "wb") as f:
                pk.dump(data_dict, f)
        elif _format == "yaml":
            if isinstance(data_dict, str):
                with open(fname, "w") as f:
                    f.write(data_dict)
            else:
                from yaml import safe_dump
                with open(fname, "w") as f:
                    safe_dump(data_dict, f)
        else:
            raise ValueError(f"Unknown format: {_format}")
        print('done')

    def load_all_rep_final(self, total_times, personal=False, reduce_user=True, reduce_rep=True,
                           user_weights=None):
        """Load all repeated results of the final evaluation."""
        assert user_weights is not None
        res = {}
        for i in range(total_times):
            _res = self.load_results(personal, time=i, name="final_eval")
            # keys: ["test_acc", "train_acc", "train_loss"] each entry is a list of users' final metric values.
            for k, v in _res.items():
                if k == "extra_info":
                    continue
                if reduce_user:
                    v = np.average(v, weights=user_weights)
                res.setdefault(k, []).append(v)
        if reduce_rep:
            res = dict((k, np.mean(v, axis=0)) for k, v in res.items())
        return res

    def load_all_repetitions(self, total_times, num_glob_iters=None, personal=False,
                             reduce_user=True):
        """Load all repeated experimental results.
        Args:
            reduce_user: Reduce users by mean if user-wise results are available.

        Returns:
            glob_acc, train_acc, train_loss. Of shape [n_repetition, num_glob_iters] if
                reduce_user=True, else [n_repetition, num_glob_iters, n_users].
        """
        num_glob_iters = num_glob_iters if num_glob_iters is not None else self.num_glob_iters
        shape = [total_times, num_glob_iters]
        if not reduce_user:
            shape += [self.total_num_users]
        train_acc = np.zeros(shape)
        train_loss = np.zeros(shape)
        glob_acc = np.zeros(shape)
        for i in range(total_times):
            res = np.array(self.load_results(personal, time=i))
            if len(res.shape) <= 2:  # shape: [3, n_iters]
                assert len(train_acc.shape) == len(res.shape), \
                    f"res is expected to have {len(train_acc.shape)} dimensions but got" \
                    f" {len(res.shape)} dimensions. Perhaps, user dimension is not saved. " \
                    f"Try to re-run the training which may save user-wise records. If still" \
                    f" failed, try to check the codes of evaluation."
                n_iters = min((res.shape[1], num_glob_iters))
                train_acc[i, :n_iters], train_loss[i, :n_iters], glob_acc[i, :n_iters] = res[:,
                                                                                         :n_iters]
            else:  # shape: [3, n_iters, n_users]
                n_iters = min((res.shape[1], num_glob_iters))
                assert res.shape[2] == len(self.tr_num_samples), \
                    f"Total number of users is {len(self.tr_num_samples)}, but each epoch" \
                    f" only compress {res.shape[2]} users. To do average, please use full" \
                    f" evaluation instead of partial evaluation."
                if reduce_user:
                    train_acc[i, :n_iters] = np.average(res[0, :n_iters, :], axis=1,
                                                        weights=self.tr_num_samples)
                    train_loss[i, :n_iters] = np.average(res[1, :n_iters, :], axis=1,
                                                         weights=self.tr_num_samples)
                    glob_acc[i, :n_iters] = np.average(res[2, :n_iters, :], axis=1,
                                                       weights=self.te_num_samples)
                else:
                    train_acc[i, :n_iters, :], train_loss[i, :n_iters, :], glob_acc[i, :n_iters,
                                                                           :] = res[:, :n_iters, :]

        return glob_acc, train_acc, train_loss

    def load_final_eval(self, personal=False, time=None, full_eval=True):
        """Load the full-info evaluation results which requires to run with `main.py --action eval`.

        time (int): The index of repetition time starting from zero.
        """
        key = "user_eval"
        if full_eval:
            key = "full_" + key
        try:
            rs_dict, mod_time = self.load_from_file(personal=personal, key=key,
                                                    return_date=True, time=time)
            print(f" Evaluation time: {mod_time}")
        except FileNotFoundError as e:
            print(f"File not found. Try to run `main.py` with `action=eval` first.")
            raise e
        return rs_dict

    def average_results_and_save(self, total_times, num_glob_iters=None, return_personal=False):
        """Load previous repeated experimental results and compute average.
        Returns:

        """
        result_stat = {}
        ServerClass = import_fed_server(self.algorithm)
        for model_type in ["personal", "global"]:
            personal = model_type == "personal"
            if personal and not ServerClass.if_personal_local_adaptation:
                # no personal model.
                continue
            print(f"Average {model_type} model results...")
            try:
                # Load previously saved repetition results (shape: [n_repetition, n_glob_iters]).
                test_acc, train_acc, train_loss = \
                    self.load_all_repetitions(total_times, num_glob_iters, personal=personal)
            except FileNotFoundError as e:
                print(f" Results not exists: {e} skip...")
                continue
            glob_acc_data = np.average(test_acc, axis=0)
            train_acc_data = np.average(train_acc, axis=0)
            train_loss_data = np.average(train_loss, axis=0)
            max_accurancy = []  # max acc across iterations
            for i in range(total_times):
                max_accurancy.append(test_acc[i].max())

            res = {
                # The name is standard for ray.tune logging.
                "std_accuracy": np.std(max_accurancy),
                "mean_accuracy": np.mean(max_accurancy),
            }

            print("Stat of acc (repeats)")
            print("  Std:", res['std_accuracy'])
            print("  Mean:", res['mean_accuracy'])

            if personal:
                result_stat["personal"] = res
            else:
                result_stat["global"] = res

            alg = self.get_hash_name(personal, time="avg")
            self._write_result(alg, glob_acc_data, train_acc_data, train_loss_data)
        if return_personal:
            assert "personal" in result_stat, "Require personal model results, but personal " \
                                              "models are not available."
            return result_stat["personal"]
        else:
            return result_stat["global"]

    def load_results(self, personal=False, averaged=False, name="iter", time=None):
        """Load result at specific time.

        Args:
            averaged: If true, the averaged iterations will be loaded. This will be ignored
                unless name="iterations".
            name ('iter'|'final_eval'|'final_full_eval'): The result name.
                iter - This will also load the iteration records into object variables.
                final_eval - Final evaluation. Saved in main.py
                full_final_eval - Final evaluation. Saved in main.py by using `--action eval` arg.
            time (int): The repetition time (index).

        Returns:
            Depends on the stored file.
        """
        if averaged:
            hash_name = self.get_hash_name(personal, time="avg")
        else:
            hash_name = self.get_hash_name(personal, time=time)
        if name == "iter":
            _fname = FileManager.out(os.path.join("results", f'{hash_name}.pk'), create_new=False)
            print(f" Load result <= {_fname}")
            with open(_fname, "rb") as f:
                data_dict = pk.load(f)
                self.rs_glob_acc = data_dict['rs_glob_acc']
                self.rs_train_acc = data_dict['rs_train_acc']
                self.rs_train_loss = data_dict['rs_train_loss']
            return self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc
        elif name == "final_eval":
            return self.load_final_eval(personal=personal, time=time, full_eval=False)
        elif name == "final_full_eval":
            return self.load_final_eval(personal=personal, time=time, full_eval=True)
        else:
            raise ValueError(f"wrong name: {name}")

    def save_test_result(self):
        # FIXME This funciton is not used.
        # store global performance value
        alg = self.get_hash_name(False)
        self._write_result(alg, self.rs_glob_acc, self.rs_train_acc, self.rs_train_loss)

    def compute_group_metric(self):
        pass

    def log(self, info_dict: dict, is_summary=False, **kwargs):
        if hasattr(self, "full_config"):
            _info_dict = {}
            user_weights = [u.train_samples * 1. / self.total_train_samples for u in self.users]
            for k, v in info_dict.items():
                if 'extra_info' in k:
                    extra_info = v
                    _info_dict.update(self.process_and_log_extra_info(extra_info, reduction='mean'))
                    continue
                elif 'ids' in k:
                    pass
                elif np.isscalar(v):
                    _info_dict[k] = v
                    assert not np.isnan(v), f"{k}: {v}"
                else:
                    v = np.array(v)
                    assert not np.any(np.isnan(v)), f"{k}: {v} with nan"
                    assert len(
                        v.shape) == 1, f"Only support vector. But get key[{k}] with value shape {v.shape}"
                    _info_dict[k] = np.average(v, weights=user_weights)
                    # ALERT: The histogram may fail if v contains nan!
                    # _info_dict[k+"_hist"] = wandb.Histogram(v)

                    group_values = {}
                    group_weights = {}

                    if self.full_config.logger.log_user:
                        assert not self.partial_eval  # need to align user index.
                        for user, user_v, user_w in zip(self.users, v, user_weights):
                            _info_dict[user.id + "_" + k] = user_v

                            group_values.setdefault(user.group_name + "_" + k, [])
                            group_values[user.group_name + "_" + k] += [user_v]
                            group_weights.setdefault(user.group_name + "_" + k, [])
                            group_weights[user.group_name + "_" + k] += [user_w]

                    for group_k in list(group_values.keys()):
                        weights = group_weights[group_k]
                        sum_w = sum(weights)
                        weights = [w * 1. / sum_w for w in weights]
                        _info_dict[group_k] = np.average(group_values[group_k], weights=weights)
                        print(f"## log group {group_k}: {_info_dict[group_k]}")

            info_dict = _info_dict
            for k, v in info_dict.items():
                if np.any(np.isnan(v)):
                    raise ValueError(f"Try to log nan. Key: {k}: {v}")

            if "wandb" in self.full_config.logger.loggers:
                if is_summary:
                    for k, v in info_dict.items():
                        wandb.run.summary[k] = v
                else:
                    try:
                        wandb.log(info_dict, **kwargs)
                    except ValueError as e:
                        print("Error raises when logging ", info_dict)
                        raise e

    def process_and_log_extra_info(self, extra_info, reduction='none'):
        _info_dict = {}
        for subset in extra_info:
            group_res_dict = {}
            # gather results by groups
            for user in self.users:  # extra_info[subset]:
                group_res_dict.setdefault(user.group_name, {})
                for key in extra_info[subset][user.id]:
                    print(f"### extra_info[{subset}][{user.id}][{key}]: {extra_info[subset][user.id][key]}")
                    group_res_dict[user.group_name].setdefault(key, 0)
                    group_res_dict[user.group_name][key] += extra_info[subset][user.id][
                        key]
            for g in group_res_dict:
                if "FN" in group_res_dict[g]:
                    group_res_dict[g]["FNR"] = group_res_dict[g]["FN"] * 1. / \
                                               (group_res_dict[g]["Pos"] - 1e-8)
                if "FP" in group_res_dict[g]:
                    group_res_dict[g]['FPR'] = group_res_dict[g]["FP"] * 1. / \
                                               (group_res_dict[g]["Neg"] - 1e-8)
                for key in group_res_dict[g]:
                    _info_dict[subset + "_" + g + "_" + key] = group_res_dict[g][key]
                    # print(f"## log group {subset + '_' + g + '_' + key}: {group_res_dict[g][key]}")
            if len(group_res_dict) == 2:  # two group case
                groups = list(group_res_dict.keys())
                if 'FPR' in group_res_dict[groups[0]] and 'FNR' in group_res_dict[
                    groups[0]] \
                        and 'FPR' in group_res_dict[groups[1]] and 'FNR' in \
                        group_res_dict[groups[1]]:
                    _info_dict[subset + '_deltaEO'] = np.abs(
                        group_res_dict[groups[0]]['FPR'] - group_res_dict[groups[1]][
                            'FPR']) + np.abs(
                        group_res_dict[groups[0]]['FNR'] - group_res_dict[groups[1]][
                            'FNR'])
                    # print(
                    #     f"## log group {subset + '_deltaEO'}: {_info_dict[subset + '_deltaEO']}")

                if 'FP' in group_res_dict[groups[0]] and 'FN' in group_res_dict[
                    groups[0]] \
                        and 'FP' in group_res_dict[groups[1]] and 'FN' in \
                        group_res_dict[groups[1]]:
                    Pos = sum([group_res_dict[g]["Pos"] for g in group_res_dict])
                    FN = sum([group_res_dict[g]["FN"] for g in group_res_dict])
                    FP = sum([group_res_dict[g]["FP"] for g in group_res_dict])
                    PredPos = sum(
                        [group_res_dict[g]["PredPos"] for g in group_res_dict])
                    _info_dict[subset + '_F1'] = (Pos - FN) / (Pos - FN / 2. + FP / 2. - 1e-8)
                    _info_dict[subset + '_prec'] = (Pos - FN) * 1. / (PredPos - 1e-8) # if PredPos > 0 else -1
                    _info_dict[subset + '_sens'] = (Pos - FN) * 1. / (Pos - 1e-8)
                    # print(f"## log group {subset + '_F1'}: {_info_dict[subset + '_F1']}")

        for k in _info_dict:
            if reduction == 'mean':
                _info_dict[k] = np.mean(_info_dict[k])
            elif reduction == 'none':
                pass
            else:
                raise ValueError(f"reduction: {reduction}")
            print(f"## log group {k:>25s}: {_info_dict[k]:.3f}")
        return _info_dict

    def getLogger(self, logger_name=None):
        if logger_name is None:
            logger_name = self.__class__.__name__
        log_fname = FileManager.data(os.path.join(self.get_hash_name(), "server.log"), is_dir=False,
                                     overwrite=True)
        print(f"Server log to file: {log_fname}")
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(filename=log_fname, mode='w')
        fh.setFormatter(logging.Formatter('[%(asctime)s - %(levelname)s -'
                                          ' %(filename)s:%(funcName)s] %(message)s'))
        logger.addHandler(fh)
        logger.propagate = False  # Set False to disable stdout print.
        return logger


class Server(ServerAgent):
    def __init__(self, *args, device="cuda", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.create_users()

    def create_users(self):
        # Initialize data for all  users
        for i in tqdm(range(self.total_num_users), desc='init user', file=sys.stdout):
            id, group, train, test = extract_user_data(i, self.preloaded_data)
            user = self.init_user(id, group, train, test)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Number of users each epoch / total users:", self.num_users, " / ",
              self.total_num_users)
        print(f"Finished creating {self.__class__.__name__} server.")

    def init_user(self, id, group, train, test):
        self.logger.debug(f"[init user {id}] START")
        if self.initiated_by_hydra_config():
            if self.user_cfg.name == "group_adv":
                self.logger.debug(f"  self.full_config.server.group_label_mode "
                                  f"{self.full_config.server.group_label_mode}, group: {group}")
                if hasattr(self.full_config.server, 'group_label_mode'):
                    if group[1] in self.full_config.server.group_label_mode:
                        self.logger.debug(
                            f"  {group[1]} with label mode: {self.full_config.server.group_label_mode[group[1]]}")
                        label_mode = self.full_config.server.group_label_mode[group[1]]
                    else:
                        raise ValueError(
                            f"Not found group {group[1]} in:  {self.full_config.server.group_label_mode}")
                else:
                    label_mode = "supervised"

                from fade.user.group_adv import GroupAdvUser
                user = GroupAdvUser(id, train, test, (self.model, self.model_name), group=group,
                                    label_mode=label_mode, num_glob_iters=self.num_glob_iters,
                                    **self.user_cfg)
                self.logger.debug(f"  [user {user.id}] Training batch size: {user.batch_size}, 1"
                                  f" epoch should include {len(user.trainloader)} batches.")
            elif self.user_cfg.name == "generic":
                if hasattr(self.full_config.server, 'group_label_mode'):
                    raise ValueError("Generic User does not support group_label_mode but found.")
                from fade.user.generic import GenericUser
                user = GenericUser(id, train, test, (self.model, self.model_name), group=group,
                                   num_glob_iters=self.num_glob_iters, **self.user_cfg)
            else:
                raise ValueError(f"self.user_cfg.name: {self.user_cfg.name}")
        else:
            raise RuntimeError(f"Only support config by hydra.")
        self.logger.debug(f"[init user {id}] DONE")
        return user

    def send_parameters(self, partial=None, glob_iter=None):
        if partial is None:
            if self.full_config.server.share_mode == "partial":
                partial = True
            elif self.full_config.server.share_mode == "mix_all_partial":
                assert glob_iter is not None, "glob_iter is required when mix share mode is chosen."
                partial = glob_iter > self.full_config.server.share_since_n_iter or glob_iter
            else:
                partial = False
                assert self.full_config.server.share_mode != "private"
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            if partial:  # share only subset of parameters
                user.set_shared_parameters(self.model)
            else:  # share all parameters
                user.set_parameters(self.model)

    def add_parameters(self, user, ratio, dst_params=None, only_shared_params=False):
        if only_shared_params:
            assert dst_params is None, "dst_params should be None when only add shared params."
            iterator_tuple = (self.model.get_shared_parameters(detach=False),
                              user.model.get_shared_parameters(detach=False))
        else:
            if dst_params is None:
                iterator_tuple = (self.model.parameters(), user.get_parameters())
            else:
                iterator_tuple = (dst_params, user.get_parameters())
        for server_param, user_param in zip(*iterator_tuple):
            server_param.data.add_(user_param, alpha=ratio)
            # server_param.data = server_param.data + user_param.data.clone() * ratio

    def personalized_aggregate_parameters(self, partial=False, clip_norm=None, dp_sigma=None,
                                          weights=None):
        """TODO: Rename to aggregate_parameters"""
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        for user in self.selected_users:
            if user.has_sharable_model():
                total_train += user.train_samples

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data.zero_()
        if weights is None:
            weights = [1.] * len(self.selected_users)
        for w, user in zip(weights, self.selected_users):
            if user.has_sharable_model():
                self.add_parameters(user, w * user.train_samples / total_train,
                                    only_shared_params=partial)

        if not np.isclose(self.beta, 1.):
            # aggregate average model with previous model using parameter beta
            for pre_param, param in zip(previous_param, self.model.parameters()):
                # large memory cost
                param.data.mul_(self.beta).add_(pre_param, alpha=1 - self.beta)

    @property
    def online_user_idxs(self):
        """The list of index of users who are able to join in for training."""
        idxs = []
        if self.full_config.server.share_mode == "private":
            return []
        for idx, user in enumerate(self.users):
            if user.can_join_for_train():
                idxs.append(idx)
        return idxs

    def select_users(self, round, num_users, only_online_users=True, probs=None):
        """selects num_clients clients weighted by number of samples from possible_clients
        Args:
            round: The server iteration number.
            num_users: number of clients to select; default 20
                NOTE: that within function, num_clients is set to
                    min(num_clients, len(possible_clients))
            only_online_users: True if select only users.
            probs: An array of the size as `num_users` and the sum should be 1..

        Return:
            list of selected clients objects
        """
        if (num_users == len(self.users)):
            print("All users are selected")
            return self.users

        selection_mode = self.full_config.server.user_selection

        if isinstance(selection_mode, (list, tuple, ListConfig)):
            assert only_online_users
            assert np.isclose(sum(selection_mode),
                              1.), f"selection_mode ({selection_mode}) does not" \
                                   f"sum to 1 but is {sum(selection_mode)}."
            idxs = np.random.choice(len(self.online_user_idxs), num_users,
                                    p=selection_mode, replace=False)
            selected_users = [self.users[i] for i in idxs]
        elif selection_mode == "exchange":
            assert only_online_users
            assert len(self.online_user_idxs) == len(self.users)
            assert num_users == 1
            idx = int((1. * round / self.num_glob_iters) * len(self.users)) + 1
            idx = idx % len(self.users)
            selected_users = [self.users[idx]]
        elif selection_mode == "sequential":  # user_selection
            # sequentially select online users.
            assert only_online_users, "Sequential user selection is only supported when " \
                                      "only_online_users is True."
            assert len(self.online_user_idxs) >= num_users, \
                f"Only {len(self.online_user_idxs)} users are online, but want to" \
                f" select {num_users} users."
            online_user_idxs = self.online_user_idxs
            # assert len(self.online_user_idxs) == 2
            n_online = len(online_user_idxs)
            start_idx = round * num_users
            end_idx = (round + 1) * num_users
            selected_users = [self.users[online_user_idxs[i % n_online]]
                              for i in range(start_idx, end_idx)]
        # elif self.full_config.server.user_selection == "group_seq":
        #   # TODO implement group selection: Alternatively select groups sequentially.
        elif selection_mode in ("uniform_random", "random_uniform"):
            if only_online_users:
                if len(self.online_user_idxs) >= num_users:
                    if probs is not None:
                        probs = [probs[i] for i in self.online_user_idxs]
                        total_prob = sum(probs)
                        probs = [p / total_prob for p in probs]
                    selected_users = np.random.choice([self.users[i] for i in self.online_user_idxs],
                                            num_users, replace=False,
                                            p=probs)
                else:
                    selected_users = [self.users[i] for i in self.online_user_idxs]
                    offline_users = [self.users[i] for i in range(len(self.users)) if
                                     i not in self.online_user_idxs]

                    if len(offline_users) < 1:
                        raise ValueError(f"No offline users are available. Total #user: "
                                         f"{len(self.users)}, #online: {len(self.online_user_idxs)}:"
                                         f" {self.online_user_idxs} while we want to select "
                                         f"{num_users} users.")

                    offline_probs = [probs[i] for i in range(len(self.users)) if
                                     i not in self.online_user_idxs] if probs is not None else None
                    if offline_probs is not None:
                        total_prob = sum(offline_probs)
                        offline_probs = [p / total_prob for p in offline_probs]

                    # num_users = min(num_users, len(self.online_user_idxs))
                    # np.random.seed(round)
                    selected_users += np.random.choice(offline_users,
                                                       num_users - len(self.online_user_idxs),
                                                       replace=False,
                                                       p=offline_probs).tolist()  # , p=pk)
            else:
                selected_users = np.random.choice(self.users, num_users, replace=False, p=probs)
        else:
            raise NotImplementedError(f"Invalid `user_selection` modes ({type(selection_mode)}): "
                                      f"{self.full_config.server.user_selection}")

        print(f"Select user ({len(selected_users)}/{len(self.online_user_idxs)}): "
              f"{[u.id for u in selected_users]}")
        return selected_users

    # define function for personalized agegatation.
    def personalized_update_parameters(self, user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def test(self, users: List[User], full_info=False, personal=False, model=None):
        num_samples = []
        tot_correct = []
        extra_info_by_id = {}
        verbose = 1 if len(users) < 100 else 0
        for c in (tqdm(users, desc="eval te user", file=sys.stdout) if verbose == 0 else users):
            ct, ns, extra_info = c.test(full_info=full_info, personal=personal,
                                        model=model, verbose=verbose)
            extra_info_by_id[c.id] = extra_info
            if verbose > 0:
                print(f"### User {c.id} test acc: {ct}/{ns}={ct * 1.0 / ns}")
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in users]

        return ids, num_samples, tot_correct, extra_info_by_id

    def evaluate(self, users: List[User] = None, reduce_users=True, full_info=False,
                 add_res_to_record=True, return_dict=False, personal=False, model=None):
        """Evaluate the test acc, train acc & loss. Save results in the instance variables.

        Args:
            users: The list of users to evaluate. If is None, use all users in the server.
                NOTE: When `users` is provided, the recorded acc/loss will be
                averaged instead of user-wise.
            reduce_users: If true, reduce results w.r.t. users and print.
            full_info: If true, return all useful information.
            add_res_to_record: If true, add major results to the instance variables.
            return_dict: If true, return in a single dict instead of a tuple.
            personal: If true, use personalized models.

        Returns:
            test_accs, train_accs, train_losses: List of user acc/loss.
            dict with kyes: "test_acc", "train_acc", "train_loss". Values are list of user metric
                values or scalars of averaged metric values.
        """
        tag = 'personal' if personal else 'global'
        print(f"Evaluate {tag} model")
        if users is None:
            users = self.users
            # record_average = False
        else:
            # When selected users are used here, we will enforce to reduce user
            #   Otherwise, the num of users will be undetermined.
            # add_res_to_record = True
            # reduce_users = True
            if users != self.users:
                assert reduce_users, "When user set is the the whole set, the number of users may vary. " \
                                     "Thus, the results has to reduce users."
        extra_info = {}
        ids, test_num_samples, test_tot_correct, extra_res_info = self.test(
            users, full_info=full_info, personal=personal, model=model)
        extra_info["test"] = extra_res_info

        ids, num_samples, tot_correct, losses, extra_res_info = \
            self.train_error_and_loss(users, full_info=full_info,
                                      personal=personal, model=model)
        extra_info["train"] = extra_res_info

        # list of results by user
        test_accs = [ncor * 1. / ns for (ns, ncor) in zip(test_num_samples, test_tot_correct)]
        train_accs = [ncor * 1. / ns for (ns, ncor) in zip(num_samples, tot_correct)]
        train_losses = losses

        # Reduce users by average.
        avg_test_accs = np.sum(test_tot_correct) * 1.0 / np.sum(test_num_samples)
        avg_train_accs = np.sum(tot_correct) * 1.0 / np.sum(num_samples)
        avg_train_losses = np.average(train_losses, weights=num_samples)

        # print("stats_train[1]",stats_train[3][0])
        print(f" Average {tag} test  acc: ", avg_test_accs)
        print(f" Average {tag} train acc: ", avg_train_accs)
        print(f" Average {tag} train loss: ", avg_train_losses)
        if reduce_users:
            test_accs = avg_test_accs
            train_accs = avg_train_accs
            train_losses = avg_train_losses

        if add_res_to_record:
            if not reduce_users:
                if self.partial_eval:
                    assert len(
                        test_accs.shape) == self.num_users, f"# test users are not consistent: {len(test_accs)} vs {self.num_users}"
                else:
                    assert len(test_accs) == len(
                        self.tr_num_samples), f"# test users are not consistent: {len(test_accs)} vs {len(self.tr_num_samples)}"
            self.rs_glob_acc.append(test_accs)
            self.rs_train_acc.append(train_accs)
            self.rs_train_loss.append(train_losses)

        ret_keys = ["test_acc", "train_acc", "train_loss"]
        ret_dict = dict(zip(ret_keys, [test_accs, train_accs, train_losses]))
        # if full_info:  # add extra info
        ret_dict.update({"extra_info": extra_info, "ids": ids})
        ret_keys.extend(["extra_info", "ids"])

        if return_dict:
            return ret_dict
        else:
            # Note the return has to be in order of ret_keys.
            return [ret_dict[k] for k in ret_keys]

    def train_error_and_loss(self, users, full_info=False, personal=False, model=None):
        num_samples = []
        tot_correct = []
        losses = []
        extra_info_by_id = {}
        verbose = 1 if len(users) < 100 else 0
        for c in (tqdm(users, desc="eval tr user") if verbose == 0 else users):
            result = c.train_error_and_loss(full_info=full_info, personal=personal,
                                            model=model, verbose=verbose)
            correct_cnt, sum_loss, n_sample = result[:3]
            if verbose > 0:
                print(f"### User {c.id} train acc: "
                      f"{correct_cnt}/{n_sample}={correct_cnt * 1.0 / n_sample}")
            extra_info = result[3]
            extra_info_by_id[c.id] = extra_info
            tot_correct.append(correct_cnt * 1.0)
            losses.append(sum_loss * 1.0 / n_sample)
            num_samples.append(n_sample)

        ids = [c.id for c in users]
        # groups = [c.group for c in self.clients]
        return ids, num_samples, tot_correct, losses, extra_info_by_id

    def train_users(self):
        """Train users and aggregate parameters."""
        for user in self.selected_users:
            user.train()
        self.personalized_aggregate_parameters()
