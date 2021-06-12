"""Fed set including multiple domains.

Examples:
    # Mnist + MnistM
    python -m fade.data.multi_domain -cn comb/MnistM
    # change n_user of Mnist
    python -m fade.data.multi_domain -cn comb/MnistM --cfg job dataset.meta_datasets.0.n_user=10

    # Office31_A2W
    python -m fade.data.multi_domain -cn comb/Office31_A2W

    ===================
    # MnistM + Mnist
    # Step 1: generate meta sets
    python -m fade.data.extend -d Mnist -c 5 -u 20
    python -m fade.data.extend -d MnistM -c 5 -u 20
    # python -m fade.data.extend -d Svhn -c 5 -u 20
    # Step 1: combine them
    python -m fade.data.multi_domain -d comb/MnistM_c5u40
"""
import os
import torch
import numpy as np
from typing import List
# import argparse
import hydra
from omegaconf import DictConfig, ListConfig
from typing import Iterable

from fade.file import FileManager
from fade.utils import _log_time_usage
from .utils import load_fed_dataset
from . import FedDataset


class MDFedDataset(FedDataset):
    def _generate(self):
        assert isinstance(self.cfg.meta_datasets, Iterable), f"type is {type(self.cfg.meta_datasets)}"
        assert isinstance(self.cfg.meta_datasets[0], DictConfig)

        if hasattr(self.cfg, "meta_fed_ds") and self.cfg.meta_fed_ds != "extend":
            if self.cfg.meta_fed_ds == "federalize":
                from .federalize import FedExtDataset as ExtendedDataset
            else:
                raise RuntimeError(f"meta_fed_ds: {self.cfg.meta_fed_ds}")
        else:
            raise ValueError(f"Not support 'extend' module anymore. Set 'meta_fed_ds' in "
                             f"dataset as 'federalize'.")

        # Auto set n user for the unset one.
        if hasattr(self.cfg, 'total_n_user'):
            n_users = [md.n_user for md in self.cfg.meta_datasets]
            void_idx = np.nonzero(np.array(n_users) == -1)[0]
            num_void = len(void_idx)
            assert num_void <= 1, f"Only allow one meta dataset to set n_user " \
                                  f"as -1, but get {num_void}. All settings " \
                                  f"are {n_users}."
            if num_void > 0:
                void_idx = int(void_idx[0])
                cur_n_user = sum(n_users) + 1  # complement -1
                assert cur_n_user < self.cfg.total_n_user,\
                    f"Already have {cur_n_user} users more than total " \
                    f"{self.cfg.total_n_user} users."
                self.cfg.meta_datasets[void_idx].n_user = self.cfg.total_n_user - cur_n_user
                print(f"Set the {void_idx}-th meta dataset with {self.cfg.meta_datasets[void_idx].n_user} users.")

        # check meta datasets existence or generate new.
        for ds_cfg in self.cfg.meta_datasets:
            ext_ds = ExtendedDataset(ds_cfg)
            for subset in ("train", "test"):
                if not ext_ds.exist(subset):
                    print(f"Not found {subset} for {ds_cfg.name}")
                    ext_ds.generate()
                    break
            else:
                print(f"CHECKED: {ds_cfg.name} has been generated with specific config.")

        return combine(self.cfg.meta_datasets, ExtendedDataset)


def combine(meta_datasets, ExtendedDataset):
    """Read and combine datasets from the list.

    Args:
        meta_datasets (List[str] | ListConfig): List of meta fed datasets.

    Returns:
        A dict compress train/test fed sets.
    """
    comb_ds = {
        "train": {"users": [], "user_data": {}, "hierarchies": [], "num_samples": []},
        "test": {"users": [], "user_data": {}, "hierarchies": [], "num_samples": []},
    }

    def rename_client_id(data_name, old_id):
        return f"{data_name}_{old_id}"

    if isinstance(meta_datasets[0], DictConfig):
        subsets = ("train", "test")
        for cfg in meta_datasets:
            fed_dict = ExtendedDataset(cfg).load(subsets)

            for subset in subsets:
                data_dict = fed_dict[subset]
                assert ("hierarchies" not in data_dict) or len(data_dict["hierarchies"]) == 0, \
                    "Not support: meta dataset include groups."
                new_users = [rename_client_id(cfg.name, c) for c in data_dict["users"]]
                comb_ds[subset]["users"].extend(new_users)
                comb_ds[subset]["num_samples"].extend(data_dict["num_samples"])
                for c in data_dict["users"]:
                    comb_ds[subset]["user_data"][rename_client_id(cfg.name, c)] = \
                    data_dict["user_data"][c]
                # NOTE: not sure how to construct this. just a try.
                comb_ds[subset]["hierarchies"].extend([cfg.name] * len(new_users))
    else:
        for data_name in meta_datasets:
            for subset in ("train", "test"):
                print(f"Reading {data_name}, {subset}...")
                fed_data_dict = load_fed_dataset(data_name, subset=subset)

                assert ("hierarchies" not in fed_data_dict) or len(fed_data_dict["hierarchies"]) == 0, \
                    "Not support: meta dataset include groups."
                new_users = [rename_client_id(data_name, c) for c in fed_data_dict["users"]]
                comb_ds[subset]["users"].extend(new_users)
                comb_ds[subset]["num_samples"].extend(fed_data_dict["num_samples"])
                for c in fed_data_dict["users"]:
                    comb_ds[subset]["user_data"][rename_client_id(data_name, c)] = fed_data_dict["user_data"][c]
                # NOTE: not sure how to construct this. just a try.
                comb_ds[subset]["hierarchies"].extend([data_name] * len(new_users))

    return comb_ds


@hydra.main(config_name="comb/MnistM", config_path="../config/dataset/")
def main(cfg):
    MDFedDataset(cfg.dataset).generate()


if __name__ == '__main__':
    main()
