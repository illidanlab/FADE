from __future__ import annotations
import os
import torch
import numpy as np
import logging
from omegaconf import DictConfig, OmegaConf
from fade.utils import hash_config
from fade.file import FileManager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict


class FedDataset(object):
    """Multi-domain federated dataset."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = None  # Random State for generating

    def get_hash_name(self):
        return "/".join([self.cfg.name, hash_config(self.cfg, exclude_keys="viz")])

    def generate(self):
        print("==" * 20)
        print(f"/// Generating {self.cfg.name} dataset ///")
        if hasattr(self.cfg, "seed"):
            self.rng = np.random.RandomState(self.cfg.seed)
        fed_dict = self._generate()
        self.save(fed_dict)
        print("==" * 20)

    def _generate(self) -> Dict:
        """Generate federated dataset.

        Returns:
            fed_dict: A dict consists of federated data.
        """
        raise NotImplementedError()

    def viz(self):
        """Visualize to explore data."""
        raise NotImplementedError()

    def exist(self, subset):
        """Return true if the `subset`.pt file exists."""
        root_path = FileManager.data(os.path.join(self.get_hash_name(), subset), is_dir=True, overwrite=False, create_new=False)
        file_path = os.path.join(root_path, f"{subset}.pt")
        if not os.path.exists(file_path):
            print(f"Not found: {file_path}")
            return False
        else:
            return True

    def save(self, fed_dict):
        for subset in fed_dict:
            root_path = FileManager.data(os.path.join(self.get_hash_name(), subset), is_dir=True, overwrite=True)
            file_path = os.path.join(root_path, f"{subset}.pt")

            # with open(file_path, 'wb') as outfile:
            print(f"Dumping {subset} data => {file_path}")
            torch.save(fed_dict[subset], file_path)

    def load(self, generate_if_not_exist=False, subsets=["train", "test"]):
        """Load fed dict."""
        # check existence.
        for subset in ("train", "test"):
            ex = self.exist(subset)
            if not ex:
                print(f"NOT found {subset} for {self.cfg.name} dataset.")
                if generate_if_not_exist:
                    print(f"\n====== Regenerate =====")
                    self.generate()
                    print(f"====== Generation Finished =====")
                else:
                    raise FileNotFoundError(f"{subset} for {self.cfg.name}")
                print()
        # Loading
        fed_dict = {}
        for subset in subsets:
            root_path = FileManager.data(os.path.join(self.get_hash_name(), subset),
                                         is_dir=True, overwrite=True)
            file_path = os.path.join(root_path, f"{subset}.pt")

            # with open(file_path, 'rb') as f:
            print(f"Load {subset} data <= {file_path}")
            fed_dict[subset] = torch.load(file_path)
        return fed_dict

    def getLogger(self, fname, subset, root_path=None, logger_name=None):
        if logger_name is None:
            logger_name = self.__class__.__name__
        if root_path is None:
            log_fname = FileManager.data(os.path.join(self.get_hash_name(), subset, fname), is_dir=False,
                                         overwrite=True)
        else:
            log_fname = os.path.join(root_path, fname)
        print(f"Detail log to file: {log_fname}")
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(filename=log_fname, mode='w')
        fh.setFormatter(logging.Formatter('[%(asctime)s - %(levelname)s -'
                                          ' %(filename)s:%(funcName)s] %(message)s'))
        logger.addHandler(fh)
        logger.propagate = False  # Set False to disable stdout print.
        return logger, log_fname


def read_fed_dataset(cfg: DictConfig):
    # if cfg.name == "comb/MnistM":
    if cfg.name.startswith("comb/"):
        from .multi_domain import MDFedDataset as FedDataset
    elif cfg.name in ("Mnist", "MnistM", "SVHN", "USPS") or cfg.name.startswith("ReviewBow") \
            or cfg.name.startswith("ReviewTok") \
            or cfg.name.startswith("Office31") or cfg.name.startswith("OfficeHome65")\
            or cfg.name.startswith("DomainNet"):
        from .federalize import FedExtDataset as FedDataset
    else:
        raise ValueError(f"Unknown data: {cfg.name} with config: \n{OmegaConf.to_yaml(cfg)}")
    fed_dict = FedDataset(cfg).load(generate_if_not_exist=True)

    groups = fed_dict["train"]["hierarchies"] if "hierarchies" in fed_dict["train"] else []
    fed_dict["train"]["hierarchies"] = groups
    fed_dict["test"]["hierarchies"] = groups
    assert fed_dict["train"]['users'] == fed_dict["test"]['users']

    return fed_dict
