"""Transform central dataset into the format of federated.

Available datasets:
DigitFive: Mnist | MnistM | SVHN | USPS | SynDigit
Example: python -m fade.data.extend -cn DigitFive name=SVHN n_class=5 n_user=20
"""
import logging
import os
import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

from fade.file import FileManager
from . import FedDataset
from .meta import load_meta_dataset
from .utils import basic_stat_fed_dict

logger = logging.getLogger(__name__)


class FedExtDataset(FedDataset):
    """Extend meta dataset (formatted by pytorch Dataset) to federated.

    To add new dataset:
    1. Update `fade.data.meta.load_meta_dataset` with the data name and configures.
    2. Add the data name entry to `read_fed_dataset` in fade.data.__init__.
    """
    def _generate(self):
        print("Number of users: {}".format(self.cfg.n_user))
        print("Number of classes: {}".format(self.cfg.n_class))

        print(f"=== Reading source dataset ===")
        _root_name = self.cfg.name
        if _root_name.lower().startswith("emnist"):
            _root_name = _root_name[:-1]
        dataset_root_path = FileManager.data(os.path.join(_root_name, "data"), is_dir=True)

        test_set, train_set, _est_max_n_sample_per_shard, _min_n_sample_per_shard = \
            load_meta_dataset(self.cfg, self.cfg.name, dataset_root_path)
        # FIXME This is ad-hoc. Set the default value in cfg.
        if self.cfg.max_n_sample_per_share is None:
            self.cfg.max_n_sample_per_share = _est_max_n_sample_per_shard
        if self.cfg.min_n_sample_per_share is None:
            self.cfg.min_n_sample_per_share = _min_n_sample_per_shard

        print("\n=== Processing training set ===")
        _, SRC_N_CLASS, train_idx_by_class = preprocess_dataset(train_set)

        # n_test_sample, train_idx_by_class, test_idx_by_class, SRC_N_CLASS = \
        #     preprocess_dataset(testset, trainset)
        assert SRC_N_CLASS >= self.cfg.n_class, \
            f"Found N_CLASS_PER_USER={self.cfg.n_class} larger than SRC_N_CLASS={SRC_N_CLASS}"

        class_by_user = split_classes_by_user(SRC_N_CLASS, self.cfg.n_user,
                                              self.cfg.n_class, self.cfg.class_stride)

        # Split class into user shares
        classes, n_share_by_class = np.unique(class_by_user, return_counts=True)
        if len(classes) != SRC_N_CLASS:
            logger.warning(f"After user class splitting, only {len(classes)} are used which "
                           f"is not equal to total {SRC_N_CLASS}.")

        partitioner = Partitioner(
            self.rng, partition_mode=self.cfg.partition_mode,
            max_n_sample_per_share=self.cfg.max_n_sample_per_share,
            min_n_sample_per_share=self.cfg.min_n_sample_per_share,
            max_n_sample=self.cfg.max_n_sample_per_class)
        train_idx_by_user = self.split_data(n_share_by_class, train_idx_by_class,
                                            class_by_user, partitioner)

        print("\n=== Processing test set ===")
        n_test_sample, test_SRC_N_CLASS, test_idx_by_class = preprocess_dataset(test_set)
        # SRC_N_CLASS may not equal test_SRC_N_CLASS
        assert SRC_N_CLASS == test_SRC_N_CLASS
        assert test_SRC_N_CLASS >= self.cfg.n_class, \
            f"Found N_CLASS_PER_USER={self.cfg.n_class} larger than SRC_N_CLASS={test_SRC_N_CLASS}"

        # FIXME for test set, n_test_sample=-1 ==> Use all samples, Otherwise, randomly select.
        assert not hasattr(self.cfg, "n_test_sample"), "Not support cfg: n_test_sample"
        partitioner = Partitioner(self.rng, partition_mode="uni",
                                  max_n_sample=-1, max_n_sample_per_share=-1,
                                  min_n_sample_per_share=2)
        test_idx_by_user = self.split_data(n_share_by_class, test_idx_by_class,
                                           class_by_user, partitioner)

        # Create data structure
        print("\n=== Construct data dict ===")
        fed_dict = {
            'train': self.construct_fed_dict(train_set, train_idx_by_user, is_train=True),
            'test': self.construct_fed_dict(test_set, test_idx_by_user, is_train=False),
        }

        for subset in ["train", "test"]:
            print(f"{subset.upper()} #sample by user:", fed_dict['train']['num_samples'])
            simple_stat(fed_dict['train']['num_samples'])
        print("Total_samples:", sum(fed_dict['train']['num_samples'] + fed_dict['test']['num_samples']), "TRAIN",
              sum(fed_dict['train']['num_samples']), "TEST", sum(fed_dict['test']['num_samples']))

        return fed_dict

    def viz(self, do=True, subset="train", user_idx=0, title=''):
        print(f"Analysis of dataset: {self.cfg.name}")
        fed_dict = self.load(generate_if_not_exist=True)
        dataset = fed_dict[subset]
        print(f"== Basic stat of {subset} set ==")
        basic_stat_fed_dict(dataset, verbose=True)

        import matplotlib.pyplot as plt
        from .utils import plot_sample_size_dist
        ax = plot_sample_size_dist(fed_dict, subset)
        ax.set(title=f"{self.cfg.name} {subset}")
        plt.tight_layout()
        plt.show()

        if hasattr(self.cfg, 'feature_type') and self.cfg.feature_type == "images":
            # Plot the images of one user.
            user = dataset['users'][user_idx]
            from .utils import grid_imshow
            if self.cfg.user_data_format == "index":
                ds = Subset(dataset['dataset'], dataset['user_data'][user]['idx'])
                class_name_fh = lambda i: dataset['dataset'].classes[targets[i].numpy()]
            elif self.cfg.user_data_format == "dataset":
                ds = dataset['user_data'][user]['dataset']
                classes = ds.dataset.classes if isinstance(ds, Subset) else ds.classes
                class_name_fh = lambda i: classes[targets[i].numpy()]
            elif self.cfg.user_data_format == "tensor":
                ds = TensorDataset(dataset['user_data'][user]['x'], dataset['user_data'][user]['y'])
                class_name_fh = lambda i: targets[i]
            else:
                raise ValueError(f"self.cfg.user_data_format: {self.cfg.user_data_format}")
            # print(dataset['dataset'].targets[dataset['user_data'][user]['idx']])
            loader = DataLoader(ds, batch_size=16, shuffle=True)

            imgs, targets = next(iter(loader))
            grid_imshow(imgs, normalize=True, title=class_name_fh)
            plt.show()

    def split_data(self, n_share_by_class, train_idx_by_class, class_by_user,
                   partitioner):
        """The train_idx_by_class will be split into user according to class_by_user and
        n_share_by_class. Partitioned by partitioner."""
        n_user = len(class_by_user)
        data_idx_by_user = [[] for _ in range(n_user)]
        print(f"  # of shards for each class: {n_share_by_class}")
        for cl in range(len(n_share_by_class)):
            n_share = n_share_by_class[cl]
            data_idxs = train_idx_by_class[cl]
            n_smp = len(data_idxs)
            print(f"  Split {n_smp} samples of class {cl} into {n_share} shares.")

            # Split data of class.
            partitions = partitioner(n_smp, n_share)
            simple_stat(partitions)
            end_idxs = [0] + np.cumsum(partitions).tolist()

            # Assign shares to users.
            self.rng.shuffle(data_idxs)  # in-place
            i_share = 0
            for user in range(n_user):
                if cl in class_by_user[user]:
                    start_i = end_idxs[i_share]
                    end_i = end_idxs[i_share+1]
                    data_idx_by_user[user].extend(data_idxs[start_i:end_i])
                    i_share += 1
            assert i_share == n_share, f"Share is not fully used. Generate {n_share} shares, " \
                                       f"but only {i_share} shares are used."
            if end_i < len(data_idxs):
                logger.warning(f"  Use {end_i} out of {len(data_idxs)} samples. Total {len(data_idxs) - end_i} samples are droped.")
        return data_idx_by_user

    def construct_fed_dict(self, dataset, data_idxs_by_user, is_train):
        data_dict = {'users': [], 'user_data': {}, 'num_samples': []}
        if self.cfg.user_data_format == "tensor":
            load_all_data_to_tensor(dataset)
        elif self.cfg.user_data_format == "index":
            data_dict['dataset'] = dataset
        elif self.cfg.user_data_format == "dataset":
            pass
        else:
            raise ValueError(f"self.cfg.user_data_format: {self.cfg.user_data_format}")

        if hasattr(self.cfg, 'niid_distort_train') and self.cfg.niid_distort_train:
            assert self.cfg.user_data_format == "dataset", "Only support niid_distort_train " \
                                                           "when user_data_format=dataset"
        for i in range(self.cfg.n_user):
            uname = 'f_{0:05d}'.format(i)

            data_dict['users'].append(uname)
            if self.cfg.user_data_format == "index":
                data_dict['user_data'][uname] = {
                    'idx': data_idxs_by_user[i]
                }
            elif self.cfg.user_data_format == "dataset":
                if is_train and hasattr(self.cfg, 'niid_distort_train') and self.cfg.niid_distort_train:
                    # Rearrange distort transform in non-iid manner
                    from copy import deepcopy
                    from .meta.distort import DistortTransform, MultiDistortTransform,\
                        PRESET_DISTORTION_SETS
                    from .meta.office_caltech import DistortPathMaker
                    ds = deepcopy(dataset)
                    ts = []
                    if isinstance(self.cfg.distort_train, str) \
                            and self.cfg.distort_train in PRESET_DISTORTION_SETS:
                        distort_train = PRESET_DISTORTION_SETS[self.cfg.distort_train]
                    else:
                        distort_train = self.cfg.distort_train
                    assert self.cfg.n_user == len(distort_train), \
                        f"Not enough distort methods for {self.cfg.n_user} users. " \
                        f"All distort: {distort_train}."
                    for t in ds.transform.transforms:
                        # replace
                        if isinstance(t, MultiDistortTransform):
                            t = DistortTransform(distort_train[i], t.severity)
                        elif isinstance(t, DistortPathMaker):
                            t = DistortPathMaker(distort_train[i], t.severity)
                        ts.append(t)
                    ds.transform = transforms.Compose(ts)
                else:
                    ds = dataset
                data_dict['user_data'][uname] = {
                    'dataset': Subset(ds, data_idxs_by_user[i])
                }
            elif self.cfg.user_data_format == "tensor":
                idxs = data_idxs_by_user[i]
                data_dict['user_data'][uname] = {
                    'x': torch.tensor(dataset.data[idxs], dtype=torch.float32),
                    'y': torch.tensor(dataset.targets[idxs], dtype=torch.int64)
                }
            else:
                raise ValueError(f"self.cfg.user_data_format: {self.cfg.user_data_format}")
            data_dict['num_samples'].append(len(data_idxs_by_user[i]))
        return data_dict


def split_classes_by_user(total_n_class, n_user, n_class_per_user, class_stride, mode="seq"):
    """

    Args:
        total_n_class ():
        n_user ():
        n_class_per_user ():
        class_stride ():
        mode ():

    Returns:
        user_classes is a list where user_classes[i] is a list of classes for user i.
    """
    print(f"Split {total_n_class} classes into {n_user} users. ")
    if mode == "seq":
        print(f"  MODE: {mode}")
        print(f"   {n_class_per_user} classes per user and {class_stride} stride.")
        user_classes = [[] for _ in range(n_user)]

        for user in range(n_user):
            for j in range(n_class_per_user):
                l = (user * class_stride + j) % total_n_class
                user_classes[user].append(l)
        # TODO flatten user_classes
    elif mode == "random":
        # Randomly assign some classes
        raise NotImplementedError()
    else:
        raise RuntimeError(f"Unknown mode: {mode}")
    print(f"Classes by user")
    for user in range(n_user):
        print(f" user {user}: {user_classes[user]}")
    return user_classes


def preprocess_dataset(dataset):
    """Get data indexes for each class and check the sample shape."""
    n_sample = len(dataset)
    n_class = len(dataset.classes)
    data_idx_by_class = rearrange_dataset_by_class(dataset, n_class)

    smp = dataset[0][0]
    if not isinstance(smp, torch.Tensor):
        if isinstance(smp, np.ndarray):
            smp = torch.from_numpy(smp)
        else:
            smp = transforms.ToTensor()(smp)
    print(f"  Total #samples: {n_sample}. sample shape: {smp.shape}")
    print("  #samples per class:\n", [len(v) for v in data_idx_by_class])
    return n_sample, n_class, data_idx_by_class


def rearrange_dataset_by_class(dataset, n_class):
    """Get data indexes for each class"""
    data_by_class = [[] for _ in range(n_class)]
    for i, y in enumerate(dataset.targets):
        data_by_class[y].append(i)
    return data_by_class


class Partitioner:
    """Partition a sequence into shares."""
    def __init__(self, rng, partition_mode="dir",
                 max_n_sample_per_share=10,
                 min_n_sample_per_share=2,
                 max_n_sample=-1
                 ):
        self.rng = rng
        self.partition_mode = partition_mode
        self.max_n_sample_per_share = max_n_sample_per_share
        self.min_n_sample_per_share = min_n_sample_per_share
        self.max_n_sample = max_n_sample

    def __call__(self, n_sample, n_share):
        """Partition a sequence of `n_sample` into `n_share` shares.
        Returns:
            partition: A list of num of samples for each share.
        """
        print(f"{n_sample} samples => {n_share} shards by {self.partition_mode} distribution.")
        if self.max_n_sample > 0:
            n_sample = min((n_sample, self.max_n_sample))
        if self.max_n_sample_per_share > 0:
            n_sample = min((n_sample, n_share * self.max_n_sample_per_share))

        n_sample -= self.min_n_sample_per_share * n_share
        if self.partition_mode == "dir":
            partition = (self.rng.dirichlet(n_share * [1]) * n_sample).astype(int)
        elif self.partition_mode == "uni":
            partition = int(n_sample // n_share) * np.ones(n_share, dtype='int')
        else:
            raise ValueError(f"Invalid partition_mode: {self.partition_mode}")

        partition[-1] += n_sample - np.sum(partition)  # add residual
        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        partition = partition + self.min_n_sample_per_share
        n_sample += self.min_n_sample_per_share * n_share
        # partition = np.minimum(partition, max_n_sample_per_share)
        partition = partition.tolist()

        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        assert len(partition) == n_share, f"{len(partition)} != {n_share}"
        return partition


def simple_stat(arr):
    res = {}
    for metric in ("mean", "std", "max", "min", "median"):
        res[metric] = eval(f'np.{metric}(arr)')
        print(f"  {metric}: {res[metric]:.4g}", end=", ")
    print("")
    return res


def load_all_data_to_tensor(dataset):
    data_loader = DataLoader(dataset, batch_size=len(dataset), drop_last=False,
                             shuffle=False)
    for x, y in data_loader:
        dataset.data, dataset.targets = x, y
    return dataset


@hydra.main(config_name="Office31Av1", config_path="../config/dataset")
def main(cfg):
    ds = FedExtDataset(cfg)
    if hasattr(cfg, "viz") and cfg.viz.do:
        # ds.generate()
        ds.viz(**cfg.viz)
    else:
        ds.generate()


if __name__ == '__main__':
    main()
