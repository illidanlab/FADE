import os
import numpy as np
import copy
import torch
import pandas as pd
from torch.utils.data import Subset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from omegaconf import DictConfig, OmegaConf
from typing import Union, List, Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import logging
from tqdm import tqdm

from .meta.office_caltech import DefaultImageLoader

logger = logging.getLogger(__name__)

from fade.file import FileManager
from . import read_fed_dataset


def simple_stat(arr):
    res = {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "max": np.max(arr),
        "min": np.min(arr),
        "median": np.median(arr),
    }
    for metric, value in res.items():
        print(f"  {metric}: {value:.4g}", end=", ")
    print("")
    return res


def basic_stat_fed_dict(data_dict, verbose=False, group_by="none"):
    users = data_dict['users']
    num_users = len(users)
    num_samples = data_dict['num_samples']
    # if group_by == "group":
    #     groups = data_dict[]

    if verbose:
        example_user = users[0]
        shape = None

        classes = []
        n_classes = []
        mtl_flag = True
        for k in tqdm(users, desc="stat users"):
            if 'idx' in data_dict['user_data'][k]:
                cs = np.unique(data_dict['dataset'].targets[i] for i in data_dict['user_data'][k]['idx']).tolist()
                sample, label = data_dict['dataset'][data_dict['user_data'][k]['idx'][0]]
            elif 'dataset' in data_dict['user_data'][k]:
                # labels = [y for x, y in data_dict['user_data'][k]['dataset']]
                if isinstance(data_dict['user_data'][k]['dataset'], Subset):
                    ds = data_dict['user_data'][k]['dataset'].dataset
                    labels = [ds.targets[i] for i in data_dict['user_data'][k]['dataset'].indices]
                else:
                    ds = data_dict['user_data'][k]['dataset']
                    labels = ds.targets
                labels = np.array(labels)
                cs = np.unique(labels).tolist()
                sample, label = data_dict['user_data'][k]['dataset'][0]

                if len(labels.shape) > 1 and mtl_flag:
                    print(f" Multi-task w/ {labels.shape[1]} tasks")
                    mtl_flag = False
            elif 'x' in data_dict['user_data'][k] and 'y' in data_dict['user_data'][k]:
                cs = np.unique(data_dict['user_data'][k]['y']).tolist()
                sample = data_dict['user_data'][k]['x'][0]
            else:
                raise ValueError(f"Expected keys are: x, y, dataset, idx. Not found "
                                 f"wanted data keys in {data_dict['user_data'][k].keys()}")
            classes.append(cs)
            n_classes.append(len(cs))
            if shape is None:
                shape = sample.shape
            else:
                if shape != sample.shape:
                    logger.warning(f"User {k} has sample of shape not equal to other users' sample "
                                   f"shape: {shape} != {sample.shape} (others)")

            # if len(data_dict['hierarchies']) > 0:
            #     groups = data_dict['hierarchies']
            #     # TODO class balance for each group
            #
            #     # group balance check
            #     uni_groups, n_user_by_group = np.unique(groups, return_counts=True)
            #     print(f" Num user by group: "
            #           f"{dict((g, nu) for g, nu in zip(uni_groups, n_user_by_group))}")
        # classes = np.array(classes)

        print(f" Num of users: {num_users}")
        print(f" Sample: shape {shape}, type {type(sample)}")
        print(f" Num of samples (by user):", end=" ")
        for stat_method in ["sum", "mean", "median", "min", "max", "std"]:
            print(f"{stat_method} {eval('np.'+stat_method+'(num_samples)'):7.1f}", end=", ")
        print()
        if False:  # FIXME: ad-hoc 'float' in str(classes[0].dtype):
            print("mean of objective (per user)", [np.mean(c) for c in classes])
        else:
            classes_ = []
            for cs in classes:
                classes_.extend(cs)
            print(f" Total num of classes: {len(np.unique(classes_))}")
            print(f" Num of classes per user: "
                  f"mean {np.mean(n_classes):.3f}, min {np.min(n_classes)}, max {np.max(n_classes)}")
    return num_samples, num_users


def basic_stat(data_dict, verbose=False):
    """
    Args:
        data_dict: Dict<user_id: Dict<'x': samples, 'y': labels> >.
    """
    num_users = len(data_dict)
    sample = data_dict[list(data_dict.keys())[0]]['x']
    num_samples = np.array([len(data_dict[k]['y']) for k in data_dict])

    if verbose:
        shape = sample.shape[1:]
        classes = [np.unique(data_dict[k]['y']) for k in data_dict]
        n_classes = np.array([len(np.unique(data_dict[k]['y'])) for k in data_dict])

        print(f" Num of users: {num_users}")
        print(f" Sample: shape {shape}, type {type(sample)}")
        print(f" Num of samples (by user):", end=" ")
        for stat_method in ["mean", "median", "min", "max", "std"]:
            print(f"{stat_method} {eval('np.'+stat_method+'(num_samples)'):7.1f}", end=", ")
        if 'float' in str(classes[0].dtype):
            print("mean of objective (per user)", [np.mean(c) for c in classes])
        else:
            print(f" Total num of classes: {len(np.unique(np.concatenate(classes, axis=0)))}")
            print(f" Num of classes per user: "
                  f"mean {np.mean(n_classes):.1f}, min {np.min(n_classes)}, max {np.max(n_classes)}")
    print()
    return num_samples, num_users


def user_class_shards(data_dict, visualize=False, ax=None, show_xticks=True, by_percent=False):
    """Stat the class shards in users.

    Args:
        data_dict: Dict<user_id: Dict<'x': samples, 'y': labels> >.
        visualize: Plot if True.
        by_percent: Transform the count to frequency (percent).

    Returns:
        ucs: An dict<User_name: Dict<class: #samples>> where each element is the number of
            samples of each class in each user.
        ucs_arr: An array of shape (n_user, n_class).
    """
    ucs = dict()
    all_classes = []
    for _id in data_dict:
        y = data_dict[_id]['y']
        classes, cnts = np.unique(y, return_counts=True)
        ucs[_id] = dict((c, cnt) for c, cnt in zip(classes, cnts))
        all_classes = np.union1d(classes, all_classes)
    users = list(data_dict.keys())
    ucs_arr = []
    for _id in users:
        ucs_arr.append([ucs[_id][c] if c in ucs[_id] else 0 for c in all_classes])
    ucs_arr = np.cumsum(np.array(ucs_arr), axis=1)  # shape: (n_user, n_class)
    if by_percent:
        ucs_arr = ucs_arr / ucs_arr[:, -1:] * 100.
    if visualize:
        for ic in np.arange(len(all_classes))[::-1]:
            ax.bar(list(range(len(ucs_arr))), ucs_arr[:, ic], label=f"class-{all_classes[ic]}")
    if show_xticks:
        ax.set(xticks=list(range(len(ucs_arr))))
    return ucs, ucs_arr


def save_data(data_name, data_dict):
    """Save data into files.

    Args:
        data_name ():
        data_dict (dict): Should include keys: 'train' and 'test' at least.
    """
    for s in data_dict:
        fname = FileManager.data(os.path.join(data_name, s, f"{s}.pt"), is_dir=False, overwrite=True)
        with open(fname, 'wb') as outfile:
            print(f"Dumping {s} data => {fname}")
            torch.save(data_dict[s], outfile)


def load_fed_dataset(dataset: str, subset="train",
                     list_keys=("users", "hierarchies", "num_samples"),
                     dict_keys=("user_data",)):
    """Load dataset in fed format without unpacking.
    If multiple files exist, we will load alll and combine into one dict.

    Returns:
        fed_data_dict: A dict like {"users": [], "hierarchies": [], "user_data": {}}
    """
    # os.path.join('data', dataset, 'data', 'train')
    if "-" in dataset:
        dataset, data_fld = dataset.split("-")
    else:
        # data_fld = 'data'
        data_fld = ''

    data_dir = FileManager.data(os.path.join(dataset, data_fld, subset), is_dir=True,
                                      create_new=False)
    # test_data_dir = FileManager.data(os.path.join(dataset, data_fld, 'test'), is_dir=True,
    #                                  create_new=False)
    # clients = []
    # groups = []
    fed_data_dict = {}
    # fed_data_dict = {"users": [], "hierarchies": [], "user_data": {}, "num_samples": []}

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.json') or f.endswith('.pt')]
    for f in all_files:
        file_path = os.path.join(data_dir, f)
        print(f"Load data <== {file_path}")
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            raise ValueError(f"Found json file. Unless you intend to do this, you should use `.pt` file for dataset. "
                             f"`.json` is slow in loading. Try to remove .json file to fix this.")
            # with open(file_path, 'r') as inf:
            #     cdata = json.load(inf)
        else:
            raise TypeError(f"file: {file_path}")
        for k in list_keys:
            if k in cdata:
                fed_data_dict.setdefault(k, []).extend(cdata[k])
        for k in dict_keys:
            if k in cdata:
                fed_data_dict.setdefault(k, {}).update(cdata[k])
        if len(cdata.keys()) > 4:
            raise ValueError(f"Expected three keys. But we get {len(cdata.keys())} "
                             f"keys from the dataset ({dataset}, {subset}): {list(cdata.keys())}")
    return fed_data_dict


def get_dataset_config(name):
    return OmegaConf.load(os.path.join(os.path.dirname(__file__), name + ".yaml")).dataset


def read_data(dataset, load_config=False):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Args:
        dataset (str|DictConfig):
        load_config (bool): Enforce to use hydra config to access dataset if `dataset`
            is str.

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    logger.warning(f"read_data will be deprecated. Use read_fed_dataset instead.")
    if isinstance(dataset, str) and load_config:
        dataset = get_dataset_config(dataset)
    if isinstance(dataset, DictConfig):
        return read_fed_dataset(dataset)

    train_dict = load_fed_dataset(dataset, "train")
    test_dict = load_fed_dataset(dataset, "test")
    groups = train_dict["hierarchies"] if "hierarchies" in train_dict else []
    if "user_data" in train_dict:
        data_type = "fed"
    # elif "task_data" in train_dict:
    #     data_type = "mtl"
    else:
        raise ValueError(f"train_dict keys: {train_dict.keys()}")
    if data_type == "fed":
        train_data = train_dict["user_data"]

        test_data = test_dict["user_data"]
    # elif data_type == "mtl":
    #     train_data = train_dict["task_data"]
    #
    #     test_data = test_dict["task_data"]
    else:
        raise ValueError(f"data_type: {data_type}")

    users = list(sorted(train_data.keys()))
    # TODO verify the consistency of client keys.

    return users, groups, train_data, test_data


def extract_user_data(user_index, data):  # , dataset):
    """`data` is from the output of `read_fed_data`"""
    if isinstance(data, dict):
        train_data = data["train"]["user_data"]
        test_data = data["test"]["user_data"]
        groups = data["train"]["hierarchies"] if "hierarchies" in data["train"] else []
        users = data["train"]['users']
    else:
        users, groups, train_data, test_data = data
    user_id = users[user_index]
    if len(groups) > 0:
        if isinstance(groups[user_index], list):
            # TODO fix this.
            print(f"WARNING: found vec groups.")
            assert not isinstance(groups[user_index][0], str)
            # FIXME ad-hoc only use the first group if the groups include multiple values.
            group = (groups[user_index][0], groups[user_index][0])
        else:
            group_names = np.unique(groups).tolist()
            group_name = groups[user_index]
            # print(f"### np.unique(groups): {group_names}")
            # note: the unique return SORTED array of groups.
            group = (group_names.index(group_name), group_name)
    else:
        group = (0, 'all')
    data_dict = {'train': train_data, 'test': test_data}
    new_data_dict = {}
    for subset in data_dict:
        subset_data = data_dict[subset]
        if 'idx' in subset_data[user_id]:
            assert isinstance(data, dict), f"data is not dict but {type(data)}"
            assert 'dataset' in data[subset], f"Not found 'dataset' in {subset} data. {data[subset].keys()}"
            dataset = data[subset]['dataset']  # type: Dataset
            idxs = subset_data[user_id]['idx']
            subset_data = Subset(dataset, idxs)
        elif 'dataset' in subset_data[user_id]:
            subset_data = subset_data[user_id]['dataset']
        else:
            subset_data = subset_data[user_id]
            X_set, y_set = subset_data['x'], subset_data['y']
            if not isinstance(X_set, torch.Tensor):
                X_set = torch.Tensor(X_set).type(torch.float32)
                y_set = torch.Tensor(y_set).type(torch.int64)
            subset_data = [(x, y) for x, y in zip(X_set, y_set)]
        new_data_dict[subset] = subset_data
    return user_id, group, new_data_dict['train'], new_data_dict['test']


def grid_imshow(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        img_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        title=None,
) -> torch.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        img_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if img_range is not None:
            assert isinstance(img_range, tuple), \
                "img_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, img_range):
            if img_range is not None:
                norm_ip(t, img_range[0], img_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, img_range)
        else:
            norm_range(tensor, img_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    fig, axs = plt.subplots(ymaps, xmaps, figsize=(.03 * width * xmaps, .03 * height * ymaps))
    axs = np.atleast_2d(axs)
    # grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            ax = axs[y, x]
            if k < nmaps:
                npimg = tensor[k].numpy()
                ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
                if title is not None:
                    ax.set(title=title(k))
            ax.axis('off')
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            # grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
            #     2, x * width + padding, width - padding
            # ).copy_(tensor[k])

            k = k + 1
    return fig, axs


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class ToBinary:
    """Convert a grayscale image to rgb"""
    def __init__(self, scale=255):
        self.scale = scale

    def __call__(self, image):
        ret = (image > 0).int() * self.scale
        return ret


class SqueezeTarget:
    """Try to squeeze dimensions of targets.
    """

    def __call__(self, y):
        """
        Args:
            y (): Label of a sample.

        Returns:
            Tensor: Converted image.
        """
        return y[0] if len(y) == 1 else y

    def __repr__(self):
        return 'squeeze dimension of targets'


def make_static_dataset_copy(dataset):
    dataset = copy.deepcopy(dataset)
    if isinstance(dataset, Subset):
        ds = dataset.dataset
    else:
        ds = dataset
    if hasattr(ds, 'transform'):
        ds.transform = remove_random_transforms(ds.transform)
    else:
        logger.debug(f"Not found transform. Skip")
    # assert not hasattr(ds, 'target_transform') or ds.target_transform is None
    return dataset


def remove_random_transforms(trans):
    """Remove the random transform in `trans`."""
    if trans is None:
        return trans
    assert isinstance(trans, transforms.Compose)
    new_trans = []
    logger.debug(f"old trans: {trans}")
    for t in trans.transforms:
        if isinstance(t, (transforms.Resize, transforms.ToTensor, transforms.Normalize,
                          DefaultImageLoader)):
            new_trans.append(t)
        elif isinstance(t, transforms.RandomCrop):
            t_ = transforms.CenterCrop(t.size)
            new_trans.append(t_)
            logger.debug(f" Replace {t} by {t_}")
        elif isinstance(t, transforms.RandomResizedCrop):
            t_ = transforms.Resize(t.size)
            new_trans.append(t_)
            logger.debug(f" Replace {t} by {t_}")
        elif isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip, transforms.RandomRotation)):
            logger.debug(f" Skip transform: {t}")
        else:
            raise TypeError(f"Unable to translate transform: {t}")
    new_trans = transforms.Compose(new_trans)
    logger.debug(f"New trans: {new_trans}")
    return new_trans


def update_dataset_targets(dataset, new_targets, indices=None):
    """Update dataset targets in place."""
    if indices is None:
        indices = list(range(len(dataset)))
    if isinstance(dataset, Subset):
        indices = dataset.indices
        update_dataset_targets(dataset.dataset, new_targets, indices=indices)
        # for ii in indices:
        #     dataset.dataset.targets[indices[ii]] = new_targets[ii]
    elif isinstance(dataset, DatasetFolder):
        assert len(indices) == len(new_targets), f"len(indices) ({len(indices)}) != len(new_targets) ({len(new_targets)})"
        for i in range(len(indices)):
            s = dataset.samples[indices[i]]
            dataset.samples[indices[i]] = (s[0], new_targets[i])
        # for i in range(len(dataset.samples)):
        #     dataset.samples[indices[i]] = (s[0], 0)
        dataset.targets = [s[1] for s in dataset.samples]
    elif isinstance(dataset, list) and isinstance(dataset[0], tuple) and len(dataset[0])==2:
        for i in range(len(indices)):
            s = dataset[indices[i]]
            dataset[indices[i]] = (s[0], new_targets[i])
    else:
        raise TypeError(f"Type: {type(dataset)}")
        # TODO use this for other known types. Do not use this for all Dataset.
        # assert hasattr(dataset, 'targets')
        # for ii in indices:
        #     dataset.targets[indices[ii]] = new_targets[ii]


def merge_fed_dataset_users(dataset):
    """Merge all users into a single table."""
    assert isinstance(dataset, dict)

    dfs = dict((k, []) for k in ['x', 'y', 'group', 'user'])  # a list of all users' dataframes
    for user, group in zip(dataset['users'], dataset['hierarchies']):
        X = dataset['user_data'][user]['x']
        y = dataset['user_data'][user]['y']
        df = {'x': X.tolist(), 'y': y.tolist(), 'group': [group]*len(y), 'user': [user]*len(y)}
        for k in df:
            dfs[k].extend(df[k])
    return dfs


def compute_mean_distance(data_dict):
    users = data_dict["users"]
    n_user = len(users)
    D = np.zeros((n_user, n_user))
    for i in range(n_user):
        X1 = data_dict['user_data'][users[i]]['x']
        for j in range(i+1, n_user):
            X2 = data_dict['user_data'][users[j]]['x']
            d = np.linalg.norm(np.mean(X1, axis=0) - np.mean(X2, axis=0))
            D[i, j] = d
            D[j, i] = d
    return D


def plot_sample_size_dist(fed_dict, subset):
    users = fed_dict[subset]["users"]
    n_user = len(fed_dict[subset]["users"])
    from .utils import extract_user_data
    n_smp_by_class = {}
    for i_user in range(n_user):
        user_id, group, train_set, test_set = extract_user_data(i_user, fed_dict)
        subset_data = train_set if subset == "train" else test_set
        for _, y in subset_data:
            n_smp_by_class.setdefault(y, [0] * n_user)
            n_smp_by_class[y][i_user] += 1
    all_classes = sorted(list(n_smp_by_class.keys()))
    n_smp_mat = np.array([n_smp_by_class[cl] for cl in all_classes]).T
    # print(n_smp_mat)
    m_classes, m_users = np.meshgrid(all_classes, np.arange(0, n_user))
    m_classes = m_classes.reshape((-1,))
    m_users = m_users.reshape((-1,))
    n_smp_mat = n_smp_mat.reshape((-1,))

    import seaborn as sns
    import matplotlib.pyplot as plt
    max_size = 240
    n_smp_mat = n_smp_mat * 1. * max_size / np.max(n_smp_mat)
    fig, ax = plt.subplots(1, 1)
    plt.scatter(m_classes, m_users, s=n_smp_mat, alpha=0.5)
    # sns.scatterplot(x=m_classes, y=m_users, size=n_smp_mat, ax=ax)
    ax.set(xlabel="class", ylabel="user", yticks=list(range(n_user)), yticklabels=users)
    return ax


def get_dataloader_options(dataset):
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset.dataloader_options() \
        if hasattr(dataset, "dataloader_options") else {}
