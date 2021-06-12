from __future__ import annotations
import torch
from torchvision import transforms
from fade.file import FileManager
from omegaconf import ListConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omegaconf import OmegaConf
    from typing import Union, Tuple
    from torch.utils.data import Dataset


def load_meta_dataset(cfg: OmegaConf, data_name: str, dataset_root_path: str) \
        -> Tuple[Dataset, Dataset, int, int]:
    """Load meta dataset

    Args:
        cfg: omega config object
        data_name: The data name
        dataset_root_path: Root to where the data (data_name) is stored.

    Returns:
        testset, trainset, est_max_n_sample_per_shard, min_n_sample_per_shard
        testset, trainset are torch Dataset objects.
    """
    est_max_n_sample_per_shard = -1
    min_n_sample_per_shard = 2
    if data_name.lower() == "mnist":
        from torchvision.datasets import MNIST

        nc = cfg.n_channel
        if nc == 3:
            from ..utils import GrayscaleToRgb
            # trans = [transforms.Lambda(lambda x: x.convert("RGB"))]
            trans = [GrayscaleToRgb()]
        else:
            assert nc == 1
            trans = []
        # if cfg.binarize:  # used in DANN experiments. But need to change normalize
        #     trans += [ToBinary()]
        if hasattr(cfg, 'resize'):  # to match SVHN 32
            assert cfg.resize > 0
            trans += [transforms.Resize(cfg.resize)]
        trans += [
            # GrayscaleToRgb(),
            transforms.ToTensor(),
            # TODO when construst MnistM, we also need this.
            # ToBinary(scale=torch.tensor(1.)),  # Used in DANN (official code, https://github.com/pumpikano/tf-dann/blob/master/MNIST-DANN.ipynb
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            transforms.Normalize([.5], [.5])
        ]
        trans = transforms.Compose(trans)
        testset = MNIST(root=dataset_root_path, train=False, transform=trans, download=True)
        trainset = MNIST(root=dataset_root_path, train=True, transform=trans, download=True)

        # %% Configure %%
        est_max_n_sample_per_shard = 10
        min_n_sample_per_shard = 2
    elif data_name.lower().startswith("usps"):  # NOTE: place this before mnist.
        from .usps import USPS

        assert cfg.n_channel in [1, 3], \
            f"Invalid n_channel: {cfg.n_channel}. Expected 1 or 3."
        trans = []
        if cfg.n_channel == 3:
            from ..utils import GrayscaleToRgb
            trans += [GrayscaleToRgb()]
        if cfg.random_crop_rot:
            trans += [
                transforms.RandomCrop(28, padding=4),
                transforms.RandomRotation(10),
            ]
        trans += [
            # GrayscaleToRgb(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        trans = transforms.Compose(trans)
        trainset = USPS(root=dataset_root_path, train=True, transform=trans, download=True)

        trans = []
        if cfg.n_channel == 3:
            from ..utils import GrayscaleToRgb
            trans += [GrayscaleToRgb()]
        trans += [
            # GrayscaleToRgb(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        trans = transforms.Compose(trans)
        testset = USPS(root=dataset_root_path, train=False, transform=trans, download=True)
        trainset.classes = list(range(10))
        testset.classes = list(range(10))

        # %% Configure %%
        est_max_n_sample_per_shard = 10
        min_n_sample_per_shard = 2
    elif data_name.lower().startswith("office"):
        from .office_caltech import get_office_caltech_dataset

        if data_name.lower().startswith("OfficeCal10".lower()):
            source = "OfficeCaltech10"
        elif data_name.lower().startswith("Office31".lower()):
            source = "office31"
        elif data_name.lower().startswith("OfficeHome65".lower()):
            source = "OfficeHome65"
        else:
            raise ValueError(f"data_name: {data_name}")

        # NOTE: DO NOT use a different domain as test domain. Otherwise, you
        #   will see a drop in the loss after each test (evaluation).
        if source == "OfficeHome65":
            if hasattr(cfg, 'domain') and cfg.domain != "default":
                domain = cfg.domain
                test_domain = domain
            else:
                if  data_name[-1].lower() == "a":
                    domain = "Art"
                    test_domain = "Clipart"
                elif data_name[-1].lower() == "c":
                    domain = "Clipart"
                    test_domain = "Art"
                elif data_name[-1].lower() == "p":
                    domain = "Product"
                    test_domain = "Art"
                elif data_name[-1].lower() == "r":
                    domain = "RealWorld"
                    test_domain = "Art"
                else:
                    raise ValueError(f"data_name: {data_name}")
        else:  # Office31
            if hasattr(cfg, 'domain') and cfg.domain != "default":
                domain = cfg.domain
                test_domain = domain
            else:
                if data_name[-1].lower() == "a":
                    domain = "amazon"
                    test_domain = "webcam"
                elif data_name[-1].lower() == "d":
                    domain = "dslr"
                    test_domain = "webcam"
                elif data_name[-1].lower() == "w":
                    domain = "webcam"
                    test_domain = "dslr"
                elif source == "OfficeCaltech10" and data_name[-1].lower() == "c":
                    domain = "caltech10"
                    test_domain = "amazon"
                else:
                    raise ValueError(f"data_name: {data_name}")

        if hasattr(cfg, "ood_test_domain"):  # out-of-distribution test
            if isinstance(cfg.ood_test_domain, bool):
                if not cfg.ood_test_domain:
                    test_domain = domain
            else:
                assert isinstance(cfg.ood_test_domain, str)
                if cfg.ood_test_domain == "self":
                    test_domain = domain
                elif cfg.ood_test_domain == "default":
                    pass
                else:
                    test_domain = cfg.ood_test_domain
        else:
            test_domain = domain
        if cfg.feature_type == "images":
            # standard AlexNet and ResNet101 preprocessing.
            # Refer to
            #   - AlexNet: https://pytorch.org/hub/pytorch_vision_alexnet/
            #   - ResNet: https://pytorch.org/hub/pytorch_vision_resnet/
            train_data_kwargs = {}
            train_trans_ = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip()
            ]
            if hasattr(cfg, "distort_train") and cfg.distort_train != "none":
                raise NotImplementedError(f"Not support distortion anymore.")
            train_trans_ += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            train_trans = transforms.Compose(train_trans_)

            test_data_kwargs = {}
            test_trans_ = [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
            ]
            if hasattr(cfg, "distort") and cfg.distort != "none" and cfg.severity > 0:
                raise NotImplementedError(f"Not support distortion anymore.")
            test_trans_ += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            test_trans = transforms.Compose(test_trans_)

            trainset = get_office_caltech_dataset(source=source, domain=domain,
                                                  transform=train_trans, **train_data_kwargs)
            testset = get_office_caltech_dataset(source=source, domain=test_domain,
                                                 transform=test_trans, **test_data_kwargs)
        else:
            trainset = get_office_caltech_dataset(source=source, domain=domain,
                                                  feature_type=cfg.feature_type)
            testset = get_office_caltech_dataset(source=source, domain=test_domain,
                                                 feature_type=cfg.feature_type)
    elif data_name.lower().startswith("adult"):
        from .adult import Adult
        root = FileManager.data('adult', is_dir=True)
        if data_name.lower().endswith("w"):
            group = "white"
            group_by = "white_black"
        elif data_name.lower().endswith("b"):
            group = "black"
            group_by = "white_black"
        elif data_name.lower().endswith("f"):
            group = "female"
            group_by = "gender"
        elif data_name.lower().endswith("m"):
            group = "male"
            group_by = "gender"
        else:
            raise ValueError(f"data_name : {data_name}")
        trainset = Adult(root, train=True, group_by=group_by, group=group)
        testset = Adult(root, train=False, group_by=group_by, group=group)
    else:
        raise ValueError(f"Unknown data name: {data_name}")
    return testset, trainset, est_max_n_sample_per_shard, min_n_sample_per_shard
