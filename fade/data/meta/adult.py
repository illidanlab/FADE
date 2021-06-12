"""Adult dataset.

References
    https://github.com/jctaillandier/adult_neuralnet/blob/master/adult_nn.ipynb
    https://github.com/htwang14/fairness/blob/main/dataloaders/adult.py
"""
import gzip
import os
import pickle
import urllib

import numpy as np
import pickle as pk
import torch
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import time, os, random
from tqdm import tqdm


class Adult(data.Dataset):
    """Adult Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        group (str): Rely on group_by
        group_by (str): white_black
    """

    url = "https://www.kaggle.com/wenruliu/adult-income-dataset"
    classes = ['<=50K', '>50K']

    def __init__(self, root, train=True, transform=None, group='white', create_new=True,
                 group_by='white_black'):
        # init params
        self.root = root
        self.train = train
        subset = "train" if train else "test"
        self.filename = f"{subset}_{group}.npy"
        self.adult_csv_filename = "adult.csv"
        # Num of Train = , Num ot Test
        self.transform = transform
        # self.train_size = 30000

        if create_new and not self._check_exists():
            csv_path = os.path.join(root, self.adult_csv_filename)
            if not os.path.exists(csv_path):
                raise RuntimeError(
                    f"File not found at {csv_path}. Download csv file from {self.url} "
                    f"and place it at {csv_path}")
            prep_adult(original_csv_path=csv_path, save_path=self.root)

        if not self._check_exists():
            filename = os.path.join(self.root, self.filename)
            raise RuntimeError(f"File not found at {filename}. Use create_new=True")

        self.data, self.targets = torch.load(os.path.join(self.root, self.filename))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x, label = self.data[index], self.targets[index]
        x = torch.from_numpy(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, label.astype("int64")

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))


def whiten(X, mean, std):
    X = X - mean
    X = np.divide(X, std + 1e-6)
    return X


def prep_adult(original_csv_path="datasets/adult.csv", save_path='datasets/adult'):
    full_data = pd.read_csv(
        original_csv_path,
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python', skiprows=1,
        na_values="?",
        dtype={0: int, 1: str, 2: int, 3: str, 4: int, 5: str, 6: str, 7: str, 8: str, 9: str,
               10: int, 11: int, 12: int, 13: str, 14: str})

    print('Dataset size: ', full_data.shape[0])

    str_list = []
    for data in [full_data]:
        for colname, colvalue in data.iteritems():
            if type(colvalue[1]) == str:
                str_list.append(colname)
    num_list = data.columns.difference(str_list)

    # Replace '?' with NaN, then delete those rows:
    full_size = full_data.shape[0]
    print('Dataset size Before pruning: ', full_size)
    for data in [full_data]:
        for i in full_data:
            data[i].replace('nan', np.nan, inplace=True)
        data.dropna(inplace=True)
    real_size = full_data.shape[0]

    print('Dataset size after pruning: ', real_size)
    print('We eliminated ', (full_size - real_size), ' datapoints')

    # Take labels out and encode them:
    full_labels = full_data['Target'].copy()
    label_encoder = LabelEncoder()
    full_labels = label_encoder.fit_transform(full_labels)
    print(f"Classes: {np.unique(full_labels)}, {label_encoder.classes_}")

    # Get male_idx and female_idx:
    print(full_data.head())
    male_idx = np.array(full_data['Sex'] == 'Male')  # boolean, len=45222
    female_idx = np.array(full_data['Sex'] == 'Female')  # boolean, len=45222
    print('male_idx:', male_idx[0:5], male_idx.shape)
    print('female_idx:', female_idx[0:5], female_idx.shape)
    np.save(os.path.join(save_path, 'male_idx.npy'), male_idx)
    np.save(os.path.join(save_path, 'female_idx.npy'), female_idx)

    # get race idx:
    indian_idx = np.array(full_data['Race'] == 'Amer-Indian-Eskimo')
    asian_idx = np.array(full_data['Race'] == 'Asian-Pac-Islander')
    black_idx = np.array(full_data['Race'] == 'Black')
    other_idx = np.array(full_data['Race'] == 'Other')
    white_idx = np.array(full_data['Race'] == 'White')

    # Deal with categorical variables:
    full_data = full_data.drop(['Target'], axis=1)
    cat_data = full_data.select_dtypes(include=['object']).copy()
    other_data = full_data.select_dtypes(include=['int']).copy()
    print('cat_data:', cat_data.shape)  # cat_data: (45222, 8)
    print('other_data:', other_data.shape)  # other_data: (45222, 6)

    # Then One Hot encode other Categorical Variables:
    newcat_data = pd.get_dummies(cat_data, columns=[
        "Workclass", "Education", "Country", "Relationship", "Martial Status", "Occupation",
        "Relationship",
        "Race", "Sex"
    ])
    print('newcat_data:', newcat_data.shape)  # newcat_data: (45222, 104)

    # Append all columns back together:
    full_data = pd.concat([other_data, newcat_data], axis=1)
    print('full_data:', full_data.shape)  # full_data: (45222, 110)

    # Dataframe to npy:
    full_data = np.asarray(full_data).astype(np.float32)

    # Split and whitening:
    train_size = 30000  # Given 45222 datapoints, # test_size is the remainder

    train_x = full_data[:train_size,
              :]  # M: train_x[i,-2:] == np.array([0,1]); F: train_x[i,-2:] == np.array([1,0])
    test_x = full_data[train_size:, :]
    # print('train_x:', train_x.shape)
    # print(train_x[0:5,-2:])

    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    # print(mean, std)
    train_x = whiten(train_x, mean, std)
    print('train_x:',
          train_x.shape)  # M: train_x[i,-2:] == np.array([-0.69225496 , 0.692255]); F: train_x[i,-2:] == np.array([1.4445544, -1.4445543])
    test_x = whiten(test_x, mean, std)
    print('test_x:', test_x.shape)

    full_data = np.concatenate([train_x, test_x], axis=0)
    print('full_data:', full_data.shape)
    print()

    train_labels = full_labels[:train_size]
    test_labels = full_labels[train_size:]

    # Save male and female data seperately as .npy files:
    train_male_idx = male_idx[:train_size]
    train_female_idx = female_idx[:train_size]
    test_male_idx = male_idx[train_size:]
    test_female_idx = female_idx[train_size:]

    train_male_data = train_x[train_male_idx]  # train_male_data: (20281, 110)
    train_male_targets = train_labels[train_male_idx]  # train_male_targets: (20281,)
    train_female_data = train_x[train_female_idx]  # train_female_data: (9719, 110)
    train_female_targets = train_labels[train_female_idx]  # train_female_targets: (9719,)
    print('train_male_data:', train_male_data.shape)
    print('train_male_targets:', train_male_targets.shape)
    print('train_female_data:', train_female_data.shape)
    print('train_female_targets:', train_female_targets.shape)
    print()

    test_male_data = test_x[test_male_idx]  # test_male_data: (10246, 110)
    test_male_targets = test_labels[test_male_idx]  # test_male_targets: (10246,)
    test_female_data = test_x[test_female_idx]  # test_female_data: (4976, 110)
    test_female_targets = test_labels[test_female_idx]  # test_female_targets: (4976,)
    print('test_male_data:', test_male_data.shape)
    print('test_male_targets:', test_male_targets.shape)
    print('test_female_data:', test_female_data.shape)
    print('test_female_targets:', test_female_targets.shape)
    print()

    # np.save(os.path.join(save_path, 'train_male_data.npy'), train_male_data)
    # np.save(os.path.join(save_path, 'train_male_targets.npy'), train_male_targets)
    # np.save(os.path.join(save_path, 'train_female_data.npy'), train_female_data)
    # np.save(os.path.join(save_path, 'train_female_targets.npy'), train_female_targets)
    #
    # np.save(os.path.join(save_path, 'test_male_data.npy'), test_male_data)
    # np.save(os.path.join(save_path, 'test_male_targets.npy'), test_male_targets)
    # np.save(os.path.join(save_path, 'test_female_data.npy'), test_female_data)
    # np.save(os.path.join(save_path, 'test_female_targets.npy'), test_female_targets)

    torch.save((train_male_data, train_male_targets), os.path.join(save_path, 'train_male.npy'))
    torch.save((train_female_data, train_female_targets), os.path.join(save_path, 'train_female.npy'))

    torch.save((test_male_data, test_male_targets), os.path.join(save_path, 'test_male.npy'))
    torch.save((test_female_data, test_female_targets), os.path.join(save_path, 'test_female.npy'))

    # Save race data seperately as .npy files:
    train_white_idx = white_idx[:train_size]
    train_black_idx = black_idx[:train_size]
    test_white_idx = white_idx[train_size:]
    test_black_idx = black_idx[train_size:]

    train_white_data = train_x[train_white_idx]  # train_white_data: (25800, 110)
    train_white_targets = train_labels[train_white_idx]  # train_white_targets: (25800,)
    train_black_data = train_x[train_black_idx]  # train_black_data: (2797, 110)
    train_black_targets = train_labels[train_black_idx]  # train_black_targets: (2797,)
    print('train_white_data:', train_white_data.shape)
    print('train_white_targets:', train_white_targets.shape)
    print('train_black_data:', train_black_data.shape)
    print('train_black_targets:', train_black_targets.shape)
    print()

    test_white_data = test_x[test_white_idx]  # test_white_data: (13103, 110)
    test_white_targets = test_labels[test_white_idx]  # test_white_targets: (13103,)
    test_black_data = test_x[test_black_idx]  # test_black_data: (1431, 110)
    test_black_targets = test_labels[test_black_idx]  # test_black_targets: (1431,)
    print('test_white_data:', test_white_data.shape)
    print('test_white_targets:', test_white_targets.shape)
    print('test_black_data:', test_black_data.shape)
    print('test_black_targets:', test_black_targets.shape)
    print()

    # np.save(os.path.join(save_path, 'train_white_data.npy'), train_white_data)
    # np.save(os.path.join(save_path, 'train_white_targets.npy'), train_white_targets)
    # np.save(os.path.join(save_path, 'train_black_data.npy'), train_black_data)
    # np.save(os.path.join(save_path, 'train_black_targets.npy'), train_black_targets)
    #
    # np.save(os.path.join(save_path, 'test_white_data.npy'), test_white_data)
    # np.save(os.path.join(save_path, 'test_white_targets.npy'), test_white_targets)
    # np.save(os.path.join(save_path, 'test_black_data.npy'), test_black_data)
    # np.save(os.path.join(save_path, 'test_black_targets.npy'), test_black_targets)

    torch.save((train_white_data, train_white_targets), os.path.join(save_path, 'train_white.npy'))
    torch.save((train_black_data, train_black_targets), os.path.join(save_path, 'train_black.npy'))

    torch.save((test_white_data, test_white_targets), os.path.join(save_path, 'test_white.npy'))
    torch.save((test_black_data, test_black_targets), os.path.join(save_path, 'test_black.npy'))

    # save data as npy
    z = 0
    start = time.time()
    for x in range(full_data.shape[0]):
        for y in range(2):
            if full_labels[x] == y:
                temp = (full_data[x, :])

                directory = os.path.join(save_path, str(label_encoder.classes_[y]))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                np.save((directory + '/' + str(z) + '.npy'), temp)

                z += 1

    end = time.time()

    print('Time to process: ', end - start)
    print(z, ' datapoints saved to path')
    return label_encoder.classes_
