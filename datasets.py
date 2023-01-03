from PIL import Image
import pandas as pd
import os

from torchvision import datasets, models, transforms
import glob
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import config

class CelebASubset(Dataset):
    """CelebA Subset dataset."""

    def __init__(self, root_dir, indices, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.identities = pd.read_csv(csv_file)['id'].iloc[indices]
        files = glob.glob(os.path.join(root_dir, '*g'))
        self.images = torch.stack([transform(Image.open(file_path)) for i, file_path in enumerate(files) if i in indices])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.identities.iloc[idx]

def arrange_csv_files():
    train_sub_dataset_size = config.TRAIN_DATASET_SIZE
    test_sub_dataset_size = round(train_sub_dataset_size * 0.2)

    attr_df = pd.read_csv('CelebA_Subset/list_attr_celeba.csv')
    bbox_df = pd.read_csv('CelebA_Subset/list_bbox_celeba.csv')
    eval_partition_df = pd.read_csv('CelebA_Subset/list_eval_partition.csv')
    landmarks_df = pd.read_csv('CelebA_Subset/list_landmarks_align_celeba.csv')
    identity_df = pd.read_csv(r'CelebA_Subset/identity_CelebA.txt', sep="\s+", names=['id'])

    dfs = {'attr': attr_df,
           'bbox': bbox_df,
           'eval_partition': eval_partition_df,
           'landmarks_align': landmarks_df,
           'identity': identity_df
           }
    for name, df in dfs.items():
        if 'image_id' in df:
            df.rename(columns={'image_id': 'Imgname'}, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df[:train_sub_dataset_size + test_sub_dataset_size]
        df.to_csv(f'CelebA_Subset/sub_list_{name}_celeba.csv')

def save_sub_dataset():
    '''Pre process the text and csv data'''

    arrange_csv_files()

    train_sub_dataset_size = config.TRAIN_DATASET_SIZE
    test_sub_dataset_size = round(train_sub_dataset_size*0.2)
    indices_train = range(0, train_sub_dataset_size, 1)
    indices_test = range(train_sub_dataset_size, train_sub_dataset_size + test_sub_dataset_size, 1)

    identity_df = pd.read_csv(r'CelebA_Subset/identity_CelebA.txt', sep="\s+", names=['id'])

    sub_celebA_data_train = CelebASubset('./CelebA_Subset/data', indices_train, identity_df)
    sub_celebA_data_test = CelebASubset('./CelebA_Subset/data', indices_test, identity_df)


    torch.save(sub_celebA_data_train, '/CelebA_Subset/sub_celebA_data_train.pt')
    torch.save(sub_celebA_data_test, '/CelebA_Subset/sub_celebA_data_test.pt')

    # Fix tables and save the with fixed number of samples

