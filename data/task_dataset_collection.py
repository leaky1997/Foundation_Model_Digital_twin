'''
This is the main file for the dataset collection. It contains the class definition for the dataset collection.
By Liqi
'''

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.dataset_utils import *

import numpy as np
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, args, flag):
        self.flag = flag
        self.args = args
        self.load_data(args.data_dir, args.target)
        self.create_splits()
        self.select_data_based_on_flag()
        
    def load_data(self, data_dir, target):
        self.data = np.load(f"{data_dir}{target}_data.npy").astype(np.float32)
        self.labels = np.load(f"{data_dir}{target}_label.npy").astype(np.float32)
    
    def create_splits(self):
        train_ratio = 0.6
        val_ratio = 0.1
        test_ratio = 1 - (train_ratio + val_ratio)

        self.train_indices, self.val_indices, self.test_indices = [], [], []
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]
            n_train = int(len(label_indices) * train_ratio)
            n_val = int(len(label_indices) * val_ratio)
            n_test = len(label_indices) - n_train - n_val

            self.train_indices.extend(label_indices[:n_train])
            self.val_indices.extend(label_indices[n_train:n_train + n_val])
            self.test_indices.extend(label_indices[n_train + n_val:])
    
    def select_data_based_on_flag(self):
        if self.flag == 'train':
            selected_indices = self.train_indices
        elif self.flag == 'val':
            selected_indices = self.val_indices
        elif self.flag == 'test':
            selected_indices = self.test_indices
        else:
            raise ValueError("Invalid flag. Please choose from 'train', 'val', or 'test'.")

        self.selected_data = self.data[selected_indices]
        self.selected_labels = self.labels[selected_indices]

    def __len__(self):
        return len(self.selected_data)

    def __getitem__(self, idx):
        sample = self.selected_data[idx]
        label = self.selected_labels[idx]
        return sample, label


import numpy as np
import torch
from torch.utils.data import Dataset
import os

class BaseDataset(Dataset):
    def __init__(self, args, flag, task_type):
        self.flag = flag
        self.args = args
        self.task_type = task_type
        self.load_data(args.data_dir)
        self.create_splits()
        self.select_data_based_on_flag()

    def load_data(self, data_dict):
        self.data = []
        self.labels = []
        self.has_labels = True

        for name, file_path in data_dict.items():
            if 'data' in name:
                self.data.append(np.load(file_path).astype(np.float32))
            elif 'label' in name:
                self.labels.append(np.load(file_path).astype(np.float32))

        self.data = np.concatenate(self.data, axis=0)
        
        if self.labels:
            self.labels = np.concatenate(self.labels, axis=0)
        else:
            self.has_labels = False
            self.labels = np.zeros(len(self.data))  # 如果没有标签文件，生成一个全零的标签

    def create_splits(self):
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 1 - (train_ratio + val_ratio)

        self.train_indices, self.val_indices, self.test_indices = [], [], []
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]
            n_train = int(len(label_indices) * train_ratio)
            n_val = int(len(label_indices) * val_ratio)
            n_test = len(label_indices) - n_train - n_val

            self.train_indices.extend(label_indices[:n_train])
            self.val_indices.extend(label_indices[n_train:n_train + n_val])
            self.test_indices.extend(label_indices[n_train + n_val:])

    def select_data_based_on_flag(self):
        if self.flag == 'train':
            selected_indices = self.train_indices
        elif self.flag == 'val':
            selected_indices = self.val_indices
        elif self.flag == 'test':
            selected_indices = self.test_indices
        else:
            raise ValueError("Invalid flag. Please choose from 'train', 'val', or 'test'.")

        self.selected_data = self.data[selected_indices]
        self.selected_labels = self.labels[selected_indices]

    def __len__(self):
        return len(self.selected_data)

    def __getitem__(self, idx):
        sample = self.selected_data[idx]
        label = self.selected_labels[idx]
        return sample, label

class ClassificationDataset(BaseDataset):
    def __init__(self, args, flag):
        super().__init__(args, flag, task_type='classification')

class AnomalyDetectionDataset(BaseDataset):
    def __init__(self, args, flag):
        super().__init__(args, flag, task_type='anomaly_detection')

class ImputationDataset(BaseDataset):
    def __init__(self, args, flag):
        super().__init__(args, flag, task_type='imputation')

class ForecastingDataset(BaseDataset):
    def __init__(self, args, flag):
        super().__init__(args, flag, task_type='forecasting')
