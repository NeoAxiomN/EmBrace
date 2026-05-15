import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os.path as osp


def load_data_per_subject(
    load_path, 
    sub, 
    loading_metafeatures=True
):
    sub_code = 'sub' + str(sub) + '.pkl'
    path = osp.join(load_path, sub_code)
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    data = dataset['data']
    label = dataset['label']
    if loading_metafeatures:
        meta_feature = dataset['meta_feature']
    else:
        meta_feature = None
    return data, label, meta_feature


def load_data(
    load_path, 
    load_idx, 
    concat=True, 
    loading_metafeatures=True
):
    data, label, meta_feature = [], [], []
    for i, idx in enumerate(load_idx):
        data_per_sub, label_per_sub, meta_feature_per_sub = load_data_per_subject(
            load_path=load_path, 
            sub=idx, 
            loading_metafeatures=True
        )
        data.extend(data_per_sub)
        label.extend(label_per_sub)
        if loading_metafeatures:
            meta_feature.extend(meta_feature_per_sub)
    if concat:
        data = np.concatenate(data)  # --> seg, chan, data
        label = np.concatenate(label)
        if loading_metafeatures:
            meta_feature = np.concatenate(meta_feature)

    if loading_metafeatures:
        return data, label, meta_feature
    else:
        return data, label

def normalize(train, val, test):
    # input should be seg, chan, time/f
    # data: sample x channel x data
    mean = np.mean(train, axis=(0, 2, 3), keepdims=True)
    std = np.std(train, axis=(0, 2, 3), keepdims=True)
    std[std == 0] = 1e-6
    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std
    return train, val, test


class CustomDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            loading_metafeatures=False, 
            mode='train',
            scale=None
        ):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.loading_metafeatures = loading_metafeatures
        self.scale = scale
        if mode == 'train':
            self.load_idx = range(0, 16)
        elif mode == 'val':
            self.load_idx = range(16, 21)
        elif mode == 'test':
            self.load_idx = range(21, 26)

        self.data, self.label, self.meta_feature = load_data(
            load_path=self.data_dir, 
            load_idx=self.load_idx, 
            concat=True, 
            loading_metafeatures=True
        )
        if not self.loading_metafeatures:
            self.meta_feature = None

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        if self.scale:
            data = data * self.scale
        
        meta = None
        if self.loading_metafeatures:
            meta = self.meta_feature[idx]
        return data, label, meta


    def collate_fn(self, batch):
        data = np.array([x[0] for x in batch])
        label = np.array([x[1] for x in batch])
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).long()
        if self.loading_metafeatures:
            meta = np.array([x[2] for x in batch])
            return data, label, torch.from_numpy(meta).float()
        else:
            return data, label


def prepare_datasets_attention(
    data_dir, 
    loading_metafeatures=False,
    scale=None,
):
    datasets = {
        mode: CustomDataset(
            data_dir=data_dir,
            loading_metafeatures=loading_metafeatures and mode == "train",
            mode=mode,
            scale=scale
        ) for mode in ["train", "val", "test"]
    }
    
    train_norm, val_norm, test_norm = normalize(
        datasets["train"].data, 
        datasets["val"].data, 
        datasets["test"].data
    )
    
    datasets["train"].data = train_norm
    datasets["val"].data = val_norm
    datasets["test"].data = test_norm
    
    return datasets

