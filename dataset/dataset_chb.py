import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class CustomDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            loading_metafeatures=False, 
            mode='train',
            scale=None
        ):
        super(CustomDataset, self).__init__()
        self.loading_metafeatures = loading_metafeatures
        self.scale = scale
        self.files = [os.path.join(data_dir, mode, file) 
                      for file in os.listdir(os.path.join(data_dir, mode))]

    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['X']
        label = data_dict['y']
        if self.scale:
            data = data * self.scale
        meta = None
        if self.loading_metafeatures:
            meta = data_dict['meta_feature']
        return data, label, meta

    def collate_fn(self, batch):
        data = np.array([x[0] for x in batch])
        label = np.array([x[1] for x in batch])
        data = torch.from_numpy(data).float().contiguous()
        label = torch.from_numpy(label).long()
        if self.loading_metafeatures:
            meta = np.array([x[2] for x in batch])
            meta = torch.from_numpy(meta).float()
            return data, label, meta
        else:
            return data, label


def prepare_datasets_chb(
    data_dir,
    loading_metafeatures=False,
    scale=None
):  
    return {
        mode: CustomDataset(
            data_dir=data_dir,
            loading_metafeatures=loading_metafeatures and mode == "train",
            mode=mode,
            scale=scale
        ) for mode in ["train", "val", "test"]
    }
    