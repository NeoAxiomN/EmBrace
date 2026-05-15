import torch
from torch.utils.data import Dataset
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir,
        loading_metafeatures=False,
        scale=None,
    ):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.loading_metafeatures = loading_metafeatures
        self.scale = scale

    def __len__(self):
        return len((self.data_dir))

    def __getitem__(self, idx):
        seq_path = self.data_dir[idx][0]
        label_path = self.data_dir[idx][1]
        seq = np.load(seq_path)
        label = np.load(label_path)
        if self.scale:
            seq = seq * self.scale
        meta = None
        if self.loading_metafeatures:
            meta_path = self.data_dir[idx][2]
            meta = np.load(meta_path)
        return seq, label, meta

    def collate_fn(self, batch):
        x_seq = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        x_tensor = torch.from_numpy(x_seq).float()
        y_tensor = torch.from_numpy(y_label).long()
        
        x_tensor = x_tensor.view(-1, 6, 30, 200)
        # y_tensor = y_tensor.view(-1)
        if self.loading_metafeatures:
            z_meta = np.array([x[2] for x in batch]) 
            return x_tensor, y_tensor, torch.from_numpy(z_meta).float()
        else:
            return x_tensor, y_tensor




class LoadDataset(object):
    def __init__(
        self, 
        data_dir,
        loading_metafeatures=False,
        scale=None
    ):
        self.loading_metafeatures = loading_metafeatures
        self.scale = scale
        self.seqs_dir = os.path.join(data_dir, 'seq')
        self.labels_dir = os.path.join(data_dir, 'labels')
        self.meta_dir = os.path.join(data_dir, 'meta_features')
        self.data_path_triples = self.load_path()

    def get_datasets(self):
        train_pairs, val_pairs, test_pairs = self.split_dataset(self.data_path_triples)
        return {
            "train": CustomDataset(
                data_dir=train_pairs, 
                loading_metafeatures=self.loading_metafeatures, 
                scale=self.scale
            ),
            "val": CustomDataset(
                data_dir=val_pairs, 
                loading_metafeatures=False, 
                scale=self.scale
            ),
            "test": CustomDataset(
                data_dir=test_pairs, 
                loading_metafeatures=False, 
                scale=self.scale
            ),
        }

    def load_path(self):
        all_path_triples = []
        # subject_nums = os.listdir(self.seqs_dir)
        # print(subject_nums)
        subject_dirs_seq = []
        subject_dirs_labels = []
        subject_dirs_meta = []
        for subject_num in range(1, 11):
            subject_dirs_seq.append(os.path.join(self.seqs_dir, f'ISRUC-group3-{subject_num}'))
            subject_dirs_labels.append(os.path.join(self.labels_dir, f'ISRUC-group3-{subject_num}'))
            subject_dirs_meta.append(os.path.join(self.meta_dir, f'ISRUC-group3-{subject_num}'))

        for subject_seq, subject_label, subject_meta in zip(subject_dirs_seq, subject_dirs_labels, subject_dirs_meta):
            # print(subject_seq, subject_label)
            subject_pairs = []
            seq_fnames = os.listdir(subject_seq)
            label_fnames = os.listdir(subject_label)
            meta_fnames = os.listdir(subject_meta)
            # print(seq_fnames)
            for seq_fname, label_fname, meta_fname in zip(seq_fnames, label_fnames, meta_fnames):
                subject_pairs.append((
                    os.path.join(subject_seq, seq_fname), 
                    os.path.join(subject_label, label_fname),
                    os.path.join(subject_meta, meta_fname)))
            all_path_triples.append(subject_pairs)
        # print(seqs_labels_path_pair)
        return all_path_triples

    def split_dataset(self, seqs_labels_path_pair):
        train_pairs = []
        val_pairs = []
        test_pairs = []

        for i in range(10):
            if i < 8:
                train_pairs.extend(seqs_labels_path_pair[i])
            elif i < 9:
                val_pairs.extend(seqs_labels_path_pair[i])
            else:
                test_pairs.extend(seqs_labels_path_pair[i])
        # print(train_pairs, val_pairs, test_pairs)
        return train_pairs, val_pairs, test_pairs



def prepare_datasets_isruc_s3(
    data_dir,
    loading_metafeatures=False,
    scale=None
):
    loaddataset = LoadDataset(
        data_dir=data_dir,
        loading_metafeatures=loading_metafeatures,
        scale=scale
    )
    return loaddataset.get_datasets()