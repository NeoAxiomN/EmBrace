from .lmdb_cache import LmdbResourceCache

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class LmdbDataset(Dataset):
    def __init__(
        self,
        data_dir,
        loading_metafeatures=False,
        mode="train",
        scale=None,
        label_dtype="long",
        flatten_labels=False,
    ):
        self.data_dir = str(data_dir)
        self.loading_metafeatures = loading_metafeatures
        self.meta_dir = os.path.join(self.data_dir, "meta_features")
        self.scale = scale
        self.label_dtype = label_dtype
        self.flatten_labels = flatten_labels

        self.keys = LmdbResourceCache.get_keys_by_mode(self.data_dir)[mode]
        self.db_data = None
        self.db_meta = None

    def _init_db(self):
        if self.db_data is None:
            self.db_data = LmdbResourceCache.get_env(self.data_dir)
            self.db_meta = (
                LmdbResourceCache.get_env(self.meta_dir)
                if self.loading_metafeatures
                else None
            )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_db()
        key = self.keys[idx].encode()

        with self.db_data.begin() as txn:
            pair = pickle.loads(txn.get(key))

        x, y = pair["sample"], pair["label"]
        if self.scale:
            x = x * self.scale

        meta = None
        if self.loading_metafeatures:
            with self.db_meta.begin() as txn_meta:
                val = txn_meta.get(key)
                meta = pickle.loads(val) if val else None

        return x, y, meta

    def collate_fn(self, batch):
        samples, labels, metas = zip(*batch)

        x = torch.from_numpy(np.array(samples)).float()
        y = torch.from_numpy(np.array(labels))
        if self.label_dtype == "float":
            y = y.float()
        else:
            y = y.long()
        if self.flatten_labels:
            y = y.view(-1)

        m = torch.from_numpy(np.array(metas)).float() if metas[0] is not None else None
        if m is not None:
            return x, y, m
        return x, y


def prepare_lmdb_datasets(
    data_dir,
    loading_metafeatures=False,
    scale=None,
    *,
    label_dtype="long",
    flatten_labels=False,
):
    return {
        mode: LmdbDataset(
            data_dir=data_dir,
            loading_metafeatures=loading_metafeatures and mode == "train",
            mode=mode,
            scale=scale,
            label_dtype=label_dtype,
            flatten_labels=flatten_labels,
        )
        for mode in ["train", "val", "test"]
    }
