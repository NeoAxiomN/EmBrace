from .lmdb_dataset import prepare_lmdb_datasets


def prepare_datasets_physio(
    data_dir,
    loading_metafeatures=False,
    scale=None,
):
    return prepare_lmdb_datasets(
        data_dir=data_dir,
        loading_metafeatures=loading_metafeatures,
        scale=scale,
    )
