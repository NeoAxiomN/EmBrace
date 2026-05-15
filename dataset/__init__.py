from importlib import import_module

__all__ = ["create_datasets"]

_DATASET_LOADERS = {
    "attention": (".dataset_attention", "prepare_datasets_attention"),
    "bciciv2a": (".dataset_bciciv2a", "prepare_datasets_bciciv2a"),
    "chb-mit": (".dataset_chb", "prepare_datasets_chb"),
    "faced": (".dataset_faced", "prepare_datasets_faced"),
    "isruc-s1": (".dataset_isruc_s1", "prepare_datasets_isruc_s1"),
    "isruc-s3": (".dataset_isruc_s3", "prepare_datasets_isruc_s3"),
    "mumtaz": (".dataset_mumtaz", "prepare_datasets_mumtaz"),
    "physio": (".dataset_physio", "prepare_datasets_physio"),
    "seedv": (".dataset_seedv", "prepare_datasets_seedv"),
    "shu": (".dataset_shu", "prepare_datasets_shu"),
    "speech": (".dataset_speech", "prepare_datasets_speech"),
    "stress": (".dataset_stress", "prepare_datasets_stress"),
}


def _get_dataset_loader(dataset_name):
    if dataset_name not in _DATASET_LOADERS:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented.")

    module_name, func_name = _DATASET_LOADERS[dataset_name]
    module = import_module(module_name, package=__name__)
    return getattr(module, func_name)


def create_datasets(
    dataset, 
    data_dir,
    loading_metafeatures=False,
    scale=None
):
    args = {
        "data_dir": data_dir,
        "loading_metafeatures": loading_metafeatures,
        "scale": scale
    }

    dataset_loader = _get_dataset_loader(dataset)
    return dataset_loader(**args)
