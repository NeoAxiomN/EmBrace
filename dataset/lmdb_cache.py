import os
import pickle
import lmdb


LMDB_OPEN_OPTIONS = {
    "readonly": True,
    "lock": False,
    "readahead": True,
    "meminit": False,
}


class LmdbResourceCache:
    _keys_cache = {}
    _env_cache = {}

    @classmethod
    def get_keys_by_mode(cls, data_dir):
        normalized_path = os.path.abspath(data_dir)
        keys_by_mode = cls._keys_cache.get(normalized_path)
        if keys_by_mode is None:
            with lmdb.open(normalized_path, **LMDB_OPEN_OPTIONS) as db:
                with db.begin() as txn:
                    keys_by_mode = pickle.loads(txn.get(b"__keys__"))
            cls._keys_cache[normalized_path] = keys_by_mode
        return keys_by_mode

    @classmethod
    def get_env(cls, path):
        normalized_path = os.path.abspath(path)
        cache_key = (os.getpid(), normalized_path)
        env = cls._env_cache.get(cache_key)
        if env is None:
            env = lmdb.open(normalized_path, **LMDB_OPEN_OPTIONS)
            cls._env_cache[cache_key] = env
        return env
