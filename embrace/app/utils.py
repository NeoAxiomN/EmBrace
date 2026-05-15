import ast
import copy
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
FALSE_TOKENS = {"0", "false", "no", "n", "off"}
NULL_TOKENS = {"", "null"}


class ConfigNode(SimpleNamespace):
    def clone(self):
        return dict_to_config(config_to_dict(self))

    def to_dict(self):
        return config_to_dict(self)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Boolean value is required.")

    normalized = str(value).strip().lower()
    if normalized in TRUE_TOKENS:
        return True
    if normalized in FALSE_TOKENS:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_scalar_string(value):
    if not isinstance(value, str):
        return value

    text = value.strip()
    lowered = text.lower()
    if lowered in NULL_TOKENS:
        return None
    if lowered in TRUE_TOKENS:
        return True
    if lowered in FALSE_TOKENS:
        return False

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def normalize_scalar_strings(value):
    if isinstance(value, dict):
        return {key: normalize_scalar_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_scalar_strings(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_scalar_strings(item) for item in value]
    return parse_scalar_string(value)


def load_yaml_file(path):
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return normalize_scalar_strings(payload)


def dump_yaml_text(payload):
    return yaml.safe_dump(
        config_to_dict(payload),
        allow_unicode=False,
        sort_keys=False,
    )


def config_to_dict(value):
    if isinstance(value, dict):
        return {key: config_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [config_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [config_to_dict(item) for item in value]
    if isinstance(value, SimpleNamespace):
        return {key: config_to_dict(item) for key, item in vars(value).items()}
    return copy.deepcopy(value)


def dict_to_config(value):
    if isinstance(value, dict):
        return ConfigNode(**{key: dict_to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [dict_to_config(item) for item in value]
    return copy.deepcopy(value)


def clone_config(cfg):
    return dict_to_config(config_to_dict(cfg))


def cfg_get(cfg, dotted_key, default=None):
    current = cfg
    for part in dotted_key.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if isinstance(current, SimpleNamespace):
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
            continue
        return default

    if current == "???":
        return default
    return current


def merge_dicts(*payloads):
    merged = {}
    for payload in payloads:
        if payload is None:
            continue
        merged = _merge_two_dicts(merged, config_to_dict(payload))
    return merged


def assign_nested(mapping, value, *keys):
    if value is None:
        return

    current = mapping
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def parse_yaml_like_token(text):
    try:
        return normalize_scalar_strings(yaml.safe_load(text))
    except yaml.YAMLError:
        return text


def normalize_string_list(values):
    if values is None:
        return None

    normalized = []
    for raw_value in values:
        if not isinstance(raw_value, str):
            normalized.append(str(raw_value))
            continue

        text = raw_value.strip()
        if not text:
            continue

        parsed_value = parse_yaml_like_token(text)
        if isinstance(parsed_value, list):
            normalized.extend(str(item).strip() for item in parsed_value if str(item).strip())
            continue

        if isinstance(parsed_value, str) and "," in parsed_value:
            normalized.extend(part.strip() for part in parsed_value.split(",") if part.strip())
            continue

        normalized.append(str(parsed_value).strip())

    return [value for value in normalized if value]


def normalize_int_list(values):
    normalized = normalize_string_list(values)
    if normalized is None:
        return None
    return [int(item) for item in normalized]


def seed_dir_sort_key(seed_name):
    suffix = seed_name.split("seed_", 1)[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, suffix)


def is_numeric_scalar(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@contextmanager
def locked_file(lock_path):
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _merge_two_dicts(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_two_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
