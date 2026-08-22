"""YAML config loading with optional override."""
import copy

import yaml


def load_config(base_path: str, override_path: str | None = None) -> dict:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if override_path:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        cfg = _deep_merge(cfg, override)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
