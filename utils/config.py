"""
加载 YAML 配置，支持默认配置与覆盖。
"""

import os
from typing import Any, Dict, Optional

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def merge_dict(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def get_config(
    default_path: str = "configs/default.yaml",
    override_path: Optional[str] = None,
    overrides: Optional[Dict] = None,
) -> Dict[str, Any]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_full = os.path.join(root, default_path)
    base = load_yaml(default_full)
    if override_path:
        override_full = os.path.join(root, override_path)
        base = merge_dict(base, load_yaml(override_full))
    if overrides:
        base = merge_dict(base, overrides)
    return base
