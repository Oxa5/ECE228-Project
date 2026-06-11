from __future__ import annotations

import os
import json
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(v) for v in d]
    return d

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
