from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .noise import corrupt_sequence

@dataclass
class DataConfig:
    root: str
    image_size: tuple = (66, 200)
    seq_len: int = 5
    stride: int = 1
    val_split: float = 0.2
    use_side_cameras: bool = True
    steering_correction: float = 0.2
    batch_size: int = 32
    num_workers: int = 2
    balance_steering: bool = False
    zero_threshold: float = 0.05
    zero_keep_frac: float = 0.3

def _read_log(root: str) -> pd.DataFrame:
    csv_path = os.path.join(root, "driving_log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No driving_log.csv in {root!r}. Point data.root at a Udacity-format folder "
            f"or run with --synthetic."
        )
    cols = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
    # tolerate logs with or without a header row
    df = pd.read_csv(csv_path, header=None, names=cols)
    if str(df.iloc[0]["steering"]).strip().lower() in ("steering", "steer"):
        df = df.iloc[1:].reset_index(drop=True)
    df["steering"] = df["steering"].astype(float)
    return df

_IMAGE_CACHE: dict = {}
def _load_image(path: str, size: tuple, use_cache: bool = True) -> np.ndarray:
    key = (path, size)
    if use_cache:
        cached = _IMAGE_CACHE.get(key)
        if cached is not None:
            return cached
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h = img.shape[0]
    img = img[int(0.35 * h):int(0.9 * h), :, :]
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    out = (img.astype(np.float32) / 255.0)
    if use_cache:
        _IMAGE_CACHE[key] = out
    return out

class DrivingSequenceDataset(Dataset):

    def __init__(
        self,
        cfg: DataConfig,
        rows: pd.DataFrame,
        train: bool = True,
        noise_cfg: Optional[dict] = None,
        eval_noise: Optional[dict] = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.rows = rows.reset_index(drop=True)
        self.train = train
        self.noise_cfg = noise_cfg or {"enabled": False}
        self.eval_noise = eval_noise
        self.size = tuple(cfg.image_size)
        self.seq_len = cfg.seq_len
        self.rng = np.random.default_rng(seed)

        n = len(self.rows)
        last_start = n - self.seq_len
        self.starts = list(range(0, max(0, last_start + 1), cfg.stride))
        if not self.starts and n >= 1:
            self.starts = [0]
        if getattr(self.cfg, "balance_steering", False):
            before = len(self.starts)
            self.starts = self._balance_starts(self.starts)
            if self.train:
                print(f"[dataset] steering balance: kept {len(self.starts)}/{before} "
                      f"windows (dropped {before - len(self.starts)} near-straight; "
                      f"thr={self.cfg.zero_threshold}, keep_frac={self.cfg.zero_keep_frac})",
                      flush=True)

    def _balance_starts(self, starts):
        """Keep all 'turn' windows; keep only `zero_keep_frac` of 'straight' windows."""
        thr = float(self.cfg.zero_threshold)
        keep_frac = float(self.cfg.zero_keep_frac)
        n = len(self.rows)
        kept = []
        for s in starts:
            tgt_idx = min(s + self.seq_len - 1, n - 1)
            steer = abs(float(self.rows.iloc[tgt_idx]["steering"]))
            if steer < thr and self.rng.random() > keep_frac:
                continue
            kept.append(s)
        return kept if kept else starts

    def __len__(self) -> int:
        return len(self.starts)

    def _sample_camera(self, row) -> tuple[str, float]:
        steer = float(row["steering"])
        if self.train and self.cfg.use_side_cameras and self.rng.random() < 0.5:
            if self.rng.random() < 0.5:
                return str(row["left"]).strip(), steer + self.cfg.steering_correction
            return str(row["right"]).strip(), steer - self.cfg.steering_correction
        return str(row["center"]).strip(), steer

    def _resolve(self, img_ref: str) -> str:
        img_ref = img_ref.replace("\\", "/")
        base = os.path.basename(img_ref)
        cand = os.path.join(self.cfg.root, "IMG", base)
        if os.path.exists(cand):
            return cand
        return os.path.join(self.cfg.root, img_ref)

    def __getitem__(self, i: int):
        start = self.starts[i]
        idxs = [min(start + k, len(self.rows) - 1) for k in range(self.seq_len)]

        frames = np.empty((self.seq_len, self.size[0], self.size[1], 3), dtype=np.float32)
        steer_last = 0.0
        cam_choice = self.rng.random()
        for k, idx in enumerate(idxs):
            row = self.rows.iloc[idx]
            steer = float(row["steering"])
            ref = str(row["center"]).strip()
            if self.train and self.cfg.use_side_cameras and cam_choice < 0.5:
                if cam_choice < 0.25:
                    ref = str(row["left"]).strip()
                    steer = steer + self.cfg.steering_correction
                else:
                    ref = str(row["right"]).strip()
                    steer = steer - self.cfg.steering_correction
            frames[k] = _load_image(self._resolve(ref), self.size)
            steer_last = steer

        mask = np.zeros(self.seq_len, dtype=np.float32)

        if self.train and self.noise_cfg.get("enabled", False):
            if self.rng.random() < self.noise_cfg.get("p", 0.5):
                sev = self.rng.uniform(0, self.noise_cfg.get("max_severity", 0.6))
                cond = self.rng.choice(["glare", "rain", "occlusion", "frame_drop"])
                frames, mask = corrupt_sequence(
                    frames, cond, sev, corrupt_frac=0.5, rng=self.rng
                )
        if self.eval_noise is not None:
            frames, mask = corrupt_sequence(
                frames,
                self.eval_noise["condition"],
                self.eval_noise["severity"],
                self.eval_noise.get("corrupt_frac", 0.6),
                rng=self.rng,
            )
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        target_t = torch.tensor([steer_last], dtype=torch.float32)
        mask_t = torch.from_numpy(mask)
        return frames_t, target_t, mask_t

def build_datasets(cfg: DataConfig, noise_cfg: dict, seed: int = 42):
    """Chronological train/val split (no leakage across the temporal boundary)."""
    df = _read_log(cfg.root)
    n = len(df)
    split = int(n * (1 - cfg.val_split))
    train_rows = df.iloc[:split]
    val_rows = df.iloc[split:]
    train_ds = DrivingSequenceDataset(cfg, train_rows, train=True, noise_cfg=noise_cfg, seed=seed)
    val_ds = DrivingSequenceDataset(cfg, val_rows, train=False, seed=seed + 1)
    return train_ds, val_ds


def make_loaders(cfg: DataConfig, noise_cfg: dict, seed: int = 42):
    train_ds, val_ds = build_datasets(cfg, noise_cfg, seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def make_eval_loader(cfg: DataConfig, condition: str, severity: float,
                     corrupt_frac: float, seed: int = 123):
    """Validation split corrupted with a single fixed noise condition."""
    df = _read_log(cfg.root)
    split = int(len(df) * (1 - cfg.val_split))
    val_rows = df.iloc[split:]
    ds = DrivingSequenceDataset(
        cfg, val_rows, train=False,
        eval_noise={"condition": condition, "severity": severity, "corrupt_frac": corrupt_frac},
        seed=seed,
    )
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

def make_synthetic_dataset(root: str, n_frames: int = 400, size=(160, 320), seed: int = 0):
    """
    Generate a tiny procedural 'driving' dataset in Udacity format for smoke-testing.

    The 'road' is a vertical lane whose horizontal position follows a smooth sinusoid; the
    steering label is proportional to the lane's offset from center. Side cameras are simple
    horizontal shifts. This is enough to verify the full train/eval/analyze pipeline runs.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "IMG"), exist_ok=True)
    h, w = size
    t = np.linspace(0, 8 * np.pi, n_frames)
    lane_center = 0.5 + 0.25 * np.sin(t) + 0.03 * rng.standard_normal(n_frames)
    rows = []

    for i in range(n_frames):
        lc = float(np.clip(lane_center[i], 0.1, 0.9))
        steering = float(np.clip((lc - 0.5) * 2.0, -1, 1))
        cams = {}
        for cam, shift in (("center", 0.0), ("left", -0.06), ("right", 0.06)):
            img = np.full((h, w, 3), 90, dtype=np.uint8)
            img[: h // 3] = (135, 180, 230)
            cx = int((lc + shift) * w)
            cv2.line(img, (cx, h), (int(w / 2), h // 3), (255, 255, 255), 4)
            cv2.line(img, (cx - 80, h), (int(w / 2) - 30, h // 3), (230, 230, 0), 2)
            fname = f"{cam}_{i:05d}.jpg"
            cv2.imwrite(os.path.join(root, "IMG", fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cams[cam] = f"IMG/{fname}"
        rows.append([cams["center"], cams["left"], cams["right"], steering, 0.5, 0, 20.0])

    df = pd.DataFrame(rows, columns=["center", "left", "right",
                                     "steering", "throttle", "reverse", "speed"])
    df.to_csv(os.path.join(root, "driving_log.csv"), index=False)
    return root
