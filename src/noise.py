from __future__ import annotations

import numpy as np
import cv2

def _clip(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def add_glare(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    if severity <= 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()

    out = out * (1.0 - 0.35 * severity) + 0.35 * severity

    cy = rng.integers(0, h // 2 + 1)
    cx = rng.integers(0, w)
    radius = int((0.15 + 0.45 * severity) * max(h, w))
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    blob = np.exp(-(dist ** 2) / (2 * (radius / 2.0 + 1e-6) ** 2))
    blob = (blob * severity)[..., None]
    out = out * (1.0 - blob) + blob
    return _clip(out)

def add_rain(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Diagonal rain streaks + slight blur + contrast reduction."""
    if severity <= 0:
        return img
    h, w = img.shape[:2]
    layer = np.zeros((h, w), dtype=np.float32)

    n_drops = int(severity * 0.012 * h * w)
    length = int(8 + 18 * severity)
    angle = rng.uniform(-0.35, 0.35)
    dx = int(np.sin(angle) * length)
    for _ in range(n_drops):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        x2 = np.clip(x + dx, 0, w - 1)
        y2 = np.clip(y + length, 0, h - 1)
        cv2.line(layer, (x, y), (int(x2), int(y2)), color=1.0, thickness=1)

    k = max(1, int(3 + 4 * severity) | 1)
    layer = cv2.GaussianBlur(layer, (k, k), 0)

    out = img.copy()
    out = out * (1.0 - 0.5 * severity) + 0.5 * severity * img.mean()
    out = out + layer[..., None] * (0.6 * severity)
    out = cv2.GaussianBlur(out, (k, k), 0)
    return _clip(out)

def add_occlusion(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    if severity <= 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()

    frac = 0.2 + 0.6 * severity
    pw = int(frac * w)
    ph = int(frac * h)
    x0 = rng.integers(0, max(1, w - pw))
    y0 = rng.integers(0, max(1, h - ph))

    patch = out[y0:y0 + ph, x0:x0 + pw]
    smear = cv2.GaussianBlur(patch, (0, 0), sigmaX=8) * (1.0 - 0.85 * severity)
    out[y0:y0 + ph, x0:x0 + pw] = smear
    return _clip(out)


def add_frame_drop(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    if severity <= 0:
        return img
    return _clip(img * (1.0 - severity))

_TRANSFORMS = {
    "glare": add_glare,
    "rain": add_rain,
    "occlusion": add_occlusion,
    "frame_drop": add_frame_drop,
}


def apply_noise(
    img: np.ndarray,
    condition: str,
    severity: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if condition in ("clean", "none") or severity <= 0:
        return img.astype(np.float32)
    if condition == "mixed":
        condition = rng.choice(list(_TRANSFORMS.keys()))
    fn = _TRANSFORMS.get(condition)
    if fn is None:
        raise ValueError(f"Unknown noise condition: {condition!r}")
    return fn(img, severity, rng)


def corrupt_sequence(
    frames: np.ndarray,
    condition: str,
    severity: float,
    corrupt_frac: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    T = frames.shape[0]
    out = frames.copy()
    mask = np.zeros(T, dtype=np.float32)
    if condition in ("clean", "none") or severity <= 0:
        return out, mask
    n_corrupt = max(1, int(round(corrupt_frac * T)))
    idx = rng.choice(T, size=n_corrupt, replace=False)
    for t in idx:
        out[t] = apply_noise(out[t], condition, severity, rng)
        mask[t] = 1.0
    return out, mask
