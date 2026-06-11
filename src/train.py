from __future__ import annotations

import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass

from .utils import (load_config, set_seed, get_device, ensure_dir, save_json,
                    count_params, mse, mae)
from .dataset import DataConfig, make_loaders, make_synthetic_dataset
from .models import build_model, all_model_names


def _data_config(cfg) -> DataConfig:
    d = cfg["data"]
    return DataConfig(
        root=d["root"],
        image_size=tuple(d["image_size"]),
        seq_len=int(d["seq_len"]),
        stride=int(d["stride"]),
        val_split=float(d["val_split"]),
        use_side_cameras=bool(d["use_side_cameras"]),
        steering_correction=float(d["steering_correction"]),
        batch_size=int(d["batch_size"]),
        num_workers=int(d["num_workers"]),
        balance_steering=bool(d.get("balance_steering", False)),
        zero_threshold=float(d.get("zero_threshold", 0.05)),
        zero_keep_frac=float(d.get("zero_keep_frac", 0.3)),
    )


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    preds, trues = [], []
    for frames, target, _mask in loader:
        frames = frames.to(device)
        out = model(frames).cpu().numpy().ravel()
        preds.append(out)
        trues.append(target.numpy().ravel())
    preds = np.concatenate(preds) if preds else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    return {"mse": mse(preds, trues), "mae": mae(preds, trues), "n": int(len(preds))}


def train_one(model_name: str, cfg: dict, device, out_dir: str) -> dict:
    dcfg = _data_config(cfg)
    noise_cfg = cfg["data"]["train_noise"]
    train_loader, val_loader = make_loaders(dcfg, noise_cfg, seed=cfg["seed"])

    model = build_model(model_name, cfg["model"]).to(device)

    sample = next(iter(train_loader))[0][:2].to(device)
    model(sample)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.MSELoss()
    grad_clip = cfg["train"]["grad_clip"]
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["early_stop_patience"]

    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
    history = []
    best_val = float("inf")
    bad_epochs = 0

    print(f"\n=== Training {model_name}  ({count_params(model):,} params) ===", flush=True)
    n_total_batches = len(train_loader)
    print(f"train batches/epoch: {n_total_batches} | val batches: {len(val_loader)}", flush=True)
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        t_epoch = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for frames, target, _mask in pbar:
            frames, target = frames.to(device), target.to(device)
            opt.zero_grad()
            pred = model(frames)
            loss = loss_fn(pred, target)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if n_batches == 1 or n_batches % 20 == 0:
                print(f"  [{model_name}] epoch {epoch}/{epochs} | "
                      f"batch {n_batches}/{n_total_batches} | "
                      f"loss {loss.item():.4f} | "
                      f"{time.time() - t_epoch:.1f}s", flush=True)

        train_loss = running / max(1, n_batches)
        val = evaluate_loader(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_mse": val["mse"], "val_mae": val["mae"]})
        print(f"epoch {epoch:3d}/{epochs} | train {train_loss:.4f} | "
              f"val MSE {val['mse']:.4f} | val MAE {val['mae']:.4f} | "
              f"{time.time() - t_epoch:.1f}s/epoch | {time.time() - t_start:.0f}s total",
              flush=True)

        if val["mse"] < best_val - 1e-6:
            best_val = val["mse"]
            bad_epochs = 0
            torch.save({"model_name": model_name,
                        "state_dict": model.state_dict(),
                        "config": cfg,
                        "val_mse": best_val}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best val MSE {best_val:.4f})", flush=True)
                break

    summary = {"model": model_name, "best_val_mse": best_val,
               "params": count_params(model), "history": history,
               "checkpoint": ckpt_path}
    save_json(summary, os.path.join(out_dir, "metrics", f"train_{model_name}.json"))
    print(f"saved checkpoint -> {ckpt_path}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default=None,
                    help="pilotnet | cnn_lstm | cnn_lstm_attention | all")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="generate & use a tiny synthetic dataset (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.model is not None:
        cfg["model"]["name"] = args.model

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    print(f"device: {device}")

    if args.synthetic:
        root = os.path.join(cfg["train"]["out_dir"], "synthetic_data")
        print(f"generating synthetic dataset at {root} ...")
        make_synthetic_dataset(root, n_frames=300, size=(160, 320), seed=cfg["seed"])
        cfg["data"]["root"] = root

    out_dir = ensure_dir(cfg["train"]["out_dir"])
    names = all_model_names() if cfg["model"]["name"] == "all" else [cfg["model"]["name"]]

    t0 = time.time()
    for name in names:
        train_one(name, cfg, device, out_dir)
    print(f"\nall done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
