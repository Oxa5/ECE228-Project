from __future__ import annotations

import os
import argparse

import numpy as np
import torch

from .utils import load_config, set_seed, get_device, ensure_dir, save_json, mse, mae
from .dataset import DataConfig, make_eval_loader, make_synthetic_dataset
from .models import build_model, all_model_names


def _data_config(cfg) -> DataConfig:
    d = cfg["data"]
    return DataConfig(
        root=d["root"], image_size=tuple(d["image_size"]),
        seq_len=int(d["seq_len"]), stride=int(d["stride"]),
        val_split=float(d["val_split"]), use_side_cameras=False,
        steering_correction=float(d["steering_correction"]),
        batch_size=int(d["batch_size"]), num_workers=int(d["num_workers"]),
        balance_steering=bool(d.get("balance_steering", False)),
        zero_threshold=float(d.get("zero_threshold", 0.05)),
        zero_keep_frac=float(d.get("zero_keep_frac", 0.3)),
    )


@torch.no_grad()
def eval_condition(model, loader, device, want_attention: bool):
    model.eval()
    preds, trues = [], []
    attn_corrupt, attn_clean = [], []
    for frames, target, mask in loader:
        frames = frames.to(device)
        out = model(frames).cpu().numpy().ravel()
        preds.append(out)
        trues.append(target.numpy().ravel())

        if want_attention and getattr(model, "last_attention", None) is not None:
            w = model.last_attention.cpu().numpy()
            m = mask.numpy()
            if m.sum() > 0:
                attn_corrupt.append(w[m > 0.5])
            if (m < 0.5).sum() > 0:
                attn_clean.append(w[m < 0.5])

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    res = {"mse": mse(preds, trues), "mae": mae(preds, trues), "n": int(len(preds))}
    if want_attention:
        res["attn_corrupt_mean"] = float(np.concatenate(attn_corrupt).mean()) if attn_corrupt else None
        res["attn_clean_mean"] = float(np.concatenate(attn_clean).mean()) if attn_clean else None
    return res


def evaluate_model(model_name: str, cfg: dict, device, out_dir: str) -> dict:
    ckpt_path = os.path.join(out_dir, "checkpoints", f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint for {model_name} at {ckpt_path}. Train it first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name, cfg["model"]).to(device)
    dcfg = _data_config(cfg)
    probe = make_eval_loader(dcfg, "clean", 0.0, 0.0, seed=cfg["seed"])
    sample = next(iter(probe))[0][:2].to(device)
    model(sample)
    model.load_state_dict(ckpt["state_dict"])

    want_attn = model_name in ("cnn_lstm_attention", "cnn_lstm_featattn")
    conditions = cfg["data"]["eval_noise"]["conditions"]
    severities = cfg["data"]["eval_noise"]["severities"]
    corrupt_frac = cfg["data"]["eval_noise"]["corrupt_frac"]

    results = {"model": model_name, "grid": []}
    print(f"\n=== Evaluating {model_name} ===")
    for cond in conditions:
        sevs = [0.0] if cond == "clean" else severities
        for sev in sevs:
            loader = make_eval_loader(dcfg, cond, sev, corrupt_frac, seed=cfg["seed"])
            r = eval_condition(model, loader, device, want_attn)
            r.update({"condition": cond, "severity": sev})
            results["grid"].append(r)
            line = f"{cond:10s} sev={sev:.2f} | MSE {r['mse']:.4f} | MAE {r['mae']:.4f}"
            if want_attn and r.get("attn_corrupt_mean") is not None:
                line += (f" | attn(corrupt)={r['attn_corrupt_mean']:.3f} "
                         f"attn(clean)={r['attn_clean_mean']:.3f}")
            print(line)

    save_json(results, os.path.join(out_dir, "metrics", f"eval_{model_name}.json"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="all")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    if args.synthetic:
        root = os.path.join(cfg["train"]["out_dir"], "synthetic_data")
        if not os.path.exists(os.path.join(root, "driving_log.csv")):
            make_synthetic_dataset(root, n_frames=300, size=(160, 320), seed=cfg["seed"])
        cfg["data"]["root"] = root

    out_dir = ensure_dir(cfg["train"]["out_dir"])
    names = all_model_names() if args.model == "all" else [args.model]

    all_results = {}
    for name in names:
        all_results[name] = evaluate_model(name, cfg, device, out_dir)
    save_json(all_results, os.path.join(out_dir, "metrics", "eval_all.json"))
    print(f"\nsaved combined results -> {os.path.join(out_dir, 'metrics', 'eval_all.json')}")


if __name__ == "__main__":
    main()
