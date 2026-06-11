from __future__ import annotations

import os
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import load_config, load_json, ensure_dir


MODEL_LABELS = {
    "pilotnet": "PilotNet (CNN)",
    "cnn_lstm": "CNN+LSTM (baseline)",
    "cnn_lstm_attention": "CNN+LSTM+Attention (output-level)",
    "cnn_lstm_featattn": "CNN+FeatAttn+LSTM (proposed)",
}
MODEL_ORDER = ["pilotnet", "cnn_lstm", "cnn_lstm_attention", "cnn_lstm_featattn"]


def load_all_eval(metrics_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(metrics_dir, "eval_*.json")):
        if os.path.basename(path) == "eval_all.json":
            continue
        data = load_json(path)
        model = data["model"]
        for g in data["grid"]:
            rows.append({
                "model": model,
                "condition": g["condition"],
                "severity": g["severity"],
                "mse": g["mse"],
                "mae": g["mae"],
                "attn_corrupt_mean": g.get("attn_corrupt_mean"),
                "attn_clean_mean": g.get("attn_clean_mean"),
            })
    if not rows:
        raise RuntimeError(f"No eval_*.json found in {metrics_dir}. Run src.evaluate first.")
    return pd.DataFrame(rows)


def plot_mse_vs_severity(df: pd.DataFrame, fig_dir: str):
    conditions = [c for c in df["condition"].unique() if c != "clean"]
    for cond in conditions:
        plt.figure(figsize=(6, 4.2))
        for model in MODEL_ORDER:
            sub = df[(df.model == model) & (df.condition == cond)].sort_values("severity")
            if sub.empty:
                continue
            clean = df[(df.model == model) & (df.condition == "clean")]
            xs = list(sub["severity"])
            ys = list(sub["mse"])
            if not clean.empty and 0.0 not in xs:
                xs = [0.0] + xs
                ys = [float(clean["mse"].iloc[0])] + ys
            plt.plot(xs, ys, marker="o", label=MODEL_LABELS.get(model, model))
        plt.title(f"Steering MSE vs. {cond} severity")
        plt.xlabel("noise severity")
        plt.ylabel("steering MSE (lower is better)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(fig_dir, f"mse_vs_severity_{cond}.png")
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"wrote {out}")


def plot_robustness_gap(df: pd.DataFrame, fig_dir: str):
    """Average MSE increase relative to clean, across all noisy points, per model."""
    gaps = {}
    for model in MODEL_ORDER:
        clean = df[(df.model == model) & (df.condition == "clean")]["mse"]
        noisy = df[(df.model == model) & (df.condition != "clean")]["mse"]
        if clean.empty or noisy.empty:
            continue
        gaps[model] = float(noisy.mean() - clean.mean())
    if not gaps:
        return
    plt.figure(figsize=(6, 4))
    labels = [MODEL_LABELS.get(m, m) for m in gaps]
    vals = [gaps[m] for m in gaps]
    bars = plt.bar(labels, vals, color=["#888", "#3b7", "#36c"][:len(vals)])
    plt.ylabel("mean MSE increase under noise\n(robustness gap, lower is better)")
    plt.title("Robustness gap: how much noise hurts each model")
    plt.xticks(rotation=12, fontsize=8)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out = os.path.join(fig_dir, "robustness_gap.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"wrote {out}")


def plot_attention_suppression(df: pd.DataFrame, fig_dir: str):
    attn_models = [
        m for m in MODEL_ORDER
        if not df[df.model == m].dropna(subset=["attn_corrupt_mean", "attn_clean_mean"]).empty
    ]
    if not attn_models:
        print("no attention data to plot (attention model not evaluated?) — skipping")
        return
    for model in attn_models:
        sub = df[(df.model == model) & (df.condition != "clean")]
        sub = sub.dropna(subset=["attn_corrupt_mean", "attn_clean_mean"])
        if sub.empty:
            continue
        agg = sub.groupby("severity")[["attn_corrupt_mean", "attn_clean_mean"]].mean().reset_index()
        plt.figure(figsize=(6, 4.2))
        plt.plot(agg["severity"], agg["attn_clean_mean"], marker="o", label="clean frames")
        plt.plot(agg["severity"], agg["attn_corrupt_mean"], marker="s", label="corrupted frames")
        plt.title(f"Attention weight: {MODEL_LABELS.get(model, model)}")
        plt.xlabel("noise severity")
        plt.ylabel("mean attention weight")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(fig_dir, f"attention_suppression_{model}.png")
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg["train"]["out_dir"]
    metrics_dir = os.path.join(out_dir, "metrics")
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    df = load_all_eval(metrics_dir)
    df.to_csv(os.path.join(fig_dir, "summary_table.csv"), index=False)
    print(f"wrote {os.path.join(fig_dir, 'summary_table.csv')}")

    print("\n--- mean steering MSE (averaged over all noisy conditions) ---")
    for model in MODEL_ORDER:
        noisy = df[(df.model == model) & (df.condition != "clean")]["mse"]
        clean = df[(df.model == model) & (df.condition == "clean")]["mse"]
        if noisy.empty:
            continue
        print(f"{MODEL_LABELS.get(model, model):35s} "
              f"clean={clean.mean():.4f}  noisy={noisy.mean():.4f}")

    plot_mse_vs_severity(df, fig_dir)
    plot_robustness_gap(df, fig_dir)
    plot_attention_suppression(df, fig_dir)
    print(f"\nAll figures in {fig_dir}")


if __name__ == "__main__":
    main()
