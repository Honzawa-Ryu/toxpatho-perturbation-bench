"""One-off analysis comparing the stain-normalized Exp0004 run against the
unnormalized baseline: per-model overall-accuracy shift, and which
perturbation-kind robustness ratios best predict that shift.

Not a pipeline experiment -- reads two already-completed Exp0004 output
dirs and writes a figure + prints a correlation table. Rerun manually:

    .venv/bin/python notebooks/compare_stainnorm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "outputs/0004_20260821_eval_retrieval_invariance/256px_mpp0.5_n1000x20"
NORM_DIR = PROJECT_ROOT / "outputs/0004_20260821_eval_retrieval_invariance/256px_mpp0.5_n1000x20_orig"
OUT_DIR = NORM_DIR / "figures"


def load_overall_comparison() -> pd.DataFrame:
    base = pd.read_parquet(BASE_DIR / "metrics_by_model_perturbation.parquet")
    norm = pd.read_parquet(NORM_DIR / "metrics_by_model_perturbation.parquet")
    ov_base = base[base["kind"] == "ALL"][["model", "top1_acc"]].rename(columns={"top1_acc": "base_overall"})
    ov_norm = norm[norm["kind"] == "ALL"][["model", "top1_acc"]].rename(columns={"top1_acc": "norm_overall"})
    merged = ov_base.merge(ov_norm, on="model")
    merged["norm_delta"] = merged["norm_overall"] - merged["base_overall"]
    merged["norm_ratio"] = merged["norm_overall"] / merged["base_overall"]
    return merged, base


def plot_delta_dumbbell(merged: pd.DataFrame, out_path: Path) -> None:
    """Per-model base_overall -> norm_overall as connected points, sorted by delta."""
    df = merged.sort_values("norm_delta")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(df) + 1.5))
    y = range(len(df))
    ax.hlines(y, df["norm_overall"], df["base_overall"], color="lightgray", linewidth=2, zorder=1)
    ax.scatter(df["base_overall"], y, color="tab:blue", label="baseline (no norm)", zorder=2)
    ax.scatter(df["norm_overall"], y, color="tab:red", label="stain-normalized", zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"], fontsize=8)
    ax.set_xlabel("overall top1_acc")
    ax.set_title("Effect of stain normalization on overall accuracy (sorted by magnitude)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_heatmap(level: int, out_path: Path) -> None:
    """Model x kind heatmap of norm_acc - base_acc at a given level, plus a mean column.

    Complements plot_delta_dumbbell (overall only): this shows which specific
    perturbation kinds drive each model's normalization-induced drop.
    """
    base = pd.read_parquet(BASE_DIR / "metrics_by_model_perturbation.parquet")
    norm = pd.read_parquet(NORM_DIR / "metrics_by_model_perturbation.parquet")
    b = base[base["level"] == level].set_index(["model", "kind"])["top1_acc"]
    n = norm[norm["level"] == level].set_index(["model", "kind"])["top1_acc"]
    delta = (n - b).dropna().reset_index(name="delta")
    pivot = delta.pivot(index="model", columns="kind", values="delta")
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False)  # least affected (closest to 0) first

    vmax = pivot.abs().to_numpy().max()
    fig, ax = plt.subplots(figsize=(0.55 * len(pivot.columns) + 3, 0.3 * len(pivot.index) + 2))
    values = pivot.to_numpy()
    im = ax.imshow(values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.axvline(len(pivot.columns) - 1.5, color="black", linewidth=1)  # separator before the mean column
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"top1_acc change from stain normalization at level {level} (norm - base)")
    fig.colorbar(im, ax=ax, label="Δ top1_acc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ratio_rank_heatmap(level: int, out_path: Path) -> None:
    """Model x kind heatmap: rank (1=best) by % of baseline top1_acc retained.

    A raw-percentage version of this heatmap is dominated by a handful of
    genuine but extreme outliers (e.g. gigapath-flash retains 1499% of its
    jpeg_compression base_acc -- base_acc itself is tiny but still above any
    reasonable "too close to zero" cutoff, so it's real, not noise). Any
    single fixed color range either clips those into illegibility or
    crushes the contrast among everything else. Masking those cells instead
    leaves the heatmap full of blank "n/a" gaps.

    Ranking side-steps both problems: no cell needs excluding (rank is
    defined however extreme the underlying ratio is), and no color-range
    choice is needed (rank is always 1..n_models). Same convention as
    plot_rank_heatmap in experiment.py: RdYlGn_r, 1=best, mean column added.
    """
    base = pd.read_parquet(BASE_DIR / "metrics_by_model_perturbation.parquet")
    norm = pd.read_parquet(NORM_DIR / "metrics_by_model_perturbation.parquet")
    b = base[base["level"] == level].set_index(["model", "kind"])["top1_acc"]
    n = norm[norm["level"] == level].set_index(["model", "kind"])["top1_acc"]
    df = pd.DataFrame({"base": b, "norm": n}).dropna()
    df["pct_retained"] = 100 * df["norm"] / df["base"]
    pivot = df["pct_retained"].unstack("kind")
    rank = pivot.rank(axis=0, ascending=False)  # rank 1 = best (highest % retained) within each kind column
    n_models = len(rank.index)
    rank["mean"] = rank.mean(axis=1)
    rank = rank.sort_values("mean")  # best average rank first

    fig, ax = plt.subplots(figsize=(0.55 * len(rank.columns) + 3, 0.3 * len(rank.index) + 2))
    values = rank.to_numpy()
    im = ax.imshow(values, cmap="RdYlGn_r", vmin=1, vmax=n_models, aspect="auto")
    ax.axvline(len(rank.columns) - 1.5, color="black", linewidth=1)  # separator before the mean column
    norm_values = (values - 1) / (n_models - 1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if norm_values[i, j] < 0.2 or norm_values[i, j] > 0.8 else "black"
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=7)
    ax.set_xticks(range(len(rank.columns)))
    ax.set_xticklabels(rank.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rank.index)))
    ax.set_yticklabels(rank.index, fontsize=8)
    ax.set_title(f"Rank by % of baseline top1_acc retained after normalization, level {level} (1=best of {n_models})")
    fig.colorbar(im, ax=ax, label="rank (1=best)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_kind_correlations(merged: pd.DataFrame, base: pd.DataFrame) -> None:
    kinds = [k for k in base["kind"].unique() if k != "ALL"]
    for kind in kinds:
        l1 = base[(base["kind"] == kind) & (base["level"] == 1)][["model", "top1_acc"]].rename(columns={"top1_acc": "l1"})
        l3 = base[(base["kind"] == kind) & (base["level"] == 3)][["model", "top1_acc"]].rename(columns={"top1_acc": "l3"})
        ratio = l1.merge(l3, on="model")
        ratio[f"{kind}_ratio"] = ratio["l3"] / ratio["l1"]
        merged.loc[:, f"{kind}_ratio"] = merged["model"].map(ratio.set_index("model")[f"{kind}_ratio"])

    print("=== Spearman correlation with norm_ratio (norm_acc / base_acc) ===")
    results = []
    for col in [f"{k}_ratio" for k in kinds]:
        r, p = spearmanr(merged["norm_ratio"], merged[col])
        results.append((col, r, p))
    results.sort(key=lambda x: -abs(x[1]))
    for col, r, p in results:
        print(f"{col:25s} r={r:+.3f}  p={p:.4f}")


if __name__ == "__main__":
    merged, base = load_overall_comparison()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_delta_dumbbell(merged, OUT_DIR / "stainnorm_delta_dumbbell.png")
    print(f"Wrote {OUT_DIR / 'stainnorm_delta_dumbbell.png'}")
    plot_delta_heatmap(level=3, out_path=OUT_DIR / "stainnorm_delta_heatmap_level3.png")
    print(f"Wrote {OUT_DIR / 'stainnorm_delta_heatmap_level3.png'}")
    plot_ratio_rank_heatmap(level=3, out_path=OUT_DIR / "stainnorm_ratio_heatmap_level3.png")
    print(f"Wrote {OUT_DIR / 'stainnorm_ratio_heatmap_level3.png'}")
    print()
    print_kind_correlations(merged, base)
