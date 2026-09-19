"""Regenerates Exp 0006's cosine-bearing figures with the centered cosine
alongside the raw one, from notebooks/centering_ablation.py's output.

Exp 0006's figures plot `cos_sim_mean`, which keeps the set mean that
`cka_linear` subtracts -- the ablation showed that single difference is what
makes the two metrics disagree across models (rank agreement with cka_linear
rises from 0.26 to 0.89 once the cosine is centered). Each figure here
therefore carries the raw and centered cosine side by side rather than
replacing one with the other, so the old and new readings stay comparable.

Written as *_centered.png next to Exp 0006's originals, which are left in
place (docs/task_a_implementation_plan.md section 11 cites them).

Rerun manually (reads a CSV, no embeddings):
    .venv/bin/python notebooks/centering_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20"
OUT_DIR = RUN_DIR / "figures"

METRIC_LABELS = {
    "cos_raw": "cos_sim (raw, Exp 0006)",
    "cos_centered": "cos_sim (centered)",
    "cka_linear": "cka_linear",
}


def plot_perturbation_heatmap(pert: pd.DataFrame, level: int, out_path: Path) -> None:
    """{cos_raw, cos_centered, cka_linear} x {base, norm} model x kind heatmaps.

    Same layout and viridis 0-1 scale as Exp 0006's
    perturbation_axis_heatmap_level3.png, with the centered cosine added as a
    middle row so the raw row above and the CKA row below bracket it.
    """
    metrics = ["cos_raw", "cos_centered", "cka_linear"]
    variants = ["base", "norm"]
    sub_all = pert[(pert["level"] == level) & (pert["kind"] != "ALL")]
    n_kinds = sub_all["kind"].nunique()
    n_models = sub_all["model"].nunique()

    fig, axes = plt.subplots(3, 2, figsize=(0.5 * n_kinds * 2 + 4, 0.28 * n_models * 3 + 3))
    for i, metric in enumerate(metrics):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            pivot = (
                sub_all[sub_all["variant"] == variant]
                .pivot(index="model", columns="kind", values=metric)
                .sort_index()
            )
            im = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            ax.set_title(f"{METRIC_LABELS[metric]}, {variant}, level {level}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_normalization_summary(norm: pd.DataFrame, out_path: Path) -> None:
    """Per-model normalization-axis agreement, raw vs centered cosine vs CKA.

    Exp 0006's version plotted two points per model (cos_sim, cka_linear) and
    read the span between them as the model's pairwise/set-level disagreement.
    The centered cosine is added as a third point: where it sits next to
    cka_linear, that span was the set mean rather than a property of the model.
    """
    df = norm.sort_values("cka_linear")
    y = range(len(df))
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(df) + 1.5))
    ax.hlines(y, df["cos_raw"], df["cka_linear"], color="lightgray", linewidth=1, zorder=1)
    ax.scatter(df["cos_raw"], y, color="tab:blue", label=METRIC_LABELS["cos_raw"], zorder=2, s=28)
    ax.scatter(df["cos_centered"], y, color="tab:green", label=METRIC_LABELS["cos_centered"], zorder=3, s=28, marker="D")
    ax.scatter(df["cka_linear"], y, color="tab:red", label=METRIC_LABELS["cka_linear"], zorder=2, s=28)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("similarity (1 = unchanged by normalization)")
    ax.set_title("Normalization axis: base vs. stain-normalized, original patches only")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_normalization_scatter(norm: pd.DataFrame, out_path: Path, annotate: list[str]) -> None:
    """cos vs cka per model, raw (left) and centered (right), same axes.

    Exp 0006 read distance from the diagonal here as representation collapse.
    Centering is the whole of that distance for most models; what stays off
    the diagonal on the right is the part collapse could explain.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    for ax, metric in zip(axes, ["cos_raw", "cos_centered"]):
        ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--", linewidth=1, zorder=1)
        ax.scatter(norm[metric], norm["cka_linear"], color="tab:blue", s=40, zorder=2)
        for model in annotate:
            row = norm[norm["model"] == model]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.annotate(model, (row[metric], row["cka_linear"]), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} vs cka_linear", fontsize=10)
    axes[0].set_ylabel("cka_linear (set-level)")
    fig.suptitle("Normalization axis: does the pairwise/set-level gap survive centering?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    d = pd.read_csv(RUN_DIR / "centering_ablation.csv")
    pert = d[d["axis"] == "perturbation"]
    norm = d[d["axis"] == "normalization"]
    print(f"{d['model'].nunique()} models; {len(pert)} perturbation rows, {len(norm)} normalization rows")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_perturbation_heatmap(pert, level=3, out_path=OUT_DIR / "perturbation_axis_heatmap_level3_centered.png")
    plot_normalization_summary(norm, OUT_DIR / "normalization_axis_summary_centered.png")
    plot_normalization_scatter(
        norm,
        OUT_DIR / "normalization_axis_scatter_centered.png",
        annotate=["openmidnight", "hibou_l", "lunit-vits8", "genbio-pathfm", "uni_v2"],
    )
    print(f"Wrote figures -> {OUT_DIR}")
