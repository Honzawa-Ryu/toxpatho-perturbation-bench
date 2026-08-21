import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    return parser.parse_args()


def discover_completed_models(embed_dir: Path) -> list[str]:
    models = []
    for d in sorted(embed_dir.iterdir()):
        if not d.is_dir():
            continue
        completion_path = d / "completion.json"
        if not completion_path.exists():
            continue
        try:
            status = json.loads(completion_path.read_text()).get("status")
        except Exception:
            continue
        if status == "completed":
            models.append(d.name)
    return models


def eval_model(model_dir: Path) -> pd.DataFrame:
    """Return a per-perturbation (kind, level) metrics DataFrame for one model."""
    df = pd.read_parquet(model_dir / "embeddings.parquet")

    orig = df[df["source_type"] == "original"]
    pert = df[df["source_type"] == "perturbed"]

    gallery_ids = orig["parent_patch_id"].to_numpy()
    gallery_emb = np.stack(orig["embedding"].to_numpy()).astype(np.float32)
    gallery_emb /= np.linalg.norm(gallery_emb, axis=1, keepdims=True)
    id_to_idx = {pid: i for i, pid in enumerate(gallery_ids)}

    query_emb = np.stack(pert["embedding"].to_numpy()).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)
    true_idx = pert["parent_patch_id"].map(id_to_idx).to_numpy()

    sims = query_emb @ gallery_emb.T  # (n_query, n_gallery) cosine similarity
    true_sim = sims[np.arange(len(sims)), true_idx]
    # Rank of the true match among the gallery (1 = nearest neighbor). Ties are
    # broken optimistically (strict '>' only counts genuinely closer items).
    rank = (sims > true_sim[:, None]).sum(axis=1) + 1

    result = pert[["kind", "level"]].copy()
    result["rank"] = rank
    result["cos_sim"] = true_sim

    def _summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": len(g),
                "top1_acc": (g["rank"] == 1).mean(),
                "top5_acc": (g["rank"] <= 5).mean(),
                "top10_acc": (g["rank"] <= 10).mean(),
                "mrr": (1 / g["rank"]).mean(),
                "mean_cos_sim": g["cos_sim"].mean(),
                "mean_rank": g["rank"].mean(),
            }
        )

    by_kind_level = (
        result.groupby(["kind", "level"]).apply(_summarize, include_groups=False).reset_index()
    )

    overall = _summarize(result)
    overall["kind"] = "ALL"
    overall["level"] = None
    overall_df = pd.DataFrame([overall])[by_kind_level.columns]

    return pd.concat([by_kind_level, overall_df], ignore_index=True)


def plot_heatmap(metrics: pd.DataFrame, level: int, out_path: Path) -> None:
    sub = metrics[metrics["level"] == level]
    pivot = sub.pivot(index="kind", columns="model", values="top1_acc")
    fig, ax = plt.subplots(figsize=(0.5 * len(pivot.columns) + 3, 0.4 * len(pivot.index) + 2))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"Top-1 retrieval accuracy at level {level}")
    fig.colorbar(im, ax=ax, label="top1_acc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_degradation_curves(metrics: pd.DataFrame, out_path: Path) -> None:
    kinds = sorted(k for k in metrics["kind"].unique() if k != "ALL")
    models = sorted(metrics["model"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    for ax, kind in zip(axes.flat, kinds):
        for model in models:
            sub = metrics[(metrics["kind"] == kind) & (metrics["model"] == model)].sort_values("level")
            ax.plot(sub["level"], sub["top1_acc"], marker="o", markersize=3, linewidth=1, label=model)
        ax.set_title(kind, fontsize=9)
        ax.set_xticks([1, 2, 3])
        ax.set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("top1_acc")
    axes[1, 0].set_ylabel("top1_acc")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=7, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    embed_exp: str = config["embed_exp"]

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, seed=seed, embed_exp=embed_exp)

    embed_dir = project_root / "outputs" / embed_exp
    models = discover_completed_models(embed_dir)
    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"embed_dir: {embed_dir}")
    logger.info(f"Discovered {len(models)} completed models: {models}")

    all_metrics = []
    for i, model in enumerate(models):
        logger.info(f"[{i + 1}/{len(models)}] evaluating {model}")
        m = eval_model(embed_dir / model)
        m.insert(0, "model", model)
        all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics_path = run_dir / "metrics_by_model_perturbation.parquet"
    metrics.to_parquet(metrics_path, index=False)
    logger.info(f"Wrote {len(metrics)} rows -> {metrics_path}")

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(metrics, level=3, out_path=figures_dir / "heatmap_top1_level3.png")
    plot_degradation_curves(metrics, out_path=figures_dir / "degradation_curves.png")
    logger.info(f"Wrote figures -> {figures_dir}")

    overall = metrics[metrics["kind"] == "ALL"][["model", "top1_acc", "top5_acc", "mrr"]]
    overall = overall.sort_values("top1_acc", ascending=False)
    logger.info("Overall (all perturbations combined):\n" + overall.to_string(index=False))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
