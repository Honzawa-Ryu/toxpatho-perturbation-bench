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
import pyarrow.parquet as pq
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


def load_config(exp_dir: Path, config_name: str = "config.yml") -> dict:
    """Load a config file from the experiment directory."""
    config_path = exp_dir / config_name
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    return parser.parse_args()


def discover_completed_models(embed_dir: Path, variant_suffix: str) -> dict[str, str]:
    """Map clean model name -> its output directory name under embed_dir.

    Exp 0003's variant_key is "{model}__{source_variant}" so pilot-scale and
    full-scale runs of the same model don't collide (see git history for why:
    without the suffix, a full-scale rerun silently no-ops against a pilot
    run's "completed" marker). `variant_suffix` selects which scale to
    evaluate here, e.g. "__256px_mpp0.5_n1000x20".
    """
    models: dict[str, str] = {}
    for d in sorted(embed_dir.iterdir()):
        if not d.is_dir() or not d.name.endswith(variant_suffix):
            continue
        completion_path = d / "completion.json"
        if not completion_path.exists():
            continue
        try:
            status = json.loads(completion_path.read_text()).get("status")
        except Exception:
            continue
        if status == "completed":
            models[d.name[: -len(variant_suffix)]] = d.name
    return models


def eval_model(model_dir: Path, query_batch_size: int = 4000) -> pd.DataFrame:
    """Return a per-perturbation (kind, level) metrics DataFrame for one model.

    The (n_query x n_gallery) similarity matrix is computed in query batches
    rather than all at once -- at pilot scale (48k x 2k) a single dense matrix
    is fine (384MB), but at full scale (both dimensions grow together) it
    would be tens of GB per model. Batching keeps peak memory bounded
    regardless of dataset size, at no extra FLOPs cost.
    """
    # Read via pyarrow and reshape the embedding column's flat float buffer
    # directly, instead of pd.read_parquet(): pandas materializes a list<float>
    # column as one Python list object per row, which is several times the raw
    # data size in memory. That was fine at pilot scale but OOM'd at full scale
    # on the largest-embedding-dim model (genbio-pathfm, ~5800-d, 11.7GB file)
    # even with 32GB requested.
    table = pq.read_table(model_dir / "embeddings.parquet")
    n_rows = table.num_rows
    embedding_col = table.column("embedding")
    # combine_chunks()/concat over the whole column can overflow pyarrow's
    # 32-bit list offsets once n_rows * dim gets large enough (hit this on
    # genbio-pathfm: ~500k rows x ~5800-d = ~2.9B float elements, past the
    # ~2.1B int32 limit). Each chunk individually (one Exp 0003 write batch,
    # a few hundred rows) is always well within range, so fill a
    # preallocated array chunk by chunk instead of concatenating first.
    dim = len(embedding_col.chunk(0)[0])
    embeddings = np.empty((n_rows, dim), dtype=np.float32)
    offset = 0
    for chunk in embedding_col.chunks:
        n = len(chunk)
        embeddings[offset : offset + n] = chunk.values.to_numpy(zero_copy_only=False).reshape(n, dim)
        offset += n
    meta = table.select(["parent_patch_id", "source_type", "kind", "level"]).to_pandas()

    is_orig = (meta["source_type"] == "original").to_numpy()
    gallery_ids = meta.loc[is_orig, "parent_patch_id"].to_numpy()
    gallery_emb = embeddings[is_orig].copy()
    gallery_emb /= np.linalg.norm(gallery_emb, axis=1, keepdims=True)
    id_to_idx = {pid: i for i, pid in enumerate(gallery_ids)}

    pert = meta.loc[~is_orig].reset_index(drop=True)
    query_emb = embeddings[~is_orig].copy()
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)
    true_idx = pert["parent_patch_id"].map(id_to_idx).to_numpy()

    rank = np.empty(len(pert), dtype=np.int64)
    true_sim = np.empty(len(pert), dtype=np.float32)
    for start in range(0, len(query_emb), query_batch_size):
        end = start + query_batch_size
        sims = query_emb[start:end] @ gallery_emb.T  # (batch, n_gallery) cosine similarity
        batch_true_idx = true_idx[start:end]
        batch_true_sim = sims[np.arange(len(sims)), batch_true_idx]
        # Rank of the true match among the gallery (1 = nearest neighbor). Ties
        # are broken optimistically (strict '>' only counts genuinely closer items).
        rank[start:end] = (sims > batch_true_sim[:, None]).sum(axis=1) + 1
        true_sim[start:end] = batch_true_sim

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


def plot_heatmap(
    metrics: pd.DataFrame, level: int, out_path: Path, metric: str = "top1_acc", metric_label: str = "Top-1"
) -> None:
    sub = metrics[metrics["level"] == level]
    pivot = sub.pivot(index="kind", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(0.5 * len(pivot.columns) + 3, 0.4 * len(pivot.index) + 2))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"{metric_label} retrieval accuracy at level {level}")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def rank_table(metrics: pd.DataFrame, level: int, metric: str = "top1_acc") -> pd.DataFrame:
    """Model x kind rank (1=best) within each kind's column, plus a `mean` column, sorted best-first."""
    sub = metrics[metrics["level"] == level]
    pivot = sub.pivot(index="kind", columns="model", values=metric).T  # model=rows, kind=columns
    rank = pivot.rank(axis=0, ascending=False)  # rank 1 = best (highest score) within each kind column
    rank["mean"] = rank.mean(axis=1)
    return rank.sort_values("mean")  # best average rank first


def plot_rank_heatmap(
    metrics: pd.DataFrame, level: int, out_path: Path, metric: str = "top1_acc", metric_label: str = "Top-1"
) -> None:
    """Model x kind heatmap colored by rank (1=best) within each kind, not raw score.

    Raw-score heatmaps let the hardest kinds (e.g. gaussian_noise, where
    everyone collapses toward 0) drown out real differences on easier kinds.
    Ranking within each kind's column removes that scale mismatch, so a
    model that's consistently near the top across kinds -- not just strong
    on the kinds that happen to be easy overall -- reads as a clean row.
    """
    rank = rank_table(metrics, level, metric)
    n_models = len(rank.index)
    fig, ax = plt.subplots(figsize=(0.5 * len(rank.columns) + 3, 0.3 * len(rank.index) + 2))
    values = rank.to_numpy()
    im = ax.imshow(values, cmap="RdYlGn_r", vmin=1, vmax=n_models, aspect="auto")
    ax.axvline(len(rank.columns) - 1.5, color="black", linewidth=1)  # separator before the mean column

    # Annotate each cell with its value; flip text color near the colormap's
    # dark extremes (best/worst) so it stays readable against the fill.
    norm_values = (values - 1) / (n_models - 1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if norm_values[i, j] < 0.2 or norm_values[i, j] > 0.8 else "black"
            ax.text(
                j, i, f"{values[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=7
            )
    ax.set_xticks(range(len(rank.columns)))
    ax.set_xticklabels(rank.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rank.index)))
    ax.set_yticklabels(rank.index, fontsize=8)
    ax.set_title(f"{metric_label} rank per kind at level {level} (1=best of {n_models})")
    fig.colorbar(im, ax=ax, label="rank (1=best)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_degradation_curves(
    metrics: pd.DataFrame, out_path: Path, metric: str = "top1_acc", log_scale: bool = False
) -> None:
    """Per-kind `metric` vs. level, one line per model.

    log_scale=True puts the y-axis on a log scale, which spreads out models
    that are bunched up near the 1.0 ceiling (e.g. stain_jitter, where every
    model stays >=0.84 and the linear plot looks flat) at the cost of
    compressing the near-zero (fully collapsed) end.
    """
    kinds = sorted(k for k in metrics["kind"].unique() if k != "ALL")
    models = sorted(metrics["model"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    for ax, kind in zip(axes.flat, kinds):
        for model in models:
            sub = metrics[(metrics["kind"] == kind) & (metrics["model"] == model)].sort_values("level")
            ax.plot(sub["level"], sub[metric], marker="o", markersize=3, linewidth=1, label=model)
        ax.set_title(kind, fontsize=9)
        ax.set_xticks([1, 2, 3])
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.3)
        else:
            ax.set_ylim(0, 1.05)
    axes[0, 0].set_ylabel(metric + (" (log)" if log_scale else ""))
    axes[1, 0].set_ylabel(metric + (" (log)" if log_scale else ""))
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

    args = parse_args()

    config = load_config(Path(__file__).parent, args.config)
    seed: int = config.get("seed", 42)
    embed_exp: str = config["embed_exp"]
    embed_variant: str = config["embed_variant"]

    variant_key = embed_variant
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, seed=seed, embed_exp=embed_exp)

    embed_dir = project_root / "outputs" / embed_exp
    models = discover_completed_models(embed_dir, variant_suffix=f"__{embed_variant}")
    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"embed_dir: {embed_dir}")
    logger.info(f"Discovered {len(models)} completed models: {sorted(models)}")

    all_metrics = []
    for i, (model, dir_name) in enumerate(sorted(models.items())):
        logger.info(f"[{i + 1}/{len(models)}] evaluating {model}")
        m = eval_model(embed_dir / dir_name)
        m.insert(0, "model", model)
        all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics_path = run_dir / "metrics_by_model_perturbation.parquet"
    metrics.to_parquet(metrics_path, index=False)
    logger.info(f"Wrote {len(metrics)} rows -> {metrics_path}")

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(metrics, level=3, out_path=figures_dir / "heatmap_top1_level3.png", metric="top1_acc", metric_label="Top-1")
    plot_heatmap(metrics, level=3, out_path=figures_dir / "heatmap_top5_level3.png", metric="top5_acc", metric_label="Top-5")
    plot_rank_heatmap(metrics, level=3, out_path=figures_dir / "heatmap_rank_level3.png", metric="top1_acc", metric_label="Top-1")
    plot_degradation_curves(metrics, out_path=figures_dir / "degradation_curves.png", metric="top1_acc")
    plot_degradation_curves(metrics, out_path=figures_dir / "degradation_curves_log.png", metric="top1_acc", log_scale=True)
    plot_degradation_curves(metrics, out_path=figures_dir / "degradation_curves_top5.png", metric="top5_acc")
    plot_degradation_curves(metrics, out_path=figures_dir / "degradation_curves_top5_log.png", metric="top5_acc", log_scale=True)
    logger.info(f"Wrote figures -> {figures_dir}")

    overall = metrics[metrics["kind"] == "ALL"][["model", "top1_acc", "top5_acc", "mrr"]]
    overall = overall.sort_values("top1_acc", ascending=False)
    logger.info("Overall (all perturbations combined):\n" + overall.to_string(index=False))

    rank_by_kind = rank_table(metrics, level=3, metric="top1_acc")
    rank_path = run_dir / "rank_by_kind_level3.csv"
    rank_by_kind.round(1).to_csv(rank_path)
    logger.info(f"Top-1 rank per kind at level 3 (1=best of {len(rank_by_kind)}), sorted by mean:\n" + rank_by_kind.round(1).to_string())

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
