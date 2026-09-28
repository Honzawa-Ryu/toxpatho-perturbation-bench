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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.repr_metrics import (
    centered_paired_cosine_sim,
    linear_cka,
    paired_cosine_sim,
    paired_mse,
    participation_ratio,
    relative_mse,
)


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


def discover_completed_models(embed_dir: Path, variant_suffix: str) -> dict[str, str]:
    """Map clean model name -> its output directory name under embed_dir.

    Same convention as Exp 0004: Exp 0003's variant_key is
    "{model}__{source_variant}", so `variant_suffix` (e.g.
    "__256px_mpp0.5_n1000x20") selects which scale/variant to evaluate.
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


def load_embeddings(model_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Read embeddings.parquet into a dense (N, D) array + a metadata frame.

    Same pyarrow-direct-read pattern as Exp 0004's eval_model: pd.read_parquet
    boxes each list<float> row as a separate Python list (several x memory),
    and combine_chunks() can overflow pyarrow's int32 list-offset limit on
    the largest models -- filling a preallocated array chunk by chunk avoids
    both.
    """
    table = pq.read_table(model_dir / "embeddings.parquet")
    n_rows = table.num_rows
    embedding_col = table.column("embedding")
    dim = len(embedding_col.chunk(0)[0])
    embeddings = np.empty((n_rows, dim), dtype=np.float32)
    offset = 0
    for chunk in embedding_col.chunks:
        n = len(chunk)
        embeddings[offset : offset + n] = chunk.values.to_numpy(zero_copy_only=False).reshape(n, dim)
        offset += n
    meta = table.select(["parent_patch_id", "source_type", "kind", "level"]).to_pandas()

    # Same exclusion as Exp 0004: patches Macenko cannot process, dropped from
    # both variants so the A/B (raw) and C/D (normalized) axes are measured on
    # one identical patch set.
    from lib.patch_sampling import EXCLUDED_PATCH_IDS

    keep = ~meta["parent_patch_id"].isin(EXCLUDED_PATCH_IDS)
    if not keep.all():
        embeddings = embeddings[keep.to_numpy()]
        meta = meta.loc[keep].reset_index(drop=True)

    return embeddings, meta


def split_orig_pert(
    embeddings: np.ndarray, meta: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Split into (orig_emb, orig_ids, matched_orig_idx, pert_meta).

    matched_orig_idx[i] is the row of orig_emb holding pert_meta.iloc[i]'s
    parent patch, so orig_emb[matched_orig_idx] lines up 1:1 with pert_meta
    (and the embeddings passed alongside it).
    """
    is_orig = (meta["source_type"] == "original").to_numpy()
    orig_ids = meta.loc[is_orig, "parent_patch_id"].to_numpy()
    orig_emb = embeddings[is_orig]
    id_to_idx = {pid: i for i, pid in enumerate(orig_ids)}

    pert_meta = meta.loc[~is_orig].reset_index(drop=True)
    pert_emb = embeddings[~is_orig]
    matched_orig_idx = pert_meta["parent_patch_id"].map(id_to_idx).to_numpy()

    return orig_emb, orig_ids, matched_orig_idx, pert_meta.assign(_pert_row=np.arange(len(pert_meta))), pert_emb


def _pair_summary(a: np.ndarray, b: np.ndarray) -> dict:
    """cos/mse/cka/effective-rank summary for one matched (a, b) pair of sets."""
    cos = paired_cosine_sim(a, b)
    cos_c = centered_paired_cosine_sim(a, b)
    mse = paired_mse(a, b)
    a_pr, a_pr_ratio = participation_ratio(a)
    b_pr, b_pr_ratio = participation_ratio(b)
    return {
        "n": len(a),
        "cos_sim_mean": float(cos.mean()),
        "cos_sim_std": float(cos.std()),
        "cos_sim_centered_mean": float(cos_c.mean()),
        "cos_sim_centered_std": float(cos_c.std()),
        "mse_mean": float(mse.mean()),
        "mse_mean_relative": relative_mse(float(mse.mean()), a),
        "cka_linear": linear_cka(a, b),
        "a_eff_rank": a_pr,
        "a_eff_rank_ratio": a_pr_ratio,
        "b_eff_rank": b_pr,
        "b_eff_rank_ratio": b_pr_ratio,
    }


def perturbation_axis_metrics(model_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """original vs. perturbed, per (kind, level), within a single embed variant.

    Also returns (orig_emb, orig_ids) -- the small (~20k-row) original-patch
    subset -- so the caller can feed it straight into
    `normalization_axis_metrics` instead of re-reading the full (~500k-row)
    embeddings.parquet a second time. Loading both variants' full arrays at
    once would roughly double this experiment's peak memory over Exp 0004's
    proven single-variant budget for no reason: the perturbed rows aren't
    needed for the normalization axis, only the originals are.
    """
    embeddings, meta = load_embeddings(model_dir)
    orig_emb, orig_ids, matched_orig_idx, pert_meta, pert_emb = split_orig_pert(embeddings, meta)

    rows = []
    for (kind, level), g in pert_meta.groupby(["kind", "level"]):
        idx = g["_pert_row"].to_numpy()
        summary = _pair_summary(orig_emb[matched_orig_idx[idx]], pert_emb[idx])
        rows.append({"kind": kind, "level": level, **summary})

    summary_all = _pair_summary(orig_emb[matched_orig_idx], pert_emb)
    rows.append({"kind": "ALL", "level": None, **summary_all})

    df = pd.DataFrame(rows).rename(
        columns={"a_eff_rank": "orig_eff_rank", "a_eff_rank_ratio": "orig_eff_rank_ratio",
                 "b_eff_rank": "pert_eff_rank", "b_eff_rank_ratio": "pert_eff_rank_ratio"}
    )
    return df, orig_emb, orig_ids


def normalization_axis_metrics(
    base_orig: np.ndarray, base_ids: np.ndarray, norm_orig: np.ndarray, norm_ids: np.ndarray
) -> dict:
    """embed_variant_base vs. embed_variant_norm, on unperturbed original patches only.

    Takes the small original-patch subsets already extracted by
    `perturbation_axis_metrics` for each variant (see its docstring).
    """
    # Align by patch id rather than assuming row order matches across the two
    # independently-written embeddings.parquet files.
    norm_id_to_idx = {pid: i for i, pid in enumerate(norm_ids)}
    common = [pid for pid in base_ids if pid in norm_id_to_idx]
    base_idx = {pid: i for i, pid in enumerate(base_ids)}
    a = base_orig[[base_idx[pid] for pid in common]]
    b = norm_orig[[norm_id_to_idx[pid] for pid in common]]

    summary = _pair_summary(a, b)
    summary["base_eff_rank"] = summary.pop("a_eff_rank")
    summary["base_eff_rank_ratio"] = summary.pop("a_eff_rank_ratio")
    summary["norm_eff_rank"] = summary.pop("b_eff_rank")
    summary["norm_eff_rank_ratio"] = summary.pop("b_eff_rank_ratio")
    return {"kind": "ALL", "level": None, **summary}


def plot_normalization_summary(norm_metrics: pd.DataFrame, out_path: Path) -> None:
    """Per-model base-vs-norm agreement on unperturbed patches: cos_sim_mean
    (pairwise) vs. cka_linear (set-level geometry), sorted by cka_linear.

    A model where these two disagree (cos_sim near 1 but cka much lower)
    has embeddings that individually barely move yet whose overall geometry
    does -- see plot_normalization_scatter for the same signal isolated.
    """
    df = norm_metrics.sort_values("cka_linear")
    y = range(len(df))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1.5))
    ax.hlines(y, df["cos_sim_mean"], df["cka_linear"], color="lightgray", linewidth=1, zorder=1)
    ax.scatter(df["cos_sim_mean"], y, color="tab:blue", label="cos_sim_mean (pairwise)", zorder=2, s=28)
    ax.scatter(df["cka_linear"], y, color="tab:red", label="cka_linear (set-level)", zorder=2, s=28)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("similarity (1 = unchanged by normalization)")
    ax.set_title("Normalization axis: base vs. stain-normalized, original patches only")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_normalization_scatter(norm_metrics: pd.DataFrame, out_path: Path, annotate: list[str] | None = None) -> None:
    """cos_sim_mean vs. cka_linear per model. A point off the diagonal (high
    cos_sim, lower cka) usually means the representation has collapsed onto
    very few effective dimensions (cross-check against eff_rank_ratio),
    which makes the set-level geometry (cka) far more sensitive to a small
    shift than any individual pairwise distance (cos_sim) shows.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--", linewidth=1, zorder=1)
    ax.scatter(norm_metrics["cos_sim_mean"], norm_metrics["cka_linear"], color="tab:blue", s=40, zorder=2)
    for model in annotate or []:
        row = norm_metrics[norm_metrics["model"] == model].iloc[0]
        ax.annotate(
            model, (row["cos_sim_mean"], row["cka_linear"]), fontsize=8,
            xytext=(5, 5), textcoords="offset points",
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("cos_sim_mean (pairwise)")
    ax.set_ylabel("cka_linear (set-level)")
    ax.set_title("Normalization axis: pairwise vs. set-level agreement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_perturbation_heatmap(pert_metrics: pd.DataFrame, level: int, out_path: Path) -> None:
    """Small multiples: {cka_linear, cos_sim_mean} x {base, norm} model x kind heatmaps at one level."""
    metrics = ["cka_linear", "cos_sim_mean"]
    variants = ["base", "norm"]
    sub_all = pert_metrics[(pert_metrics["level"] == level) & (pert_metrics["kind"] != "ALL")]
    n_kinds = sub_all["kind"].nunique()
    n_models = sub_all["model"].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(0.5 * n_kinds * 2 + 4, 0.28 * n_models * 2 + 3))
    for i, metric in enumerate(metrics):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            sub = sub_all[sub_all["variant"] == variant]
            pivot = sub.pivot(index="model", columns="kind", values=metric).sort_index()
            im = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            ax.set_title(f"{metric}, {variant}, level {level}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_perturbation_degradation(
    pert_metrics: pd.DataFrame, out_path: Path, metric: str = "cka_linear", variant: str = "base"
) -> None:
    """Per-kind `metric` vs. level, one line per model, for a single variant."""
    sub_v = pert_metrics[(pert_metrics["variant"] == variant) & (pert_metrics["kind"] != "ALL")]
    kinds = sorted(sub_v["kind"].unique())
    models = sorted(sub_v["model"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    for ax, kind in zip(axes.flat, kinds):
        for model in models:
            s = sub_v[(sub_v["kind"] == kind) & (sub_v["model"] == model)].sort_values("level")
            ax.plot(s["level"], s[metric], marker="o", markersize=3, linewidth=1, label=model)
        ax.set_title(kind, fontsize=9)
        ax.set_xticks([1, 2, 3])
        ax.set_ylim(0, 1.05)
    axes[0, 0].set_ylabel(metric)
    axes[1, 0].set_ylabel(metric)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=7, fontsize=7)
    fig.suptitle(f"{metric} vs. perturbation level, variant={variant}", y=1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    embed_exp: str = config["embed_exp"]
    embed_variant_base: str = config["embed_variant_base"]
    embed_variant_norm: str = config["embed_variant_norm"]

    variant_key = embed_variant_base
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(
        run_dir, exp_name=exp_name, variant_key=variant_key, seed=seed, embed_exp=embed_exp,
        embed_variant_base=embed_variant_base, embed_variant_norm=embed_variant_norm,
    )

    embed_dir = project_root / "outputs" / embed_exp
    models_base = discover_completed_models(embed_dir, variant_suffix=f"__{embed_variant_base}")
    models_norm = discover_completed_models(embed_dir, variant_suffix=f"__{embed_variant_norm}")
    models = sorted(set(models_base) & set(models_norm))
    skipped = (set(models_base) | set(models_norm)) - set(models)
    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"embed_dir: {embed_dir}")
    logger.info(f"{len(models)} models completed in both variants: {models}")
    if skipped:
        logger.info(f"Skipping {len(skipped)} model(s) missing one variant: {sorted(skipped)}")

    pert_rows = []
    norm_rows = []
    for i, model in enumerate(models):
        base_dir = embed_dir / models_base[model]
        norm_dir = embed_dir / models_norm[model]

        logger.info(f"[{i + 1}/{len(models)}] {model}: perturbation axis (base)")
        m, base_orig_emb, base_orig_ids = perturbation_axis_metrics(base_dir)
        m.insert(0, "variant", "base")
        m.insert(0, "model", model)
        pert_rows.append(m)

        logger.info(f"[{i + 1}/{len(models)}] {model}: perturbation axis (norm)")
        m, norm_orig_emb, norm_orig_ids = perturbation_axis_metrics(norm_dir)
        m.insert(0, "variant", "norm")
        m.insert(0, "model", model)
        pert_rows.append(m)

        logger.info(f"[{i + 1}/{len(models)}] {model}: normalization axis")
        n = normalization_axis_metrics(base_orig_emb, base_orig_ids, norm_orig_emb, norm_orig_ids)
        n["model"] = model
        norm_rows.append(n)

    pert_metrics = pd.concat(pert_rows, ignore_index=True)
    pert_path = run_dir / "metrics_perturbation_axis.parquet"
    pert_metrics.to_parquet(pert_path, index=False)
    logger.info(f"Wrote {len(pert_metrics)} rows -> {pert_path}")

    norm_metrics = pd.DataFrame(norm_rows)
    norm_cols = ["model", "kind", "level"] + [c for c in norm_metrics.columns if c not in ("model", "kind", "level")]
    norm_metrics = norm_metrics[norm_cols]
    norm_path = run_dir / "metrics_normalization_axis.parquet"
    norm_metrics.to_parquet(norm_path, index=False)
    logger.info(f"Wrote {len(norm_metrics)} rows -> {norm_path}")

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_normalization_summary(norm_metrics, figures_dir / "normalization_axis_summary.png")
    gap = (norm_metrics["cos_sim_mean"] - norm_metrics["cka_linear"]).abs()
    annotate = list(norm_metrics.loc[gap.nlargest(2).index, "model"])
    lowest_cka_model = norm_metrics.loc[norm_metrics["cka_linear"].idxmin(), "model"]
    if lowest_cka_model not in annotate:
        annotate.append(lowest_cka_model)
    plot_normalization_scatter(norm_metrics, figures_dir / "normalization_axis_scatter.png", annotate=annotate)
    plot_perturbation_heatmap(pert_metrics, level=3, out_path=figures_dir / "perturbation_axis_heatmap_level3.png")
    plot_perturbation_degradation(
        pert_metrics, figures_dir / "perturbation_axis_degradation_cka_base.png", metric="cka_linear", variant="base"
    )
    logger.info(f"Wrote figures -> {figures_dir}")

    logger.info(
        "Normalization axis (base vs. norm, original patches only), sorted by cka_linear:\n"
        + norm_metrics[["model", "cos_sim_mean", "mse_mean_relative", "cka_linear", "base_eff_rank_ratio", "norm_eff_rank_ratio"]]
        .sort_values("cka_linear")
        .to_string(index=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
