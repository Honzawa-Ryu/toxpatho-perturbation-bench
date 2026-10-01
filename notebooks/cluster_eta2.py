"""How much of the embedding variance do slide, study, and k-means clusters
each explain?

notebooks/cluster_ari.py compared clusters to slides by ARI, which turned out
to sit under its own seed-to-seed noise floor (section 13-1). This puts the
three partitions on one scale instead: the multivariate eta-squared

  eta2 = trace(SS_between) / trace(SS_total)

of the unperturbed patches' embeddings, for the partition by wsi_id (slide),
by EXP_ID (study), and by k-means at K in {20, 100, 261, 1000}. k-means
minimizes the within-cluster sum of squares, so its eta2 approximates the
most variance any K-way partition can explain: K=1000 is the ceiling for a
partition as fine as the slides, K=261 for one as fine as the studies.

eta2 rises with the number of groups even for labels carrying no information
(expected ~ (G-1)/(N-1)), so each row carries two corrections:
  omega2      the bias-corrected effect size,
              (SS_b - (G-1) MS_w) / (SS_t + MS_w), MS_w = SS_w / (N-G)
  eta2_perm   mean eta2 over N_PERM label shuffles, the no-information level

Evaluated in the same 50-dim PCA subspace cluster_ari.py clustered in (fit
per variant on that variant's unperturbed patches), so the k-means ceiling
holds in the space eta2 is measured in. Base (A) and norm (C) variants only.
k-means is also run at a second seed, as a check that its eta2 is stable.

Submit with notebooks/cluster_eta2.sbatch.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp0006 = _load("exp0006", PROJECT_ROOT / "experiments/0006_20260904_eval_representation_shift/experiment.py")
# Reused for the unperturbed-only loader and the TG-GATEs label join.
knn_purity = _load("study_knn_purity", PROJECT_ROOT / "notebooks/study_knn_purity.py")
# Same k-means settings as section 13-1, so the clusters are comparable.
cluster_ari = _load("cluster_ari", PROJECT_ROOT / "notebooks/cluster_ari.py")

EMBED_DIR = PROJECT_ROOT / "outputs/0003_20260820_extract_patch_embeddings"
OUT_DIR = PROJECT_ROOT / "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20"

VARIANTS = {"base": "256px_mpp0.5_n1000x20", "norm": "256px_mpp0.5_n1000x20_orig"}
N_COMPONENTS = 50
K_VALUES = [20, 100, 261, 1000]
KMEANS_SEEDS = [0, 1]
N_PERM = 100


def sum_squares(x: np.ndarray, labels: np.ndarray) -> tuple[float, float, int]:
    """(SS_between, SS_total, number of groups), traces over all dimensions."""
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    centered = x - x.mean(axis=0)
    sums = np.zeros((len(counts), x.shape[1]))
    np.add.at(sums, inv, centered)
    ss_between = float(((sums**2).sum(axis=1) / counts).sum())
    return ss_between, float((centered**2).sum()), len(counts)


def effect_sizes(x: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    ss_b, ss_t, g = sum_squares(x, labels)
    n = len(x)
    ms_w = (ss_t - ss_b) / (n - g)
    perm = [sum_squares(x, rng.permutation(labels))[0] / ss_t for _ in range(N_PERM)]
    return {"n_groups": g, "eta2": ss_b / ss_t,
            "omega2": (ss_b - (g - 1) * ms_w) / (ss_t + ms_w),
            "eta2_perm": float(np.mean(perm)), "eta2_perm_max": float(np.max(perm))}


def main() -> None:
    models = sorted(
        set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['base']}"))
        & set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['norm']}"))
    )
    print(f"{len(models)} models: {models}", flush=True)

    out_path = OUT_DIR / "cluster_eta2.csv"
    rows = []
    for model in models:
        for variant, suffix in VARIANTS.items():
            emb, ids = knn_purity.load_originals(EMBED_DIR / f"{model}__{suffix}")
            labels = knn_purity.load_labels(ids)
            x = PCA(n_components=N_COMPONENTS, random_state=0).fit_transform(emb)
            del emb
            rng = np.random.default_rng(0)

            partitions = [("wsi_id", None, labels["wsi_id"].to_numpy()),
                          ("EXP_ID", None, labels["EXP_ID"].to_numpy())]
            for k in K_VALUES:
                for seed in KMEANS_SEEDS:
                    partitions.append((f"kmeans_K{k}", seed, cluster_ari.cluster(x, k, seed=seed)))

            for name, seed, lab in partitions:
                rows.append({"model": model, "variant": variant, "partition": name,
                             "kmeans_seed": seed, **effect_sizes(x, lab, rng)})
            print(f"[{model}/{variant}] done", flush=True)

        # Rewritten after every model, so a slipped job still leaves a result.
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nWrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
