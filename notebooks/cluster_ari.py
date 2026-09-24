"""Cluster-level robustness: does a perturbation keep patches grouped
together, even when it moves them off their own parent?

Sits between the two levels already measured. Exp 0004 asks whether a
perturbed patch retrieves its own parent (identity); Exp 0006 asks whether
the whole set's covariance survives (global geometry). Neither says whether
the grouping structure survives, which is what ARI between two partitions
measures here:

  ARI(A, B)   perturbation, raw          ARI(C, D)  perturbation, normalized
  ARI(A, C)   stain normalization itself
  ARI(A, WSI) how far the clustering is just recovering slide identity

That last one is the point of including WSI: section 10-3 found stain
normalization costs retrieval accuracy and read it as a per-slide colour
shortcut being removed. If that is what the shortcut is, clusters should
track wsi_id less tightly after normalization -- ARI(A, WSI) vs ARI(C, WSI)
tests it directly, on a metric that never sees accuracy.

A noise floor is measured alongside every comparison: the same data
clustered twice under different seeds. k-means is not deterministic and
gets less stable as K grows, so an ARI drop only means something once it
clears that floor.

Level 3 only (the strongest perturbation, where Exp 0006 saw the widest
spread), PCA to 50 dims, K in {20, 100, 1000}. K=1000 matches the number of
WSIs, so at that K the clustering and the slide partition are the same
granularity. MiniBatchKMeans rather than KMeans: at K=1000 exact k-means
would need ~5.7h for 23 models, over large-andre01's 4h ceiling -- the
extra run-to-run variance that trades for is what the noise floor measures.

Submit with notebooks/cluster_ari.sbatch.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "exp0006", PROJECT_ROOT / "experiments/0006_20260904_eval_representation_shift/experiment.py"
)
exp0006 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp0006)

EMBED_DIR = PROJECT_ROOT / "outputs/0003_20260820_extract_patch_embeddings"
OUT_DIR = PROJECT_ROOT / "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20"

VARIANTS = {"base": "256px_mpp0.5_n1000x20", "norm": "256px_mpp0.5_n1000x20_orig"}
LEVEL = 3
N_COMPONENTS = 50
K_VALUES = [20, 100, 1000]


def cluster(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    # batch_size scales with k: a mini-batch smaller than the number of
    # centroids leaves most of them untouched per step, which at k=1000 makes
    # the fit far noisier than the seed-to-seed floor is meant to capture.
    return MiniBatchKMeans(
        n_clusters=k, random_state=seed, n_init=3, batch_size=max(1024, 4 * k)
    ).fit_predict(x)


def main() -> None:
    models = sorted(
        set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['base']}"))
        & set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['norm']}"))
    )
    print(f"{len(models)} models: {models}", flush=True)

    out_path = OUT_DIR / "cluster_ari.csv"
    rows = []
    for model in models:
        # labels of the unperturbed patches per variant, kept so the
        # normalization axis (A vs C) can be compared after both are done.
        orig_labels = {}
        orig_ids_by_variant = {}

        for variant, suffix in VARIANTS.items():
            print(f"[{model}/{variant}] loading", flush=True)
            embeddings, meta = exp0006.load_embeddings(EMBED_DIR / f"{model}__{suffix}")
            orig_emb, orig_ids, matched_orig_idx, pert_meta, pert_emb = exp0006.split_orig_pert(embeddings, meta)
            del embeddings

            # The unperturbed patches define the subspace; perturbed sets are
            # projected into it rather than getting their own, so a partition
            # change reflects the perturbation and not a change of basis.
            pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(orig_emb)
            orig_pca = pca.transform(orig_emb)
            wsi = np.array([pid.split("_")[0] for pid in orig_ids])

            level3 = pert_meta[pert_meta["level"] == LEVEL]
            pert_pca_by_kind = {}
            for kind, g in level3.groupby("kind"):
                idx = g["_pert_row"].to_numpy()
                # Reorder to the originals' row order so the two label vectors
                # describe the same patches position by position. Sorting only
                # achieves that if the group covers each original exactly once.
                order = np.argsort(matched_orig_idx[idx])
                if not np.array_equal(matched_orig_idx[idx][order], np.arange(len(orig_emb))):
                    raise ValueError(f"{model}/{variant}/{kind}: not one perturbed row per original patch")
                pert_pca_by_kind[kind] = pca.transform(pert_emb[idx[order]])
            del pert_emb, pert_meta, orig_emb

            for k in K_VALUES:
                lab_orig = cluster(orig_pca, k, seed=0)
                lab_orig_alt = cluster(orig_pca, k, seed=1)
                rows.append({"model": model, "variant": variant, "K": k,
                             "comparison": "noise_floor", "kind": None,
                             "ari": adjusted_rand_score(lab_orig, lab_orig_alt)})
                rows.append({"model": model, "variant": variant, "K": k,
                             "comparison": "vs_wsi", "kind": None,
                             "ari": adjusted_rand_score(lab_orig, wsi)})
                for kind, x in pert_pca_by_kind.items():
                    rows.append({"model": model, "variant": variant, "K": k,
                                 "comparison": "perturbation", "kind": kind,
                                 "ari": adjusted_rand_score(lab_orig, cluster(x, k, seed=0))})
                orig_labels[(variant, k)] = lab_orig
                print(f"  K={k} done", flush=True)

            orig_ids_by_variant[variant] = orig_ids

        # Normalization axis: same patches, clustered in each variant's own
        # PCA subspace. ARI compares labels, not coordinates, so the two
        # subspaces differing is not itself a problem.
        base_ids, norm_ids = orig_ids_by_variant["base"], orig_ids_by_variant["norm"]
        norm_pos = {pid: i for i, pid in enumerate(norm_ids)}
        common = [i for i, pid in enumerate(base_ids) if pid in norm_pos]
        norm_take = [norm_pos[base_ids[i]] for i in common]
        for k in K_VALUES:
            rows.append({"model": model, "variant": "base_vs_norm", "K": k,
                         "comparison": "normalization", "kind": None,
                         "ari": adjusted_rand_score(orig_labels[("base", k)][common],
                                                    orig_labels[("norm", k)][norm_take])})
        # Rewritten after every model: the run is estimated at ~2h against a
        # 4h ceiling, so a partial result is worth more than none if it slips.
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[{model}] normalization axis done; {len(rows)} rows written", flush=True)

    print(f"\nWrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
