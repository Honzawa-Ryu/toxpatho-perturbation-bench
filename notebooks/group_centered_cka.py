"""Is cka_linear measuring morphology, or the batch structure it sits on?

notebooks/study_knn_purity.py found the unperturbed patches are grouped by
slide at 137x chance and by study at 30.6x. cka_linear is computed on the
covariance of that same 20k-patch set, so a large part of what it compares
may be those group offsets rather than anything about a patch.

Same mechanism as notebooks/centering_ablation.py -- subtract a mean before
comparing -- with the unit changed from the whole set to the group:

  global   what Exp 0006 already does (lib.repr_metrics.linear_cka)
  wsi      each slide's own mean removed   (1000 groups, ~20 patches each)
  exp      each study's own mean removed   (261 groups, ~77 patches each)

Group-centering leaves the global mean exactly zero, so linear_cka's own
centering is a no-op on the result and the three are the same estimator
applied to differently-centered inputs.

Absolute CKA must fall under group centering -- removing 1000 group means
removes real variance -- so that is not the reading. The reading is whether
the model *ranking* survives: rank agreement with the global-centered CKA
near 1 means Exp 0006 was ordering models on something batch removal does
not touch; a low one means it was ordering them on batch.

Level 3 only, matching cluster_ari.py's scope. Submit with
notebooks/group_centered_cka.sbatch.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.repr_metrics import linear_cka  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp0006", PROJECT_ROOT / "experiments/0006_20260904_eval_representation_shift/experiment.py"
)
exp0006 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp0006)

EMBED_DIR = PROJECT_ROOT / "outputs/0003_20260820_extract_patch_embeddings"
META_CSV = PROJECT_ROOT / "data/corrected/open_tggates_pathological_image.csv"
OUT_DIR = PROJECT_ROOT / "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20"

VARIANTS = {"base": "256px_mpp0.5_n1000x20", "norm": "256px_mpp0.5_n1000x20_orig"}
LEVEL = 3


def group_index(labels: np.ndarray) -> tuple[np.ndarray, int]:
    _, idx = np.unique(labels, return_inverse=True)
    return idx.astype(np.int64), int(idx.max()) + 1


def group_center(x: np.ndarray, gidx: np.ndarray, n_groups: int) -> np.ndarray:
    """x with each group's own mean subtracted, via one sorted pass."""
    order = np.argsort(gidx, kind="stable")
    starts = np.searchsorted(gidx[order], np.arange(n_groups))
    sums = np.add.reduceat(x[order].astype(np.float64), starts, axis=0)
    means = (sums / np.bincount(gidx, minlength=n_groups)[:, None]).astype(np.float32)
    return x - means[gidx]


def study_labels(ids: np.ndarray) -> dict[str, np.ndarray]:
    meta = pd.read_csv(META_CSV)
    meta["wsi_id"] = meta["FILE_LOCATION"].str.rsplit("/", n=1).str[-1].str.replace(".svs", "", regex=False)
    meta = meta.drop_duplicates("wsi_id").set_index("wsi_id")
    wsi = np.array([pid.rsplit("_", 2)[0] for pid in ids])
    exp = meta.reindex(wsi)["EXP_ID"].to_numpy()
    if pd.isna(exp).any():
        raise ValueError("some patches did not join to TG-GATEs metadata")
    return {"wsi": wsi, "exp": exp}


def main() -> None:
    models = sorted(
        set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['base']}"))
        & set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['norm']}"))
    )
    print(f"{len(models)} models: {models}", flush=True)

    out_path = OUT_DIR / "group_centered_cka.csv"
    rows = []
    for model in models:
        orig_by_variant = {}

        for variant, suffix in VARIANTS.items():
            print(f"[{model}/{variant}] loading", flush=True)
            embeddings, meta = exp0006.load_embeddings(EMBED_DIR / f"{model}__{suffix}")
            orig_emb, orig_ids, matched_orig_idx, pert_meta, pert_emb = exp0006.split_orig_pert(embeddings, meta)
            del embeddings

            groups = study_labels(orig_ids)
            gidx = {name: group_index(lab) for name, lab in groups.items()}

            level3 = pert_meta[pert_meta["level"] == LEVEL]
            for kind, g in level3.groupby("kind"):
                idx = g["_pert_row"].to_numpy()
                order = np.argsort(matched_orig_idx[idx])
                if not np.array_equal(matched_orig_idx[idx][order], np.arange(len(orig_emb))):
                    raise ValueError(f"{model}/{variant}/{kind}: not one perturbed row per original patch")
                b = pert_emb[idx[order]]
                rows.append({"model": model, "variant": variant, "kind": kind,
                             "centering": "global", "cka": linear_cka(orig_emb, b)})
                for name, (gi, ng) in gidx.items():
                    rows.append({"model": model, "variant": variant, "kind": kind, "centering": name,
                                 "cka": linear_cka(group_center(orig_emb, gi, ng), group_center(b, gi, ng))})
            print(f"  level {LEVEL} kinds done", flush=True)

            orig_by_variant[variant] = (orig_emb, orig_ids, gidx)
            del pert_emb, pert_meta

        # Normalization axis under the same three centerings.
        (base_emb, base_ids, base_gidx) = orig_by_variant["base"]
        (norm_emb, norm_ids, _) = orig_by_variant["norm"]
        pos = {pid: i for i, pid in enumerate(norm_ids)}
        take = [pos[pid] for pid in base_ids]
        a, c = base_emb, norm_emb[take]
        rows.append({"model": model, "variant": "base_vs_norm", "kind": "ALL",
                     "centering": "global", "cka": linear_cka(a, c)})
        for name, (gi, ng) in base_gidx.items():
            rows.append({"model": model, "variant": "base_vs_norm", "kind": "ALL", "centering": name,
                         "cka": linear_cka(group_center(a, gi, ng), group_center(c, gi, ng))})
        del orig_by_variant

        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[{model}] done; {len(rows)} rows", flush=True)

    print(f"\nWrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
