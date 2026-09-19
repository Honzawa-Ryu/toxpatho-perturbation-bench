"""Why do cos_sim and cka_linear disagree? A 2x2 ablation over the two
definitional differences between them (see lib/repr_metrics.py):

                   | no centering        | centering
  -----------------+---------------------+---------------------
  per-sample       | cos_raw (current)   | cos_centered
  whole-set        | cka_uncentered      | cka_linear (current)

`paired_cosine_sim` scores each sample independently against its own pair
and keeps the set mean; `linear_cka` subtracts each set's mean and scores
the whole set's covariance at once. Which of those two differences drives
the observed gap is what this separates: if cos_centered tracks cka_linear,
centering explains it; if both cosine columns stay together and apart from
both CKA columns, sample coupling does.

Covers both of Exp 0006's axes: the perturbation axis (A-vs-B within base,
C-vs-D within norm) and the normalization axis (A-vs-C, unperturbed patches
only), so every figure that plots a cosine column has a centered
counterpart.

Reads Exp 0003 embeddings directly (not Exp 0006's summary output), reusing
Exp 0006's own loader and pairing so the cos_raw/cka_linear columns
reproduce its numbers. Too heavy for the login node -- submit with
notebooks/centering_ablation.sbatch.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.repr_metrics import linear_cka, paired_cosine_sim  # noqa: E402

# Exp 0006's directory name starts with a digit, so it cannot be imported by
# name -- load it by path rather than duplicating its loader/pairing logic.
_spec = importlib.util.spec_from_file_location(
    "exp0006", PROJECT_ROOT / "experiments/0006_20260904_eval_representation_shift/experiment.py"
)
exp0006 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp0006)

EMBED_DIR = PROJECT_ROOT / "outputs/0003_20260820_extract_patch_embeddings"
OUT_DIR = PROJECT_ROOT / "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20"

VARIANTS = {"base": "256px_mpp0.5_n1000x20", "norm": "256px_mpp0.5_n1000x20_orig"}


def centered_paired_cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """paired_cosine_sim after removing each set's own mean -- the only
    difference from paired_cosine_sim is the centering linear_cka applies.
    """
    return paired_cosine_sim(a - a.mean(axis=0, keepdims=True), b - b.mean(axis=0, keepdims=True))


def uncentered_linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """linear_cka with the mean subtraction removed, and nothing else."""
    cross = x.T @ y
    hsic_xy = np.linalg.norm(cross, ord="fro") ** 2
    hsic_xx = np.linalg.norm(x.T @ x, ord="fro")
    hsic_yy = np.linalg.norm(y.T @ y, ord="fro")
    return float(hsic_xy / (hsic_xx * hsic_yy))


def four_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "n": len(a),
        "cos_raw": float(paired_cosine_sim(a, b).mean()),
        "cos_centered": float(centered_paired_cosine_sim(a, b).mean()),
        "cka_uncentered": uncentered_linear_cka(a, b),
        "cka_linear": linear_cka(a, b),
    }


def main() -> None:
    models = sorted(
        set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['base']}"))
        & set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['norm']}"))
    )
    print(f"{len(models)} models completed under both variants: {models}", flush=True)

    rows = []
    for model in models:
        # Only the ~20k-row original subset is kept per variant (same reason as
        # Exp 0006's perturbation_axis_metrics): the normalization axis needs
        # the originals, never the 480k perturbed rows.
        orig_by_variant = {}

        for variant, suffix in VARIANTS.items():
            model_dir = EMBED_DIR / f"{model}__{suffix}"
            print(f"[{model}/{variant}] loading {model_dir}", flush=True)
            embeddings, meta = exp0006.load_embeddings(model_dir)
            orig_emb, orig_ids, matched_orig_idx, pert_meta, pert_emb = exp0006.split_orig_pert(embeddings, meta)
            del embeddings

            for (kind, level), g in pert_meta.groupby(["kind", "level"]):
                idx = g["_pert_row"].to_numpy()
                m = four_metrics(orig_emb[matched_orig_idx[idx]], pert_emb[idx])
                rows.append({"model": model, "axis": "perturbation", "variant": variant,
                             "kind": kind, "level": level, **m})
                print(f"  {kind} L{level}: {m}", flush=True)

            m_all = four_metrics(orig_emb[matched_orig_idx], pert_emb)
            rows.append({"model": model, "axis": "perturbation", "variant": variant,
                         "kind": "ALL", "level": None, **m_all})

            orig_by_variant[variant] = (orig_emb, orig_ids)
            del pert_emb, pert_meta

        # Normalization axis: base vs norm on the unperturbed originals, aligned
        # by patch id (row order is not guaranteed to match across the two
        # independently-written embeddings.parquet files).
        base_emb, base_ids = orig_by_variant["base"]
        norm_emb, norm_ids = orig_by_variant["norm"]
        norm_idx = {pid: i for i, pid in enumerate(norm_ids)}
        base_idx = {pid: i for i, pid in enumerate(base_ids)}
        common = [pid for pid in base_ids if pid in norm_idx]
        m_ac = four_metrics(
            base_emb[[base_idx[pid] for pid in common]], norm_emb[[norm_idx[pid] for pid in common]]
        )
        rows.append({"model": model, "axis": "normalization", "variant": "base_vs_norm",
                     "kind": "ALL", "level": None, **m_ac})
        print(f"[{model}/normalization] {m_ac}", flush=True)
        del orig_by_variant

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "centering_ablation.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows, {len(models)} models)")


if __name__ == "__main__":
    main()
