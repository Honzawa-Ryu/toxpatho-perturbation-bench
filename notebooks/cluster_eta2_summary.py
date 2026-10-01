"""Summarizes cluster_eta2.csv (notebooks/cluster_eta2.py) for section 13-7,
and tests whether a model's slide/study eta2 predicts how much retrieval
top1 (Exp 0004, kind=="ALL") it loses to stain normalization.

acc_drop is the relative change (norm - base) / base, the same ratio as
notebooks/compare_stainnorm.py's norm_ratio minus one. The partial Spearman
controls for acc_base, as in section 12-3.

Also correlates base eta2 with each (kind, level)'s top1 relative to the
model's overall top1, to find which perturbations high-eta2 models are
relatively weak or strong against.

Not a pipeline experiment -- reads already-completed output.
Rerun via srun + apptainer:  python notebooks/cluster_eta2_summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ETA2_CSV = PROJECT_ROOT / "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20/cluster_eta2.csv"
RETRIEVAL_DIR = PROJECT_ROOT / "outputs/0004_20260821_eval_retrieval_invariance/256px_mpp0.5_n1000x20"


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman of x and y after regressing z out of both ranks (p not df-adjusted)."""
    x, y, z = map(rankdata, (x, y, z))
    design = np.c_[np.ones_like(z), z]
    resid = lambda v: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]  # noqa: E731
    r, p = pearsonr(resid(x), resid(y))
    return float(r), float(p)


def top1(path: Path) -> pd.Series:
    df = pd.read_parquet(path / "metrics_by_model_perturbation.parquet")
    return df[df["kind"] == "ALL"].set_index("model")["top1_acc"]


def main() -> None:
    pd.set_option("display.width", 200)
    d = pd.read_csv(ETA2_CSV)

    km = d[d["partition"].str.startswith("kmeans")]
    seeds = km.pivot_table(index=["model", "variant", "partition"], columns="kmeans_seed", values="eta2")
    print(f"k-means eta2, max |seed0 - seed1|: {(seeds[0] - seeds[1]).abs().max():.4f}")

    d = d[d["kmeans_seed"].isna() | (d["kmeans_seed"] == 0)]
    eta = d.pivot_table(index=["model", "variant"], columns="partition", values="eta2")
    t = pd.DataFrame({
        "slide": eta["wsi_id"], "study": eta["EXP_ID"],
        "within_study_slide": eta["wsi_id"] - eta["EXP_ID"],
        "K261": eta["kmeans_K261"], "K1000": eta["kmeans_K1000"],
        "slide/K1000": eta["wsi_id"] / eta["kmeans_K1000"],
        "study/K261": eta["EXP_ID"] / eta["kmeans_K261"],
    })
    base, norm = t.xs("base", level="variant"), t.xs("norm", level="variant")
    summary = pd.DataFrame({
        "base": base.mean(), "norm": norm.mean(),
        "rel_change": ((norm - base) / base).mean(),
        "n_decreasing": (norm < base).sum(),
    })
    print("\nMean over models:\n", summary.round(3))

    acc = pd.DataFrame({"acc_base": top1(RETRIEVAL_DIR), "acc_norm": top1(Path(f"{RETRIEVAL_DIR}_orig"))})
    acc["acc_drop"] = (acc["acc_norm"] - acc["acc_base"]) / acc["acc_base"]
    preds = pd.DataFrame({
        "slide_base": base["slide"], "study_base": base["study"],
        "study_drop": (norm["study"] - base["study"]) / base["study"],
        "slide_drop": (norm["slide"] - base["slide"]) / base["slide"],
    }).join(acc, how="inner")

    for label, sub in [("all", preds), ("no_openmidnight", preds.drop("openmidnight"))]:
        print(f"\n--- {label} (n={len(sub)})")
        for col in ["slide_base", "study_base", "slide_drop", "study_drop"]:
            r = spearmanr(sub[col], sub["acc_drop"])
            pr, pp = partial_spearman(sub[col], sub["acc_drop"], sub["acc_base"])
            print(f"{col:11s} vs acc_drop: rho={r.statistic:+.3f} p={r.pvalue:.4f}"
                  f" | partial(acc_base) rho={pr:+.3f} p={pp:.4f}")
        r = spearmanr(sub["study_base"], sub["acc_base"])
        print(f"study_base  vs acc_base: rho={r.statistic:+.3f} p={r.pvalue:.4f}")

    print("\n", preds.sort_values("study_base").round(3).to_string())

    # Per perturbation: top1 of each (kind, level) divided by the model's
    # overall top1, so a model's general strength drops out and what remains
    # is which perturbations it is relatively weak or strong against.
    metrics = pd.read_parquet(RETRIEVAL_DIR / "metrics_by_model_perturbation.parquet")
    per_kind = metrics[metrics["kind"] != "ALL"]
    rows = []
    for (kind, level), g in per_kind.groupby(["kind", "level"]):
        rel = g.set_index("model")["top1_acc"] / acc["acc_base"]
        for part, col in [("slide", "slide_base"), ("study", "study_base")]:
            x = preds[col].reindex(rel.index)
            r = spearmanr(x, rel)
            keep = rel.index != "openmidnight"
            r_no = spearmanr(x[keep], rel[keep])
            rows.append({"eta2": part, "kind": kind, "level": int(level),
                         "rho": r.statistic, "p": r.pvalue,
                         "rho_no_openmidnight": r_no.statistic, "p_no_openmidnight": r_no.pvalue})
    per = pd.DataFrame(rows)
    per["bonferroni_ok"] = per["p"] < 0.05 / len(per)
    print(f"\nPer-perturbation relative top1 vs base eta2 ({len(per)} tests):")
    print(per.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
