"""Does the globally-centered cos_sim (Exp 0006's cos_sim_centered_mean) predict
retrieval top1 (Exp 0004, kind=="ALL") the way cka_linear does? Extends the
section 12-3 table of docs/task_a_implementation_plan.md with that column;
the cka_linear rows reproduce 12-3 as a sanity check. Results: section 12-6.

Not a pipeline experiment -- reads Exp 0006/0004's already-completed output.
Rerun via srun + apptainer:  python notebooks/centered_cos_vs_top1.py
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr
r6 = "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20/"
r4 = "outputs/0004_20260821_eval_retrieval_invariance/256px_mpp0.5_n1000x20"
p = pd.read_parquet(r6 + "metrics_perturbation_axis.parquet"); p = p[p.kind == "ALL"]
n = pd.read_parquet(r6 + "metrics_normalization_axis.parquet").set_index("model")
cols = ["cos_sim_mean", "cos_sim_centered_mean", "cka_linear"]
w = p.set_index(["model", "variant"])[cols].unstack("variant")
w.columns = [f"{'AB' if v=='base' else 'CD'}_{m}" for m, v in w.columns]
for m in cols: w[f"AC_{m}"] = n[m]
top = lambda d: pd.read_parquet(d + "/metrics_by_model_perturbation.parquet").query("kind=='ALL'").set_index("model")["top1_acc"]
w["acc_base"] = top(r4); w["acc_norm"] = top(r4 + "_orig")
def pc(x, y, z):
    x, y, z = map(rankdata, (x, y, z)); Z = np.c_[np.ones_like(z), z]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return pearsonr(res(x), res(y))  # p approx (df not adjusted)
for sub, lab in [(w, "all"), (w.drop("openmidnight"), "no_om")]:
    print(f"--- {lab} n={len(sub)}")
    for m in cols:
        for pre, acc in [("AB", "acc_base"), ("CD", "acc_norm"), ("AC", "acc_norm")]:
            r = spearmanr(sub[f"{pre}_{m}"], sub[acc]); print(f"{pre}_{m:24s} vs {acc}: {r.statistic:+.3f} p={r.pvalue:.4f}")
    for m in cols:
        for pre in ["AC", "CD"]:
            r = pc(sub[f"{pre}_{m}"], sub["acc_norm"], sub["acc_base"]); print(f"partial {pre}_{m:24s} vs acc_norm | acc_base: {r[0]:+.3f} p={r[1]:.4f}")
