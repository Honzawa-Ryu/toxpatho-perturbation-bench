"""Reorganizes Exp 0006's already-computed output into the A/B/C/D framing:
  A = original patches (20,000)
  B = A + perturbation (20,000 x 24 kind/level combos)
  C = A + stain normalization (20,000)
  D = C + perturbation (20,000 x 24 kind/level combos)

Exp 0006's "base" variant IS (A, B) and its "norm" variant IS (C, D), so
nothing needs recomputing -- this just re-labels/re-aggregates the two
existing parquet outputs:
  - EffRank of A/C (constant per model/variant, one value)
  - EffRank of B/D per (kind, level), plus a stability summary (min/max/std
    across the 24 combos) so "does it change much" has a number, not just
    a heatmap to eyeball
  - A-vs-B and C-vs-D similarity (cos_sim/cka), pooled "ALL" value
  - A-vs-C similarity (cos_sim/cka) -- Exp 0006's normalization axis

Not a pipeline experiment -- reads Exp 0006's already-completed output.
Rerun manually:  .venv/bin/python notebooks/abcd_summary.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20"
OUT_DIR = RUN_DIR / "figures"


def build_summary_table(pert: pd.DataFrame, norm: pd.DataFrame) -> pd.DataFrame:
    # A/C eff rank: constant across (kind, level) within a variant -- one row per (model, variant) is enough.
    ac_effrank = (
        pert.drop_duplicates(["model", "variant"])
        .pivot(index="model", columns="variant", values="orig_eff_rank_ratio")
        .rename(columns={"base": "A_eff_rank_ratio", "norm": "C_eff_rank_ratio"})
    )

    # B/D eff rank stability across the 24 (kind, level) combos.
    bd = pert[pert["kind"] != "ALL"]
    bd_stats = bd.groupby(["model", "variant"])["pert_eff_rank_ratio"].agg(["mean", "std", "min", "max"])
    bd_stats["range"] = bd_stats["max"] - bd_stats["min"]
    bd_wide = bd_stats[["mean", "range"]].unstack("variant")
    bd_wide.columns = [f"{'B' if v == 'base' else 'D'}_eff_rank_ratio_{stat}" for stat, v in bd_wide.columns]

    # A-vs-B (variant=base) and C-vs-D (variant=norm) similarity, pooled over all kind/level ("ALL" row).
    ab_cd = (
        pert[pert["kind"] == "ALL"]
        .set_index(["model", "variant"])[["cos_sim_mean", "cka_linear"]]
        .unstack("variant")
    )
    ab_cd.columns = [f"{'AB' if v == 'base' else 'CD'}_{m}" for m, v in ab_cd.columns]

    # A-vs-C similarity (Exp 0006's normalization axis, already exactly this comparison).
    ac_sim = norm.set_index("model")[["cos_sim_mean", "cka_linear"]].rename(
        columns={"cos_sim_mean": "AC_cos_sim_mean", "cka_linear": "AC_cka_linear"}
    )

    table = ac_effrank.join(bd_wide).join(ab_cd).join(ac_sim)
    return table.reset_index()


def plot_ac_effrank(table: pd.DataFrame, out_path: Path) -> None:
    """A vs. C effective-rank ratio per model -- does stain normalization
    itself change how many effective dimensions the (unperturbed)
    representation occupies?
    """
    df = table.sort_values("A_eff_rank_ratio")
    y = range(len(df))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1.5))
    ax.hlines(y, df["A_eff_rank_ratio"], df["C_eff_rank_ratio"], color="lightgray", linewidth=1, zorder=1)
    ax.scatter(df["A_eff_rank_ratio"], y, color="tab:blue", label="A (original)", zorder=2, s=28)
    ax.scatter(df["C_eff_rank_ratio"], y, color="tab:green", label="C (stain-normalized)", zorder=2, s=28)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"], fontsize=8)
    ax.set_xlabel("effective_rank / D")
    ax.set_title("Effective rank of unperturbed representations: A vs. C")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bd_stability(pert: pd.DataFrame, out_path: Path) -> None:
    """Spread of pert_eff_rank_ratio across the 24 (kind, level) combos, one
    box per model, split by variant (B=base, D=norm). A tall box/long
    whiskers means the effective rank actually swings a lot depending on
    which perturbation was applied; a thin line means it's stable, which
    is what the request was checking for.
    """
    bd = pert[pert["kind"] != "ALL"]
    models = sorted(bd["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.28 * len(models) + 2), sharex=True)
    for ax, variant, label in zip(axes, ["base", "norm"], ["B (base + perturbation)", "D (norm + perturbation)"]):
        data = [bd[(bd["model"] == m) & (bd["variant"] == variant)]["pert_eff_rank_ratio"].to_numpy() for m in models]
        ax.boxplot(data, vert=False, tick_labels=models, showfliers=False, widths=0.6)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("effective_rank / D")
    axes[0].tick_params(axis="y", labelsize=7)
    fig.suptitle("Effective-rank stability across perturbation kind/level")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    pert = pd.read_parquet(RUN_DIR / "metrics_perturbation_axis.parquet")
    norm = pd.read_parquet(RUN_DIR / "metrics_normalization_axis.parquet")

    table = build_summary_table(pert, norm)
    table_path = RUN_DIR / "abcd_summary.csv"
    table.round(4).to_csv(table_path, index=False)
    print(f"Wrote {table_path}")
    print(table.round(3).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_ac_effrank(table, OUT_DIR / "abcd_ac_effrank.png")
    plot_bd_stability(pert, OUT_DIR / "abcd_bd_effrank_stability.png")
    print(f"Wrote figures -> {OUT_DIR}")
