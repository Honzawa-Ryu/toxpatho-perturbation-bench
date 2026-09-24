"""How much of the embedding neighbourhood structure is study identity?

notebooks/cluster_ari.py found that stain normalization makes the embeddings
cluster *less* stably. Two readings fit that: normalization removed a
per-slide shortcut (section 10-3's hypothesis), or it simply removed variance
and left less structure of any kind. They differ in what happens to study
identity specifically, which is what this measures.

For each patch, the fraction of its k nearest neighbours (cosine, the same
similarity Exp 0004 retrieves with) carrying the same label. No clustering,
so no seed-to-seed floor to clear -- the failure mode that made cluster_ari's
ARI values uninterpretable.

Labels come from TG-GATEs (data/corrected/open_tggates_pathological_image.csv,
joined on the .svs basename in FILE_LOCATION, which is our wsi_id):

  EXP_ID            study -- what "between-study difference" means here
  COMPOUND_NAME     the biology a study is about
  DOSE, SACRIFICE_PERIOD, SINGLE_REPEAT_TYPE
  wsi_id            slide, the finest level and cluster_ari's original target

EXP_ID mixes batch and biology, but 261 studies span 144 compounds, so the
two come apart: neighbourhoods tracking EXP_ID far more tightly than
COMPOUND_NAME is a batch effect rather than a compound signal.

Purity is reported against the chance level a label's own group sizes imply
(sum of squared group proportions); 0.05 means nothing until you know that
EXP_ID's chance level is ~0.004. The ratio of the two is what to read.

Reads only the unperturbed rows (20k of each file's 500k), so this is a
fraction of the I/O the other notebooks jobs do. Submit with
notebooks/study_knn_purity.sbatch.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.patch_sampling import EXCLUDED_PATCH_IDS  # noqa: E402

# Only for discover_completed_models: several model directories exist holding
# just a completion.json from a run that never produced embeddings (the gated
# models of section 10-6), and the status check is what skips them.
_spec = importlib.util.spec_from_file_location(
    "exp0006", PROJECT_ROOT / "experiments/0006_20260904_eval_representation_shift/experiment.py"
)
exp0006 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp0006)

EMBED_DIR = PROJECT_ROOT / "outputs/0003_20260820_extract_patch_embeddings"
META_CSV = PROJECT_ROOT / "data/corrected/open_tggates_pathological_image.csv"
OUT_DIR = PROJECT_ROOT / "outputs/0006_20260904_eval_representation_shift/256px_mpp0.5_n1000x20"

VARIANTS = {"base": "256px_mpp0.5_n1000x20", "norm": "256px_mpp0.5_n1000x20_orig"}
K_VALUES = [5, 10, 25, 50, 100]
LABEL_COLS = ["EXP_ID", "COMPOUND_NAME", "DOSE", "SACRIFICE_PERIOD", "SINGLE_REPEAT_TYPE", "wsi_id"]
CHUNK = 2048

# Unconditional purity cannot separate these levels, because they nest: two
# patches from one slide are necessarily from one study and one compound, so
# a slide-level effect shows up unchanged at every coarser label. Each pair
# below asks the outcome question only of neighbours that already differ on
# the finer label -- same study but a *different* slide, same compound but a
# *different* study -- which is the part the finer effect cannot produce.
CONDITIONAL = [
    ("EXP_ID", "wsi_id"),
    ("COMPOUND_NAME", "EXP_ID"),
    ("SACRIFICE_PERIOD", "EXP_ID"),
    ("DOSE", "EXP_ID"),
]


def load_originals(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(N, D) embeddings of the unperturbed patches, plus their patch ids."""
    table = pq.read_table(model_dir / "embeddings.parquet", filters=[("source_type", "==", "original")])
    ids = table.column("parent_patch_id").to_pandas().to_numpy()
    col = table.column("embedding")
    dim = len(col.chunk(0)[0])
    emb = np.empty((table.num_rows, dim), dtype=np.float32)
    offset = 0
    for chunk in col.chunks:
        n = len(chunk)
        emb[offset : offset + n] = chunk.values.to_numpy(zero_copy_only=False).reshape(n, dim)
        offset += n
    keep = ~np.isin(ids, list(EXCLUDED_PATCH_IDS))
    return emb[keep], ids[keep]


def knn_indices(emb: np.ndarray, k_max: int) -> np.ndarray:
    """(N, k_max) indices of each row's nearest neighbours by cosine, self excluded."""
    x = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    out = np.empty((len(x), k_max), dtype=np.int32)
    for start in range(0, len(x), CHUNK):
        sims = x[start : start + CHUNK] @ x.T
        rows = np.arange(len(sims))
        sims[rows, start + rows] = -np.inf  # a patch is not its own neighbour
        part = np.argpartition(-sims, k_max, axis=1)[:, :k_max]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        out[start : start + CHUNK] = np.take_along_axis(part, order, axis=1)
    return out


def _pair_fraction(counts: np.ndarray, total_pairs: float) -> float:
    return float((counts * (counts - 1) / 2).sum() / total_pairs)


def conditional_chance(outcome: np.ndarray, condition: np.ndarray) -> float:
    """P(same outcome | different condition) for a uniformly random patch pair.

    Derived from group sizes rather than the nesting these labels happen to
    have, so it stays correct if a compound ever spans studies unevenly:
    P(same out, diff cond) = P(same out) - P(same out and same cond).
    """
    n = len(outcome)
    total_pairs = n * (n - 1) / 2
    df = pd.DataFrame({"o": outcome, "c": condition})
    p_out = _pair_fraction(df["o"].value_counts().to_numpy(), total_pairs)
    p_cond = _pair_fraction(df["c"].value_counts().to_numpy(), total_pairs)
    p_both = _pair_fraction(df.groupby(["o", "c"]).size().to_numpy(), total_pairs)
    return (p_out - p_both) / (1 - p_cond)


def load_labels(ids: np.ndarray) -> pd.DataFrame:
    meta = pd.read_csv(META_CSV)
    meta["wsi_id"] = meta["FILE_LOCATION"].str.rsplit("/", n=1).str[-1].str.replace(".svs", "", regex=False)
    meta = meta.drop_duplicates("wsi_id").set_index("wsi_id")
    wsi = pd.Index([pid.rsplit("_", 2)[0] for pid in ids])
    labels = meta.reindex(wsi)[LABEL_COLS[:-1]].reset_index(drop=True)
    labels["wsi_id"] = wsi.to_numpy()
    if labels.isna().any().any():
        raise ValueError("some patches did not join to TG-GATEs metadata")
    return labels


def main() -> None:
    models = sorted(
        set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['base']}"))
        & set(exp0006.discover_completed_models(EMBED_DIR, f"__{VARIANTS['norm']}"))
    )
    print(f"{len(models)} models: {models}", flush=True)

    out_path = OUT_DIR / "study_knn_purity.csv"
    rows = []
    for model in models:
        # The norm variant is written without the 17 originals Macenko could
        # not process, so the two variants only line up once EXCLUDED_PATCH_IDS
        # is dropped from both. Check rather than assume: comparing purity
        # across variants is only fair on one identical patch set.
        loaded = {v: load_originals(EMBED_DIR / f"{model}__{s}") for v, s in VARIANTS.items()}
        id_sets = {v: ids for v, (_, ids) in loaded.items()}
        if not np.array_equal(np.sort(id_sets["base"]), np.sort(id_sets["norm"])):
            raise ValueError(f"{model}: base and norm cover different patches")

        for variant, (emb, ids) in loaded.items():
            labels = load_labels(ids)
            nn = knn_indices(emb, max(K_VALUES))
            print(f"[{model}/{variant}] {emb.shape} knn done", flush=True)

            for col in LABEL_COLS:
                lab = labels[col].to_numpy()
                # Probability two patches drawn at random share a label: the
                # purity a neighbourhood carrying no information would show.
                p = pd.Series(lab).value_counts(normalize=True).to_numpy()
                chance = float((p**2).sum())
                same = lab[nn] == lab[:, None]
                for k in K_VALUES:
                    purity = float(same[:, :k].mean())
                    rows.append({"model": model, "variant": variant, "label": col, "k": k,
                                 "purity": purity, "chance": chance, "enrichment": purity / chance})

            for out_col, cond_col in CONDITIONAL:
                out = labels[out_col].to_numpy()
                cond = labels[cond_col].to_numpy()
                same_out = out[nn] == out[:, None]
                diff_cond = cond[nn] != cond[:, None]
                chance = conditional_chance(out, cond)
                for k in K_VALUES:
                    eligible = diff_cond[:, :k]
                    purity = float(same_out[:, :k][eligible].mean())
                    rows.append({"model": model, "variant": variant,
                                 "label": f"{out_col}|not_{cond_col}", "k": k,
                                 "purity": purity, "chance": chance, "enrichment": purity / chance,
                                 "n_eligible_pairs": int(eligible.sum())})
            del nn
        del loaded
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nWrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
