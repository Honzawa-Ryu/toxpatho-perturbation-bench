import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image


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
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def build_record_list(
    originals_df: pd.DataFrame,
    perturbed_df: pd.DataFrame,
    source_patches_dir: Path,
    perturbed_dir: Path,
) -> list[dict]:
    records: list[dict] = []
    for row in originals_df.itertuples():
        records.append(
            {
                "embedding_id": row.patch_id,
                "parent_patch_id": row.patch_id,
                "source_type": "original",
                "kind": None,
                "level": None,
                "image_path": source_patches_dir / f"{row.patch_id}.png",
            }
        )
    for row in perturbed_df.itertuples():
        records.append(
            {
                "embedding_id": f"{row.patch_id}__{row.kind}__L{row.level}",
                "parent_patch_id": row.patch_id,
                "source_type": "perturbed",
                "kind": row.kind,
                "level": int(row.level),
                "image_path": perturbed_dir / row.output_path,
            }
        )
    return records


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from trident.patch_encoder_models.load import encoder_factory

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    model_name = args.model

    config = load_config(Path(__file__).parent, args.config)
    seed: int = config.get("seed", 42)
    source_exp: str = config["source_exp"]
    source_variant: str = config["source_variant"]
    perturb_exp: str = config["perturb_exp"]
    perturb_variant: str = config["perturb_variant"]
    batch_size: int = config.get("batch_size", 128)

    variant_key = f"{model_name}__{source_variant}"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, seed=seed, model=model_name)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:  {run_dir}")

    # Upstream outputs always live under project_root/outputs (NFS), regardless
    # of this job's own OUTPUT_ROOT.
    source_dir = project_root / "outputs" / source_exp / source_variant
    perturb_dir_root = project_root / "outputs" / perturb_exp / perturb_variant

    originals_df = pd.read_parquet(source_dir / "manifest.parquet")
    perturbed_df = pd.read_parquet(perturb_dir_root / "manifest.parquet")
    records = build_record_list(
        originals_df, perturbed_df, source_dir / "patches", perturb_dir_root
    )
    logger.info(
        f"{len(originals_df)} original + {len(perturbed_df)} perturbed = {len(records)} images to encode"
    )

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    encoder = encoder_factory(model_name)
    encoder = encoder.eval().to(device)
    transform = encoder.eval_transforms
    precision = encoder.precision
    logger.info(f"Loaded encoder '{model_name}' (precision={precision})")

    manifest_path = run_dir / "embeddings.parquet"

    # Stream each batch straight to disk as its own row group instead of
    # accumulating every row in a Python list first: at full scale (500k
    # rows/model) a list of dicts holding per-row float lists costs tens of
    # GB of Python-object overhead alone. Explicit float32 for the embedding
    # column also roughly halves file size vs. the float64 pandas would infer
    # from a plain list of Python floats.
    schema = pa.schema(
        [
            ("embedding_id", pa.string()),
            ("parent_patch_id", pa.string()),
            ("source_type", pa.string()),
            ("kind", pa.string()),
            ("level", pa.int64()),
            ("embedding", pa.list_(pa.float32())),
        ]
    )
    writer = pq.ParquetWriter(manifest_path, schema)
    n_written = 0
    last_dim = None

    try:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            imgs = [transform(Image.open(r["image_path"]).convert("RGB")) for r in batch]
            x = torch.stack(imgs).to(device)

            # Mirrors TRIDENT's own inference pattern (trident/wsi_objects/WSI.py):
            # keep model/input weights in fp32 and let autocast handle mixed
            # precision -- casting the model itself to `precision` breaks encoders
            # whose submodules expect fp32 (e.g. some norm layers).
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=precision, enabled=(precision != torch.float32)
            ):
                z = encoder(x)
            z = z.detach().float().cpu().numpy().astype("float32")
            last_dim = z.shape[1]

            table = pa.table(
                {
                    "embedding_id": [r["embedding_id"] for r in batch],
                    "parent_patch_id": [r["parent_patch_id"] for r in batch],
                    "source_type": [r["source_type"] for r in batch],
                    "kind": [r["kind"] for r in batch],
                    "level": [r["level"] for r in batch],
                    "embedding": [emb.tolist() for emb in z],
                },
                schema=schema,
            )
            writer.write_table(table)
            n_written += len(batch)

            done = i + len(batch)
            if done % (batch_size * 20) == 0 or done == len(records):
                logger.info(f"[{done}/{len(records)}] encoded")
    finally:
        writer.close()

    logger.info(f"Done: {n_written} embeddings (dim={last_dim}) -> {manifest_path}")

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
