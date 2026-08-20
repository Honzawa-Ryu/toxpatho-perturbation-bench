import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
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


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
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

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    source_exp: str = config["source_exp"]
    source_variant: str = config["source_variant"]
    perturb_exp: str = config["perturb_exp"]
    perturb_variant: str = config["perturb_variant"]
    batch_size: int = config.get("batch_size", 128)

    variant_key = model_name
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
    out_rows: list[dict] = []

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
        z = z.detach().float().cpu().numpy()

        for r, emb in zip(batch, z):
            out_rows.append(
                {
                    "embedding_id": r["embedding_id"],
                    "parent_patch_id": r["parent_patch_id"],
                    "source_type": r["source_type"],
                    "kind": r["kind"],
                    "level": r["level"],
                    "embedding": emb.tolist(),
                }
            )

        done = i + len(batch)
        if done % (batch_size * 20) == 0 or done == len(records):
            logger.info(f"[{done}/{len(records)}] encoded")
            pd.DataFrame(out_rows).to_parquet(manifest_path, index=False)

    df = pd.DataFrame(out_rows)
    df.to_parquet(manifest_path, index=False)
    logger.info(f"Done: {len(df)} embeddings (dim={len(df['embedding'].iloc[0])}) -> {manifest_path}")

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
