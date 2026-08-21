import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
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
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.perturbations import PERTURBATION_LEVELS, apply_perturbation

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    source_exp: str = config["source_exp"]
    source_variant: str = config["source_variant"]

    variant_key = source_variant
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, seed=seed, source_exp=source_exp)

    # Upstream outputs always live under project_root/outputs (NFS), regardless
    # of this job's own OUTPUT_ROOT -- see USAGE.md 2-3 on the completed-guard
    # location vs. scratch staging.
    source_dir = project_root / "outputs" / source_exp / source_variant
    source_manifest_path = source_dir / "manifest.parquet"
    source_patches_dir = source_dir / "patches"

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"source_dir:  {source_dir}")
    logger.info(f"kinds:       {list(PERTURBATION_LEVELS.keys())}")

    source_manifest = pd.read_parquet(source_manifest_path)
    logger.info(f"Loaded {len(source_manifest)} source patches from {source_manifest_path}")

    perturbed_dir = run_dir / "perturbed"
    manifest_path = run_dir / "manifest.parquet"

    rows: list[dict] = []
    n_failed = 0
    total = len(source_manifest) * sum(len(levels) for levels in PERTURBATION_LEVELS.values())

    for i, source_row in enumerate(source_manifest.itertuples()):
        patch_id = source_row.patch_id
        img_path = source_patches_dir / f"{patch_id}.png"
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            logger.exception(f"Failed to load source patch {img_path}, skipping all its perturbations")
            n_failed += sum(len(levels) for levels in PERTURBATION_LEVELS.values())
            continue

        for kind, levels in PERTURBATION_LEVELS.items():
            for level in levels:
                try:
                    out_img = apply_perturbation(img, patch_id, kind, level, seed=seed)
                    out_dir = perturbed_dir / kind / str(level)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{patch_id}.png"
                    out_img.save(out_path)
                    rows.append(
                        {
                            "patch_id": patch_id,
                            "kind": kind,
                            "level": level,
                            "output_path": str(out_path.relative_to(run_dir)),
                        }
                    )
                except Exception:
                    n_failed += 1
                    logger.exception(f"Failed: patch_id={patch_id} kind={kind} level={level}")

        if (i + 1) % 200 == 0:
            logger.info(f"[{i + 1}/{len(source_manifest)}] patches processed, {len(rows)} perturbations written so far")
            pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    df = pd.DataFrame(rows)
    df.to_parquet(manifest_path, index=False)

    logger.info(
        f"Done: {len(df)}/{total} perturbations written ({n_failed} failures) -> {manifest_path}"
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
