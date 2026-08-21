import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiffslide
import yaml


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
    from lib.patch_sampling import (
        extract_patch,
        pick_level_for_mpp,
        sample_patch_coords,
        tissue_mask_from_thumbnail,
    )

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    n_wsi: int = config["n_wsi"]
    patches_per_wsi: int = config["patches_per_wsi"]
    patch_size_px: int = config["patch_size_px"]
    mpp_target: float = config["mpp_target"]
    tissue_threshold: float = config["tissue_threshold"]
    thumb_size: int = config.get("thumb_size", 1024)

    variant_key = f"{patch_size_px}px_mpp{mpp_target}_n{n_wsi}x{patches_per_wsi}"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(
        run_dir,
        exp_name=exp_name,
        variant_key=variant_key,
        seed=seed,
        n_wsi=n_wsi,
        patches_per_wsi=patches_per_wsi,
    )

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"config:      {config}")

    patches_dir = run_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.parquet"

    all_svs = sorted((dataset_dir / "raw_wsi").glob("*.svs"))
    if not all_svs:
        logger.error(f"No .svs files found under {dataset_dir / 'raw_wsi'}")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    selected = list(rng.choice(all_svs, size=min(n_wsi, len(all_svs)), replace=False))
    logger.info(f"Selected {len(selected)}/{len(all_svs)} WSIs (seed={seed})")

    rows: list[dict] = []
    n_failed_wsi = 0

    for i, svs_path in enumerate(selected):
        wsi_id = svs_path.stem
        try:
            slide = tiffslide.TiffSlide(str(svs_path))
            level, level_mpp = pick_level_for_mpp(slide, mpp_target)
            mask, mask_scale = tissue_mask_from_thumbnail(slide, thumb_max_side=thumb_size)
            coords = sample_patch_coords(
                slide=slide,
                mask=mask,
                mask_scale=mask_scale,
                level=level,
                patch_size_px=patch_size_px,
                n_patches=patches_per_wsi,
                tissue_threshold=tissue_threshold,
                rng=rng,
            )

            for x, y in coords:
                pid = f"{wsi_id}_{x}_{y}"
                img = extract_patch(slide, x, y, level, patch_size_px)
                img.save(patches_dir / f"{pid}.png")
                rows.append(
                    {
                        "patch_id": pid,
                        "wsi_id": wsi_id,
                        "x": x,
                        "y": y,
                        "level": level,
                        "mpp": level_mpp,
                        "patch_size_px": patch_size_px,
                    }
                )

            slide.close()
            logger.info(f"[{i + 1}/{len(selected)}] {wsi_id}: {len(coords)} patches (level={level}, mpp={level_mpp:.4f})")

        except Exception:
            n_failed_wsi += 1
            logger.exception(f"[{i + 1}/{len(selected)}] {wsi_id}: FAILED, skipping")

        # Periodic checkpoint so partial progress survives a timeout/interruption.
        if rows and (i + 1) % 10 == 0:
            pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    df = pd.DataFrame(rows)
    df.to_parquet(manifest_path, index=False)

    logger.info(
        f"Done: {len(df)} patches from {len(selected) - n_failed_wsi}/{len(selected)} WSIs "
        f"({n_failed_wsi} WSI failures) -> {manifest_path}"
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
