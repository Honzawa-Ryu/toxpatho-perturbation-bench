import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
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
    import yaml

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
    from lib.stain_norm import fit_reference, macenko_normalize

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()

    config = load_config(Path(__file__).parent)
    source_exp: str = config["source_exp"]
    source_variant: str = config["source_variant"]
    perturb_exp: str = config["perturb_exp"]
    perturb_variant: str = config["perturb_variant"]
    reference_patch_id: str = config["reference_patch_id"]

    source_dir = project_root / "outputs" / source_exp / source_variant
    perturb_dir_root = project_root / "outputs" / perturb_exp / perturb_variant

    # Fit once on the chosen reference patch; every other patch (original and
    # perturbed alike) is re-rendered using these stain vectors so the whole
    # dataset shares one consistent color basis.
    reference_img = Image.open(source_dir / "patches" / f"{reference_patch_id}.png").convert("RGB")
    ref_stain_matrix, ref_max_concentrations = fit_reference(reference_img)

    # ── Phase A: normalize originals (mirrors Exp0001's patches/ + manifest.parquet) ──
    orig_variant_key = f"{source_variant}_orig"
    orig_run_dir = get_run_dir(project_root, __file__, orig_variant_key, output_root=output_root)
    logger = setup_logger(orig_run_dir, exp_name)
    write_run_metadata(orig_run_dir, exp_name=exp_name, variant_key=orig_variant_key, reference_patch_id=reference_patch_id)

    logger.info(f"Starting: {exp_name} / {orig_variant_key}")
    source_manifest = pd.read_parquet(source_dir / "manifest.parquet")
    logger.info(f"Loaded {len(source_manifest)} source patches from {source_dir / 'manifest.parquet'}")

    orig_patches_dir = orig_run_dir / "patches"
    orig_patches_dir.mkdir(parents=True, exist_ok=True)
    failed_patch_ids: set[str] = set()
    for i, row in enumerate(source_manifest.itertuples()):
        try:
            img = Image.open(source_dir / "patches" / f"{row.patch_id}.png").convert("RGB")
            out_img = macenko_normalize(img, ref_stain_matrix, ref_max_concentrations)
            out_img.save(orig_patches_dir / f"{row.patch_id}.png")
        except Exception:
            failed_patch_ids.add(row.patch_id)
            logger.exception(f"Failed to normalize original patch_id={row.patch_id}")
        if (i + 1) % 2000 == 0:
            logger.info(f"[originals {i + 1}/{len(source_manifest)}] normalized")

    # Drop failed patches from the manifest -- downstream (Exp 0003) trusts
    # every manifest row to have a corresponding image on disk.
    source_manifest = source_manifest[~source_manifest["patch_id"].isin(failed_patch_ids)]
    source_manifest.to_parquet(orig_run_dir / "manifest.parquet", index=False)
    logger.info(
        f"Done originals: {len(source_manifest)}/{len(source_manifest) + len(failed_patch_ids)} normalized ({len(failed_patch_ids)} failures)"
    )
    complete_run(orig_run_dir)

    # ── Phase B: normalize perturbed (mirrors Exp0002's perturbed/{kind}/{level}/ + manifest.parquet) ──
    pert_variant_key = f"{perturb_variant}_pert"
    pert_run_dir = get_run_dir(project_root, __file__, pert_variant_key, output_root=output_root)
    logger = setup_logger(pert_run_dir, exp_name)
    write_run_metadata(pert_run_dir, exp_name=exp_name, variant_key=pert_variant_key, reference_patch_id=reference_patch_id)

    logger.info(f"Starting: {exp_name} / {pert_variant_key}")
    perturb_manifest = pd.read_parquet(perturb_dir_root / "manifest.parquet")
    logger.info(f"Loaded {len(perturb_manifest)} perturbed images from {perturb_dir_root / 'manifest.parquet'}")

    pert_out_dir = pert_run_dir / "perturbed"
    failed_rows: list[int] = []
    for i, row in enumerate(perturb_manifest.itertuples()):
        try:
            img = Image.open(perturb_dir_root / row.output_path).convert("RGB")
            out_img = macenko_normalize(img, ref_stain_matrix, ref_max_concentrations)
            out_path = pert_run_dir / row.output_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(out_path)
        except Exception:
            failed_rows.append(i)
            logger.exception(f"Failed to normalize perturbed patch_id={row.patch_id} kind={row.kind} level={row.level}")
        if (i + 1) % 20000 == 0:
            logger.info(f"[perturbed {i + 1}/{len(perturb_manifest)}] normalized")

    # Drop failed rows (by position -- a single patch_id can fail for one
    # kind/level and succeed for another) so every remaining manifest row
    # has a corresponding image on disk. Also drop any row whose parent
    # patch_id failed in Phase A: Exp0004 retrieves against the originals
    # as gallery, so a perturbed image with no surviving original has no
    # valid target and must not be scored.
    n_total = len(perturb_manifest)
    perturb_manifest = perturb_manifest.drop(perturb_manifest.index[failed_rows])
    perturb_manifest = perturb_manifest[~perturb_manifest["patch_id"].isin(failed_patch_ids)]
    perturb_manifest = perturb_manifest.reset_index(drop=True)
    perturb_manifest.to_parquet(pert_run_dir / "manifest.parquet", index=False)
    logger.info(f"Done perturbed: {len(perturb_manifest)}/{n_total} normalized/kept ({n_total - len(perturb_manifest)} dropped)")
    complete_run(pert_run_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
