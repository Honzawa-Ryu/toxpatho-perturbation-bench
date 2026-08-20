"""Tissue-aware patch sampling from WSIs, shared by Task A/B/C experiments."""

from __future__ import annotations

import numpy as np
import tiffslide
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def patch_id(wsi_id: str, x: int, y: int) -> str:
    return f"{wsi_id}_{x}_{y}"


def pick_level_for_mpp(slide: tiffslide.TiffSlide, mpp_target: float) -> tuple[int, float]:
    """Pick the finest slide level whose mpp does not exceed mpp_target (5% tolerance).

    Falls back to level 0 if even the base level is coarser than mpp_target
    (i.e. the slide was scanned at lower magnification than requested).
    """
    base_mpp_x = slide.properties.get("tiffslide.mpp-x") or slide.properties.get("openslide.mpp-x")
    if base_mpp_x is None:
        raise ValueError("Slide has no mpp-x property; cannot resolve target level.")
    base_mpp = float(base_mpp_x)

    best_level, best_mpp = 0, base_mpp
    for level, downsample in enumerate(slide.level_downsamples):
        level_mpp = base_mpp * downsample
        if level_mpp <= mpp_target * 1.05:
            best_level, best_mpp = level, level_mpp
    return best_level, best_mpp


def tissue_mask_from_thumbnail(
    slide: tiffslide.TiffSlide, thumb_max_side: int = 1024
) -> tuple[np.ndarray, float]:
    """Otsu-threshold a low-res thumbnail into a tissue(True)/background(False) mask.

    Returns (mask, scale) where scale = level0_pixels / thumbnail_pixels, so that a
    level-0 coordinate maps to the mask via `coord / scale`.
    """
    w0, h0 = slide.dimensions
    scale = max(w0, h0) / thumb_max_side
    thumb = slide.get_thumbnail((thumb_max_side, max(1, int(h0 / scale))))
    gray = rgb2gray(np.array(thumb.convert("RGB")))
    threshold = threshold_otsu(gray)
    # H&E tissue is stained (darker) relative to the near-white glass background.
    mask = gray < threshold
    return mask, scale


def sample_patch_coords(
    slide: tiffslide.TiffSlide,
    mask: np.ndarray,
    mask_scale: float,
    level: int,
    patch_size_px: int,
    n_patches: int,
    tissue_threshold: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample up to n_patches non-overlapping (x, y) level-0 coordinates.

    Candidates are drawn from a non-overlapping grid (spacing = one patch at the
    target level) and accepted if their tissue occupancy in the thumbnail mask
    is >= tissue_threshold. Candidate order is shuffled deterministically via rng.
    """
    downsample = slide.level_downsamples[level]
    step_level0 = int(round(patch_size_px * downsample))

    w0, h0 = slide.dimensions
    xs = list(range(0, max(1, w0 - step_level0), step_level0))
    ys = list(range(0, max(1, h0 - step_level0), step_level0))
    candidates = [(x, y) for y in ys for x in xs]
    if not candidates:
        return []

    order = rng.permutation(len(candidates))
    accepted: list[tuple[int, int]] = []
    for idx in order:
        if len(accepted) >= n_patches:
            break
        x, y = candidates[int(idx)]
        mx0, my0 = int(x / mask_scale), int(y / mask_scale)
        mx1, my1 = int((x + step_level0) / mask_scale), int((y + step_level0) / mask_scale)
        tile = mask[my0:my1, mx0:mx1]
        if tile.size == 0:
            continue
        if tile.mean() >= tissue_threshold:
            accepted.append((x, y))
    return accepted


def extract_patch(
    slide: tiffslide.TiffSlide, x: int, y: int, level: int, patch_size_px: int
) -> Image.Image:
    """Read a patch_size_px x patch_size_px RGB crop at (x, y) (level-0 coords)."""
    region = slide.read_region((x, y), level, (patch_size_px, patch_size_px))
    return region.convert("RGB")
