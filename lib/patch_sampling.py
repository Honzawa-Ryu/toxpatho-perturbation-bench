"""Tissue-aware patch sampling from WSIs, shared by Task A/B/C experiments."""

from __future__ import annotations

import numpy as np
import tiffslide
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


OD_TISSUE_THRESHOLD = 0.15  # kept equal to lib.stain_norm.OD_BACKGROUND_THRESHOLD
MIN_TISSUE_FRACTION = 0.2

# Patches that Macenko normalization (Exp 0005) cannot process, excluded from
# evaluation so the raw and stain-normalized variants share one gallery and
# one query set. 22 of them are blank glass (tissue fraction <= 0.0009 against
# a 5th percentile of 0.79 over the 20,000 sampled patches); the 23rd,
# 34069_97792_4352, is a featureless grey field that only goes blank under
# color_jitter level 3. Deriving the list: grep "Failed to normalize" in
# outputs/0005_20260831_stain_normalize/*/experiment.log.
EXCLUDED_PATCH_IDS = frozenset(
    {
        "20088_0_28928", "20088_18176_27904", "20088_31488_29440", "20088_3328_27392",
        "33924_8704_30976", "34069_97792_4352",
        "35349_12544_30720", "35349_2048_31744", "35349_3072_31232", "35349_37120_30976",
        "44173_1024_40448", "44173_31488_40192", "44173_40448_42240", "44173_5376_38656",
        "53068_33024_28160", "55434_18688_30720",
        "64050_31232_27392", "64050_46336_27136", "64050_47872_27392",
        "64062_33280_27904", "64062_3328_28416", "64062_33792_28416", "64062_49152_29184",
    }
)


def patch_id(wsi_id: str, x: int, y: int) -> str:
    return f"{wsi_id}_{x}_{y}"


def tissue_fraction(img: Image.Image) -> float:
    """Fraction of pixels that are tissue rather than glass, at full resolution.

    A pixel counts as tissue once its optical density magnitude reaches
    OD_TISSUE_THRESHOLD -- the same test lib/stain_norm.py uses to choose the
    pixels it estimates stain vectors from. Sharing one criterion is the
    point: a patch accepted here then has, by construction, pixels for
    Macenko to work with.

    The thumbnail Otsu mask does not guarantee that on its own. At mpp 0.5 a
    256px patch covers only ~10x10 mask pixels, so a locally mis-thresholded
    patch of glass -- a shadow or a haze on a slide whose global Otsu
    threshold looks perfectly normal -- can reach 80% "tissue" occupancy
    while holding no tissue at all. That is how EXCLUDED_PATCH_IDS got in.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    od = -np.log10(np.clip(rgb, 1e-6, 1.0))
    return float((np.linalg.norm(od, axis=-1) >= OD_TISSUE_THRESHOLD).mean())


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
    n_patches: int | None,
    tissue_threshold: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample up to n_patches non-overlapping (x, y) level-0 coordinates.

    Candidates are drawn from a non-overlapping grid (spacing = one patch at the
    target level) and accepted if their tissue occupancy in the thumbnail mask
    is >= tissue_threshold. Candidate order is shuffled deterministically via rng.
    n_patches=None returns every accepted candidate in that order, for callers
    that apply a further check and need replacements to draw from.
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
        if n_patches is not None and len(accepted) >= n_patches:
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
