"""Macenko stain normalization.

Unlike `lib/perturbations.py`'s `_stain_jitter` (which perturbs HED values
using skimage's fixed, generic Ruifrok stain matrix), normalization here
estimates each image's *own* stain vectors via SVD, then re-renders that
image's concentrations using a single fixed reference image's stain vectors.
Tissue structure (concentration) is preserved; only the color basis changes.

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis," ISBI 2009.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

OD_BACKGROUND_THRESHOLD = 0.15  # OD magnitude below this = background, excluded from SVD
ANGLE_PERCENTILE = 1.0  # robust min/max angle percentile (vs. literal min/max)
CONCENTRATION_PERCENTILE = 99.0  # robust max-concentration percentile
MIN_FOREGROUND_PIXELS = 100  # 0.15% of a 256x256 patch; below this the estimate is noise anyway


class InsufficientTissueError(ValueError):
    """An image has too few tissue pixels to estimate stain vectors from."""


def _rgb_to_od(img: Image.Image) -> np.ndarray:
    """Convert an RGB image to optical density, shape (H*W, 3)."""
    img.load()  # ensure img is loaded into memory
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    rgb = np.clip(rgb, 1e-6, 1.0)
    od = -np.log10(rgb)
    return od.reshape(-1, 3)


def estimate_stain_matrix(od: np.ndarray) -> np.ndarray:
    """Estimate a (3, 2) stain matrix [H_vector | E_vector] from OD pixels via SVD.

    Steps (see chat for the full walkthrough):
    1. Drop background pixels (OD magnitude < OD_BACKGROUND_THRESHOLD).
    2. SVD (or eigendecomposition of the covariance) of the remaining OD
       pixels -> take the top-2 singular/eigen vectors spanning the plane.
    3. Project OD pixels onto that plane, compute per-pixel angle (atan2).
    4. Take the ANGLE_PERCENTILE / (100 - ANGLE_PERCENTILE) percentile
       angles (robust extremes, not literal min/max) and map each back to
       a 3D OD direction -> two candidate stain vectors.
    5. Order them [H, E] (H direction should have larger inner product
       with the canonical Ruifrok H vector than the E candidate does).
    """
    od_magnitude = np.linalg.norm(od, axis=1)
    od_fg = od[od_magnitude >= OD_BACKGROUND_THRESHOLD]
    # Without this, a blank (all-background) patch reaches np.cov with 0 or 1
    # rows, which yields a NaN covariance, and np.linalg.eigh then fails with
    # "Eigenvalues did not converge" -- a stack trace that says nothing about
    # the real problem. See lib.patch_sampling.EXCLUDED_PATCH_IDS.
    if len(od_fg) < MIN_FOREGROUND_PIXELS:
        raise InsufficientTissueError(
            f"only {len(od_fg)} of {len(od)} pixels reach OD {OD_BACKGROUND_THRESHOLD}; "
            "too little tissue to estimate stain vectors"
        )
    od_fg_centered = od_fg - od_fg.mean(axis=0, keepdims=True)
    cov = np.cov(od_fg_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, -2:]  # shape (3, 2)
    od_proj = od_fg @ top2  # shape (N_fg, 2)
    angles = np.arctan2(od_proj[:, 1], od_proj[:, 0])  # shape (N_fg,)
    angle_mean = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))  # robust mean angle
    angles_wrapped = np.mod(angles - angle_mean + np.pi, 2*np.pi) - np.pi    
    angle_min = np.percentile(angles_wrapped, ANGLE_PERCENTILE) + angle_mean
    angle_max = np.percentile(angles_wrapped, 100 - ANGLE_PERCENTILE) + angle_mean
    od_min = np.array([np.cos(angle_min), np.sin(angle_min)])
    od_max = np.array([np.cos(angle_max), np.sin(angle_max)])
    stain_candidates = np.stack([od_min, od_max], axis=1)  # shape (2, 2)
    stain_candidates_3d = top2 @ stain_candidates  # shape (3, 2)
    canonical_h = np.array([0.65, 0.70, 0.29])  # Ruifrok H vector
    dots = stain_candidates_3d.T @ canonical_h  # shape (2,)
    if dots[0] >= dots[1]:
        stain_matrix = stain_candidates_3d  # [H, E]
    else:
        stain_matrix = stain_candidates_3d[:, ::-1]  # [E, H] -> swap to [H, E]
    return stain_matrix


def estimate_concentrations(od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Solve for per-pixel [H, E] concentrations given OD and a (3, 2) stain matrix.

    This is a least-squares solve of `od ≈ concentrations @ stain_matrix.T`
    for `concentrations`, shape (N, 2). Concentrations must be >= 0
    (clip after solving; a full non-negative least squares is overkill here).
    """
    od = od.reshape(-1, 3)
    concentrations, residuals, rank, s = np.linalg.lstsq(stain_matrix, od.T, rcond=None)
    concentrations = concentrations.T  # shape (N, 2)
    concentrations = np.clip(concentrations, 0.0, None)
    return concentrations


def fit_reference(reference_img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """One-time fit on a chosen reference patch.

    Returns (stain_matrix, max_concentrations) where max_concentrations is
    the CONCENTRATION_PERCENTILE of each of the 2 channels -- used to scale
    other images' concentrations into this reference's dynamic range before
    re-rendering with the reference's stain vectors.
    """
    ref_od = _rgb_to_od(reference_img)
    ref_stain_matrix = estimate_stain_matrix(ref_od)
    ref_concentrations = estimate_concentrations(ref_od, ref_stain_matrix)
    ref_max_concentrations = np.percentile(ref_concentrations, CONCENTRATION_PERCENTILE, axis=0)
    return ref_stain_matrix, ref_max_concentrations


def macenko_normalize(
    img: Image.Image, ref_stain_matrix: np.ndarray, ref_max_concentrations: np.ndarray
) -> Image.Image:
    """Re-render `img` using the reference's stain vectors.

    1. od = _rgb_to_od(img)
    2. stain_matrix = estimate_stain_matrix(od)      # this image's own vectors
    3. conc = estimate_concentrations(od, stain_matrix)
    4. rescale conc by (this image's own CONCENTRATION_PERCENTILE per channel)
       -> ref_max_concentrations, so dynamic ranges match the reference
    5. od_normalized = conc_rescaled @ ref_stain_matrix.T
    6. rgb = 255 * 10**(-od_normalized)   (invert the OD transform), clip to [0, 255]
    """
    od = _rgb_to_od(img)
    stain_matrix = estimate_stain_matrix(od)
    conc = estimate_concentrations(od, stain_matrix)
    conc_max = np.percentile(conc, CONCENTRATION_PERCENTILE, axis=0)
    conc_rescaled = conc * (ref_max_concentrations / conc_max)
    od_normalized = conc_rescaled @ ref_stain_matrix.T
    rgb_normalized = 255.0 * 10 ** (-od_normalized)
    rgb_normalized = np.clip(rgb_normalized, 0.0, 255.0).astype(np.uint8)
    h, w = img.size[1], img.size[0]
    rgb_normalized = rgb_normalized.reshape(h, w, 3)
    return Image.fromarray(rgb_normalized)