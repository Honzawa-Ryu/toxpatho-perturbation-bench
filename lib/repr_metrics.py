"""Representation-comparison metrics for a pair of (N, D) embedding sets that
share the same N samples (e.g. an original patch and its perturbed variant,
or the same patch embedded before/after stain normalization).

cosine_sim / mse are per-sample (paired) metrics -- the caller aggregates
(mean/std) over whatever grouping it needs. cka_linear / participation_ratio
operate on the whole (N, D) set at once and are dimension-agnostic, so they
stay comparable across models with different embedding dims D (unlike raw
MSE or an un-normalized effective rank -- see `participation_ratio`'s ratio
return value).
"""

import numpy as np


def paired_cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between matched (N, D) embeddings."""
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a_n * b_n).sum(axis=1)


def paired_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise mean squared error between matched (N, D) embeddings, on the
    raw (un-normalized) embedding scale -- not comparable across models with
    different D or output scale without also using `relative_mse`.
    """
    return ((a - b) ** 2).mean(axis=1)


def relative_mse(mse_mean: float, reference: np.ndarray) -> float:
    """`mse_mean` expressed as a fraction of `reference`'s own per-component
    signal power (mean of squared elements, matching `paired_mse`'s
    per-dimension-mean convention), which cancels each model's arbitrary
    output scale/D and makes the value comparable across models.
    """
    power = float((reference**2).mean())
    return float(mse_mean / power)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019, eq. 4) between two (N, D) sets.

    Computed via the D x D feature covariance rather than the N x N Gram
    matrix -- algebraically equivalent, but tractable at N=20000 where a
    dense Gram matrix would be ~1.6GB and recomputed per (kind, level)
    group. Dimension-agnostic (x and y may have different D) and invariant
    to each set's own isotropic scale/rotation, so it is directly
    comparable across models.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    cross = xc.T @ yc
    hsic_xy = np.linalg.norm(cross, ord="fro") ** 2
    hsic_xx = np.linalg.norm(xc.T @ xc, ord="fro")
    hsic_yy = np.linalg.norm(yc.T @ yc, ord="fro")
    return float(hsic_xy / (hsic_xx * hsic_yy))


def participation_ratio(x: np.ndarray) -> tuple[float, float]:
    """Participation ratio PR = (sum(eigvals))^2 / sum(eigvals^2) of the
    (centered) covariance of an (N, D) set -- an effective-rank measure
    bounded in [1, min(N, D)] (1 = all variance in one direction, D =
    isotropic).

    Returns (pr, pr / D). Raw PR is not comparable across models with
    different D since it is bounded by D; the ratio (fraction of available
    dimensionality actually used) is.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    cov = xc.T @ xc
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)  # guard tiny negative numerical noise
    pr = float(eigvals.sum() ** 2 / (eigvals**2).sum())
    return pr, pr / x.shape[1]
