"""Deterministic perturbation registry for Task A (perturbation invariance).

Each kind has 3 severity levels. `apply_perturbation` is deterministic given
(patch_id, kind, level, seed) so perturbed images are reproducible.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.color import hed2rgb, rgb2hed

PERTURBATION_LEVELS: dict[str, dict[int, float]] = {
    "rotation": {1: 15, 2: 45, 3: 90},          # degrees
    "color_jitter": {1: 0.1, 2: 0.25, 3: 0.4},   # brightness/contrast/saturation factor
    "stain_jitter": {1: 0.1, 2: 0.3, 3: 0.5},  # HED-space alpha/beta magnitude
    "jpeg_compression": {1: 90, 2: 50, 3: 20},   # JPEG quality (lower = more compressed)
    "gaussian_blur": {1: 0.5, 2: 1.5, 3: 3.0},   # sigma in px
    "gaussian_noise": {1: 5, 2: 15, 3: 30},      # std, 0-255 scale
    "downsample_mpp": {1: 0.75, 2: 0.5, 3: 0.25},  # downscale factor before upscaling back
    "occlusion": {1: 0.1, 2: 0.25, 3: 0.4},      # masked area fraction
}


def _rng_for(patch_id: str, kind: str, level: int, seed: int) -> np.random.Generator:
    key = f"{patch_id}|{kind}|{level}|{seed}".encode()
    digest = hashlib.sha256(key).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "little"))


def _rotation(img: Image.Image, param: float, rng: np.random.Generator) -> Image.Image:
    angle = float(rng.uniform(-param, param))
    size = img.size
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    # A naive center-crop back to `size` can land outside the rotated content
    # (black corners) once the angle gets large, e.g. 45deg on a square. Crop
    # to the largest axis-aligned square still fully inside the rotated image
    # instead, then resize back up -- this stays artifact-free for any angle.
    theta = np.deg2rad(angle)
    inscribed_side = size[0] / (abs(np.cos(theta)) + abs(np.sin(theta)))
    left = (rotated.width - inscribed_side) / 2
    top = (rotated.height - inscribed_side) / 2
    cropped = rotated.crop((left, top, left + inscribed_side, top + inscribed_side))
    return cropped.resize(size, resample=Image.BICUBIC)


def _color_jitter(img: Image.Image, param: float, rng: np.random.Generator) -> Image.Image:
    out = img
    for Enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = float(rng.uniform(1 - param, 1 + param))
        out = Enhancer(out).enhance(factor)
    return out


def _stain_jitter(img: Image.Image, param: float, rng: np.random.Generator) -> Image.Image:
    # skimage's rgb2hed values are small and image-dependent (e.g. std ~0.01-0.02
    # for a typical H&E patch), so a fixed additive beta blows out the color at
    # any reasonable `param`. Scale beta by each channel's own std instead, and
    # only jitter H/E (channel 2, "D", is unused residual for H&E and jittering
    # it just injects color-cast noise).
    arr = np.asarray(img).astype(np.float64) / 255.0
    hed = rgb2hed(arr)
    out = hed.copy()
    for c in (0, 1):
        std = hed[..., c].std() + 1e-8
        alpha = 1 + rng.uniform(-param, param)
        beta = rng.uniform(-param, param) * std
        out[..., c] = hed[..., c] * alpha + beta
    rgb = hed2rgb(out)
    rgb = np.clip(rgb, 0, 1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def _jpeg_compression(img: Image.Image, param: float, rng: np.random.Generator) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(param))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _gaussian_blur(img: Image.Image, param: float, rng: np.random.Generator) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=param))


def _gaussian_noise(img: Image.Image, param: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img).astype(np.float64)
    noise = rng.normal(0, param, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255)
    return Image.fromarray(noisy.astype(np.uint8))


def _downsample_mpp(img: Image.Image, param: float, rng: np.random.Generator) -> Image.Image:
    size = img.size
    small_size = (max(1, int(size[0] * param)), max(1, int(size[1] * param)))
    small = img.resize(small_size, resample=Image.BICUBIC)
    return small.resize(size, resample=Image.BICUBIC)


def _occlusion(img: Image.Image, param: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    side = int((param ** 0.5) * min(h, w))
    x0 = int(rng.integers(0, max(1, w - side + 1)))
    y0 = int(rng.integers(0, max(1, h - side + 1)))
    arr[y0 : y0 + side, x0 : x0 + side] = 255  # occlude with a solid (glass-like) fill
    return Image.fromarray(arr)


_PERTURBATION_FNS = {
    "rotation": _rotation,
    "color_jitter": _color_jitter,
    "stain_jitter": _stain_jitter,
    "jpeg_compression": _jpeg_compression,
    "gaussian_blur": _gaussian_blur,
    "gaussian_noise": _gaussian_noise,
    "downsample_mpp": _downsample_mpp,
    "occlusion": _occlusion,
}


def apply_perturbation(
    img: Image.Image, patch_id: str, kind: str, level: int, seed: int = 42
) -> Image.Image:
    param = PERTURBATION_LEVELS[kind][level]
    rng = _rng_for(patch_id, kind, level, seed)
    return _PERTURBATION_FNS[kind](img, param, rng)
