from __future__ import annotations

import cv2
import numpy as np


def histogram_match_u8(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Match an 8-bit grayscale source histogram to a template histogram."""
    if source.dtype != np.uint8 or template.dtype != np.uint8:
        raise ValueError("histogram_match_u8 expects uint8 grayscale images")

    src_values, src_indices, src_counts = np.unique(
        source.ravel(),
        return_inverse=True,
        return_counts=True,
    )
    tmpl_values, tmpl_counts = np.unique(
        template.ravel(),
        return_counts=True,
    )

    src_quantiles = np.cumsum(src_counts, dtype=np.float64)
    src_quantiles /= src_quantiles[-1]

    tmpl_quantiles = np.cumsum(tmpl_counts, dtype=np.float64)
    tmpl_quantiles /= tmpl_quantiles[-1]

    mapped_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return mapped_values[src_indices].reshape(source.shape).astype(np.uint8)


def _apply_clahe(gray: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size)),
    )
    return clahe.apply(gray)


def _add_gradient_structure(gray: np.ndarray, weight: float) -> np.ndarray:
    """Blend image intensity with gradient magnitude to emphasize edges."""
    if weight <= 0:
        return gray

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    high = float(np.percentile(magnitude, 99.0))
    if high <= 1e-6:
        return gray

    gradient_u8 = np.clip(magnitude * (255.0 / high), 0, 255).astype(np.uint8)
    return cv2.addWeighted(gray, 1.0 - weight, gradient_u8, weight, 0.0)


def prepare_matching_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    mode: str = "histmatch_clahe",
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    gradient_weight: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare two single-channel uint8 images for optical-flow matching.

    The returned images are used only to estimate displacement. The final
    warp is always applied to the original color right image.
    """
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    if mode == "gray":
        left_match = left_gray
        right_match = right_gray
    elif mode == "clahe":
        left_match = _apply_clahe(left_gray, clahe_clip, clahe_grid)
        right_match = _apply_clahe(right_gray, clahe_clip, clahe_grid)
    elif mode == "histmatch_clahe":
        right_matched = histogram_match_u8(right_gray, left_gray)
        left_match = _apply_clahe(left_gray, clahe_clip, clahe_grid)
        right_match = _apply_clahe(right_matched, clahe_clip, clahe_grid)
    else:
        raise ValueError(
            f"Unknown photometric mode: {mode}. "
            "Choose gray, clahe, or histmatch_clahe."
        )

    # A small blur suppresses sensor/compression noise before flow estimation.
    left_match = cv2.GaussianBlur(left_match, (3, 3), 0)
    right_match = cv2.GaussianBlur(right_match, (3, 3), 0)

    left_match = _add_gradient_structure(left_match, gradient_weight)
    right_match = _add_gradient_structure(right_match, gradient_weight)
    return left_match, right_match
