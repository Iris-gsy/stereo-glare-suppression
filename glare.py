"""Flow-constrained fusion in normal-exposure coordinates (S102-S106)."""

from dataclasses import dataclass, fields
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from dis_helpers import _create_dis, _resize_for_work, _upscale_flow
from legacy_preprocess import prepare_matching_pair


@dataclass(frozen=True)
class Config:
    max_work_size: int = 960
    preset: str = "medium"
    flow_sigma: float = 0.65
    matching_mode: str = "gain"
    max_vertical: float = 0.0
    fb_threshold: float = 1.5
    structure_window: int = 9
    min_std: float = 2.0
    confidence_threshold: float = 0.8
    core_threshold: int = 230
    signal_threshold: int = 8
    nonsaturated_max: int = 240
    halo_radius: int = 21
    halo_threshold: float = 20.0
    gate_sigma: float = 3.0
    gain_min_samples: int = 64
    dark_enter: float = 8.0
    dark_exit: float = 4.0
    dark_smoothing: float = 0.75
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    ordinary_weight: float = 0.0
    glare_weight: float = 0.9
    auxiliary_max: float = 0.25
    weight_feather_sigma: float = 0.0
    pyramid_levels: int = 3
    highlight_rolloff: float = 0.32
    fusion_drop_limit: float = 38.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            expected = type(field.default)
            if expected is int and type(value) is not int:
                raise ValueError(f"{field.name} must be an integer")
            if expected is float and (type(value) not in (int, float) or not np.isfinite(value)):
                raise ValueError(f"{field.name} must be finite")
        if self.preset not in ("medium", "fast", "ultrafast"):
            raise ValueError("Unknown DIS preset")
        if self.matching_mode not in ("gain", "histmatch_clahe"):
            raise ValueError("Unknown matching mode")
        if self.max_work_size < 64 or not 1 <= self.pyramid_levels <= 8:
            raise ValueError("Invalid work size or pyramid depth")
        if self.structure_window < 3 or self.structure_window % 2 == 0:
            raise ValueError("structure_window must be an odd integer >= 3")
        if not 0 <= self.signal_threshold < self.nonsaturated_max <= 255:
            raise ValueError("Invalid brightness sample range")
        if not 0 < self.core_threshold <= 255 or self.halo_radius < 1:
            raise ValueError("Invalid light mask parameters")
        if not 0 <= self.ordinary_weight <= self.glare_weight <= 1:
            raise ValueError("Invalid fusion weight range")
        if not 0 <= self.highlight_rolloff <= 0.5 or not 0 < self.fusion_drop_limit <= 255:
            raise ValueError("Invalid highlight rolloff or fusion drop limit")
        if not 0 < self.confidence_threshold <= 1 or not 0 <= self.auxiliary_max <= 1:
            raise ValueError("Invalid confidence threshold or auxiliary cap")
        if not 0 <= self.dark_smoothing < 1 or not 0 <= self.dark_exit < self.dark_enter:
            raise ValueError("Invalid dark-side temporal parameters")
        if min(self.fb_threshold, self.min_std, self.gate_sigma, self.clahe_clip) <= 0:
            raise ValueError("Flow, contrast and gate scales must be positive")
        if min(self.halo_threshold, self.flow_sigma, self.max_vertical, self.weight_feather_sigma) < 0 or min(self.gain_min_samples, self.clahe_grid) < 1:
            raise ValueError("Invalid halo, smoothing or enhancement parameters")

    @classmethod
    def load(cls, path: Path | None):
        return cls(**json.loads(path.read_text(encoding="utf-8-sig"))) if path else cls()


def gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)


def coordinates(shape):
    height, width = shape
    return np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))


def resample(image, x, y):
    return cv.remap(image, x, y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)


def valid_bilinear(mask, x, y):
    """Require all four pixels of the 2x2 sampling neighborhood."""
    height, width = mask.shape
    inside = (x >= 0) & (y >= 0) & (x < width - 1) & (y < height - 1)
    iy = np.floor(np.where(inside, y, 0)).astype(np.intp)
    ix = np.floor(np.where(inside, x, 0)).astype(np.intp)
    return inside & mask[iy, ix] & mask[iy + 1, ix] & mask[iy, ix + 1] & mask[iy + 1, ix + 1]


def structure_confidence(reference, moving, valid, config):
    def mean(image):
        return cv.boxFilter(image, -1, (config.structure_window,) * 2, borderType=cv.BORDER_CONSTANT)

    support = mean(valid.astype(np.float32))
    divisor = np.maximum(support, 1e-6)
    a, b = reference * valid, moving * valid
    ma, mb = mean(a) / divisor, mean(b) / divisor
    va = np.maximum(mean(a * a) / divisor - ma * ma, 0)
    vb = np.maximum(mean(b * b) / divisor - mb * mb, 0)
    covariance = mean(a * b) / divisor - ma * mb
    corr = covariance / np.sqrt(np.maximum(va * vb, 1e-6))
    sufficient = valid & (support >= 0.75) & (np.minimum(va, vb) >= config.min_std ** 2)
    return np.where(sufficient, np.clip((corr + 1) / 2, 0, 1), 0).astype(np.float32)


def flow_reliability(reference, moving, left_mask, right_mask, forward, backward, config):
    gx, gy = coordinates(reference.shape)
    x, y = gx + forward[..., 0], gy + forward[..., 1]
    valid = left_mask & valid_bilinear(right_mask, x, y)
    reverse = resample(backward, x, y)
    delta = forward + reverse
    error = cv.magnitude(delta[..., 0], delta[..., 1])
    occluded = valid & (~np.isfinite(error) | (error > config.fb_threshold))
    confidence = structure_confidence(reference, resample(moving, x, y), valid, config)
    return x, y, valid, occluded, confidence, np.where(valid, error, np.nan)


def glare_masks(normal, dark, valid, reliable, config):
    core = valid & (normal >= config.core_threshold) & (dark > config.signal_threshold)
    diameter = 2 * config.halo_radius + 1
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter))
    candidate = cv.dilate(core.astype(np.uint8), kernel).astype(bool) & valid & ~core
    pairs = reliable & ~core & (np.minimum(normal, dark) > config.signal_threshold)
    pairs &= np.maximum(normal, dark) < config.nonsaturated_max
    count = int(np.count_nonzero(pairs))
    gain = float(np.median(normal[pairs] / dark[pairs])) if count >= config.gain_min_samples else None
    # Without reliable gain samples, do not infer a halo from an uncorrected exposure difference.
    residual = normal - gain * dark if gain is not None else np.zeros_like(normal)
    halo = candidate & (residual > config.halo_threshold) if gain is not None else np.zeros_like(core)
    combined = (core | halo).astype(np.float32)
    gate = cv.GaussianBlur(combined, (0, 0), config.gate_sigma)
    return core, candidate, halo, np.where(valid, gate, 0), gain, count, residual


def fusion_weights(gate, confidence, valid, occluded, enhance, config):
    support = valid & ~occluded
    quality = np.clip(confidence / config.confidence_threshold, 0, 1)
    group = (config.ordinary_weight + (config.glare_weight - config.ordinary_weight) * gate) * quality
    group = np.where(support, group, 0).astype(np.float32)
    if config.weight_feather_sigma:
        # Taper isolated high weights without letting smoothing increase local contributions.
        group = np.minimum(group, cv.GaussianBlur(group, (0, 0), config.weight_feather_sigma))
    auxiliary = group * config.auxiliary_max * (1 - gate) if enhance else np.zeros_like(group)
    return np.stack((1 - group, group - auxiliary, auxiliary), axis=-1)


def pyramid_fuse(images, weights, levels):
    gaussian = [[image.astype(np.float32) / 255 for image in images]]
    weight_levels = [weights]
    for _ in range(levels - 1):
        if min(weight_levels[-1].shape[:2]) < 4:
            break
        gaussian.append([cv.pyrDown(image) for image in gaussian[-1]])
        weight_levels.append(cv.pyrDown(weight_levels[-1]))

    result = None
    for level in reversed(range(len(weight_levels))):
        weights_at_level = weight_levels[level]
        weights_at_level = weights_at_level / np.maximum(weights_at_level.sum(axis=-1, keepdims=True), 1e-6)
        images_at_level = gaussian[level]
        size = images_at_level[0].shape[1::-1]
        if result is not None:
            images_at_level = [fine - cv.pyrUp(coarse, dstsize=size)
                               for fine, coarse in zip(images_at_level, gaussian[level + 1])]
        mixed = sum(image * weights_at_level[..., i, None] for i, image in enumerate(images_at_level))
        result = mixed if result is None else mixed + cv.pyrUp(result, dstsize=size)
    return np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)


def render_highlights(normal, fused, rolloff, drop_limit):
    if rolloff == 0:
        return fused

    normal = normal.astype(np.float32)
    fused = fused.astype(np.float32)
    normal_y = cv.cvtColor(normal, cv.COLOR_BGR2GRAY)
    fused_y = cv.cvtColor(fused, cv.COLOR_BGR2GRAY)

    # Limit dark patches from locally strong fusion while keeping small changes.
    drop = np.maximum(normal_y - fused_y, 0)
    blend = 1 / np.sqrt(1 + (drop / drop_limit) ** 2)
    output = normal + (fused - normal) * blend[..., None]
    luma = normal_y + (fused_y - normal_y) * blend

    # A wide, softly varying shoulder around bright lights also works at fallback pixels.
    strength = np.clip((normal_y - 208) / 40, 0, 1)
    strength *= strength * (3 - 2 * strength)
    h, w = strength.shape
    small = cv.resize(strength, (max(1, w // 8), max(1, h // 8)), interpolation=cv.INTER_AREA)
    glow = cv.GaussianBlur(small, (0, 0), max(0.5, min(small.shape) * 0.024))
    glow = cv.resize(glow, (w, h), interpolation=cv.INTER_LINEAR)
    glow = np.clip(glow * 1.5, 0, 1)
    knee = 150 - 40 * glow
    excess = np.maximum(luma - knee, 0)
    compression = rolloff * excess ** 2 / (255 - knee)
    output *= (1 - compression / np.maximum(luma, 1))[..., None]
    return np.rint(np.clip(output, 0, 255)).astype(np.uint8)


class GlareSuppressor:
    def __init__(self, config: Config):
        self.config = config
        self.forward = _create_dis(config.preset)
        self.backward = _create_dis(config.preset)
        self.clahe = cv.createCLAHE(config.clahe_clip, (config.clahe_grid,) * 2)
        self.dark_difference = None
        self.dark_side = False

    def process(self, normal, dark, masks=None):
        c = self.config
        if (normal.shape != dark.shape or normal.dtype != np.uint8 or dark.dtype != np.uint8
                or normal.ndim != 3 or normal.shape[2] != 3 or min(normal.shape[:2]) < 32
                or max(normal.shape[:2]) >= 32767):
            raise ValueError("Expected equal-sized uint8 BGR frames (32 to 32766 px per dimension)")
        ng, dg = gray(normal), gray(dark)
        if masks is None:
            left_mask = right_mask = np.ones(ng.shape, dtype=bool)
        else:
            left_mask, right_mask = (np.asarray(mask, dtype=bool) for mask in masks)
            if left_mask.shape != ng.shape or right_mask.shape != dg.shape:
                raise ValueError("Input masks must match the frame dimensions")

        def usable(image, mask):
            return mask & (image > c.signal_threshold) & (image < c.nonsaturated_max)

        ln, rd = ng[usable(ng, left_mask)], dg[usable(dg, right_mask)]
        initial_gain = float(ln.mean() / rd.mean()) if min(ln.size, rd.size) >= c.gain_min_samples else 1.0
        if c.matching_mode == "histmatch_clahe":
            match_n, match_d = prepare_matching_pair(normal, dark, clahe_clip=c.clahe_clip, clahe_grid=c.clahe_grid)
        else:
            match_n = cv.GaussianBlur(ng.astype(np.uint8), (3, 3), 0)
            match_d = cv.GaussianBlur(np.rint(np.clip(initial_gain * dg, 0, 255)).astype(np.uint8), (3, 3), 0)
        small_n, small_d, sx, sy = _resize_for_work(match_n, match_d, c.max_work_size)
        height, width = ng.shape

        def estimate(dis, a, b):
            flow = dis.calc(a, b, None)
            flow = _upscale_flow(flow, width, height, sx, sy)
            if c.max_vertical:
                flow[..., 1] = np.clip(flow[..., 1], -c.max_vertical, c.max_vertical)
            return cv.GaussianBlur(flow, (0, 0), c.flow_sigma) if c.flow_sigma else flow

        forward, backward = estimate(self.forward, small_n, small_d), estimate(self.backward, small_d, small_n)
        x, y, valid, occluded, confidence, fb = flow_reliability(
            match_n.astype(np.float32), match_d.astype(np.float32), left_mask, right_mask, forward, backward, c)
        aligned = resample(dark, x, y)
        aligned[~valid] = normal[~valid]
        aligned_gray = gray(aligned)
        reliable = valid & ~occluded & (confidence >= c.confidence_threshold)
        core, candidate, halo, gate, gain, count, residual = glare_masks(ng, aligned_gray, valid, reliable, c)
        auxiliary = self.enhance(aligned, ng, aligned_gray, valid & ~occluded)
        weights = fusion_weights(gate, confidence, valid, occluded, self.dark_side, c)
        raw_fused = pyramid_fuse((normal, aligned, auxiliary), weights, c.pyramid_levels)
        # Coarse pyramid weights can spread past hard validity boundaries.
        fallback = ~valid | occluded | (confidence <= 0)
        raw_fused[fallback] = normal[fallback]
        fused = render_highlights(normal, raw_fused, c.highlight_rolloff, c.fusion_drop_limit)
        return dict(fused=fused, raw_fused=raw_fused, aligned=aligned, auxiliary=auxiliary, valid=valid, occluded=occluded,
                    confidence=confidence, reliable=reliable, core=core, candidate=candidate, halo=halo,
                    gate=gate, weights=weights, forward=forward, backward=backward, fb_error=fb,
                    residual=residual, initial_gain=initial_gain, gain=gain, gain_samples=count,
                    dark_side=self.dark_side, dark_difference=self.dark_difference)

    def enhance(self, aligned, ng, dg, support):
        c = self.config
        h, w = ng.shape
        center = np.s_[h // 5:h - h // 5, w // 5:w - w // 5]
        use = support[center] & (ng[center] > 0) & (dg[center] > 0)
        if np.count_nonzero(use) < c.gain_min_samples:
            self.dark_difference = None
            self.dark_side = False
        else:
            difference = float((ng[center][use] - dg[center][use]).mean())
            old = self.dark_difference
            self.dark_difference = difference if old is None else c.dark_smoothing * old + (1 - c.dark_smoothing) * difference
            self.dark_side = self.dark_difference > (c.dark_exit if self.dark_side else c.dark_enter)
        if not self.dark_side:
            return aligned.copy()
        lab = cv.cvtColor(cv.medianBlur(aligned, 3), cv.COLOR_BGR2LAB)
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        enhanced = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        enhanced[~support] = aligned[~support]
        return enhanced
