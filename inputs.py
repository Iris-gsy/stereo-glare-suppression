"""Packed-frame splitting, stereo calibration and image I/O (S101)."""

import json

import cv2 as cv
import numpy as np

from glare import resample, valid_bilinear


def read_image(path, flag=cv.IMREAD_COLOR):
    image = cv.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def write_png(path, image):
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    success, data = cv.imencode(".png", image)
    if not success:
        raise OSError(f"Cannot encode {path}")
    data.tofile(path)


def build_maps(data):
    size = tuple(data["image_size"])
    k1, d1, k2, d2 = (np.asarray(data[key], dtype=np.float64) for key in
                       ("left_camera_matrix", "left_distortion", "right_camera_matrix", "right_distortion"))
    if all(key in data for key in ("R1", "R2", "P1", "P2")):
        r1, r2, p1, p2 = (np.asarray(data[key], dtype=np.float64) for key in ("R1", "R2", "P1", "P2"))
    else:
        r1, r2, p1, p2, *_ = cv.stereoRectify(k1, d1, k2, d2, size,
                                            np.asarray(data["R"], dtype=np.float64),
                                            np.asarray(data["T"], dtype=np.float64).reshape(3, 1), alpha=0)
    return (*cv.initUndistortRectifyMap(k1, d1, r1, p1, size, cv.CV_32FC1),
            *cv.initUndistortRectifyMap(k2, d2, r2, p2, size, cv.CV_32FC1))


class StereoInput:
    def __init__(self, calibration=None, crop_config=None, normal_side="left", mask_paths=None):
        self.normal_side = normal_side
        self.crops = None
        if crop_config:
            data = json.loads(crop_config.read_text(encoding="utf-8-sig"))
            self.crops = [tuple(data[key][field] for field in ("x", "y", "width", "height"))
                          for key in ("left_crop", "right_crop")]
        self.maps = None
        if calibration:
            if calibration.suffix.lower() == ".npz":
                with np.load(calibration, allow_pickle=False) as data:
                    self.maps = tuple(data[k].astype(np.float32) for k in ("map1x", "map1y", "map2x", "map2y"))
            else:
                self.maps = build_maps(json.loads(calibration.read_text(encoding="utf-8-sig")))
            if len({m.shape for m in self.maps}) != 1 or self.maps[0].ndim != 2:
                raise ValueError("Rectification maps must have identical 2-D dimensions")
        self.source_masks = tuple(read_image(p, cv.IMREAD_GRAYSCALE) > 0 for p in mask_paths) if mask_paths else None
        self.cached_shape = None
        self.output_masks = None

    def split(self, packed):
        h, w = packed.shape[:2]
        if self.crops is None:
            if w % 2:
                raise ValueError("Packed image needs even width or explicit crop regions")
            return packed[:, :w // 2], packed[:, w // 2:]
        frames = []
        for x, y, cw, ch in self.crops:
            if not (0 <= x < x + cw <= w and 0 <= y < y + ch <= h):
                raise ValueError(f"Crop {(x, y, cw, ch)} exceeds packed image {w}x{h}")
            frames.append(packed[y:y + ch, x:x + cw])
        return frames

    def prepare(self, left, right):
        if left.shape != right.shape or min(left.shape[:2]) < 32:
            raise ValueError("The two views must be equal-sized and at least 32x32")
        shape = left.shape[:2]
        if self.cached_shape != shape:
            masks = self.source_masks if self.source_masks is not None else (np.ones(shape, dtype=bool),) * 2
            if any(mask.shape != shape for mask in masks):
                raise ValueError("Input validity masks must have the view dimensions")
            if self.maps is not None:
                if self.maps[0].shape != shape:
                    raise ValueError("Calibration resolution does not match the input view")
                m1x, m1y, m2x, m2y = self.maps
                masks = (valid_bilinear(masks[0], m1x, m1y), valid_bilinear(masks[1], m2x, m2y))
            self.output_masks = masks
            self.cached_shape = shape
        if self.maps is not None:
            m1x, m1y, m2x, m2y = self.maps
            left, right = resample(left, m1x, m1y), resample(right, m2x, m2y)
        if self.normal_side == "right":
            return right, left, self.output_masks[::-1]
        return left, right, self.output_masks
