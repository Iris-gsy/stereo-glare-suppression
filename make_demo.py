"""Create a deterministic stereo pair with known translation and added glare."""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from inputs import write_png


def scene():
    h, w = 240, 400
    rng = np.random.default_rng(413)
    texture = cv.GaussianBlur(rng.normal(0, 22, (h, w)).astype(np.float32), (0, 0), 0.8)
    yy, xx = np.mgrid[:h, :w]
    base = 42 + 0.22 * yy + 9 * np.sin(xx / 7) + 8 * np.cos((xx + yy) / 6) + texture
    for x in range(16, w, 32):
        cv.line(base, (x, 0), (x + 8, h - 1), 115, 2)
    r = np.hypot(xx - 200, yy - 118)
    lamp = 220 * np.exp(-0.5 * (r / 6) ** 2)
    halo = 115 * np.exp(-0.5 * (r / 20) ** 2)
    normal = np.clip(base + lamp + halo, 0, 255).astype(np.uint8)
    attenuated = np.clip((base + lamp) * 0.43, 0, 255).astype(np.uint8)
    dark = cv.warpAffine(attenuated, np.float32([[1, 0, 6], [0, 1, 2]]), (w, h))
    return cv.cvtColor(normal, cv.COLOR_GRAY2BGR), cv.cvtColor(dark, cv.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("samples/synthetic"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    normal, dark = scene()
    write_png(args.out / "normal.png", normal)
    write_png(args.out / "dark.png", dark)
    write_png(args.out / "packed.png", np.hstack((normal, dark)))


if __name__ == "__main__":
    main()
