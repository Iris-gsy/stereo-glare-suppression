"""Command-line entry point for images, recorded video and packed stereo capture."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter

import cv2 as cv
import numpy as np

from glare import Config, GlareSuppressor, gray
from inputs import StereoInput, read_image, write_png


def fraction(mask):
    return float(mask.mean())


def diagnostics(path, normal, dark, result, config):
    path.mkdir(parents=True, exist_ok=True)
    for name, image in {"normal": normal, "dark": dark, **result}.items():
        if not isinstance(image, np.ndarray) or image.ndim == 3 and name in ("weights", "forward", "backward"):
            continue
        if name in ("residual", "fb_error"):
            continue
        if image.dtype.kind == "f":
            image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        write_png(path / f"{name}.png", image)
    for i, name in enumerate(("normal", "aligned", "auxiliary")):
        write_png(path / f"weight_{name}.png", np.rint(result["weights"][..., i] * 255).astype(np.uint8))
    data = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    data.update({f"{k}_fraction": fraction(result[k]) for k in ("valid", "occluded", "reliable", "core", "halo")})
    ng, fg = gray(normal), gray(result["fused"])
    raw = gray(result["raw_fused"])
    shadows = ng < 90
    data["tone_metrics"] = {
        "near_white_fraction_at_245": {name: fraction(luma >= 245) for name, luma in
                                       (("normal", ng), ("raw_fused", raw), ("output", fg))},
        "shadow_mean_abs_change_8bit": float(np.abs(fg[shadows] - ng[shadows]).mean()) if shadows.any() else None,
    }
    for name in ("core", "halo"):
        mask = result[name] & result["reliable"]
        data[f"reliable_{name}_mean_drop_8bit"] = float((ng[mask] - fg[mask]).mean()) if mask.any() else None
    data.update(config=asdict(config), opencv=cv.__version__, numpy=np.__version__)
    (path / "report.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(path / "fields.npz", **{k: result[k] for k in
                        ("forward", "backward", "fb_error", "residual", "confidence", "weights")})
    write_png(path / "comparison.png", np.hstack((normal, result["aligned"], result["fused"])))


def open_writer(path, fps, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = {".mp4": "mp4v", ".avi": "MJPG"}.get(path.suffix.lower())
    if not codec:
        raise ValueError("Output video must use .mp4 or .avi")
    if any(n % 2 for n in frame.shape[:2]):
        raise ValueError("Video output needs even width and height; use an even-sized crop")
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*codec), fps, frame.shape[1::-1])
    if not writer.isOpened():
        writer.release()
        raise OSError(f"Cannot create output video {path}")
    return writer


def video_fps(cap):
    fps = cap.get(cv.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS is unavailable")
    return fps


def show(frame):
    cv.imshow("Stereo glare suppression", frame)
    return cv.waitKey(1) & 255 in (27, ord("q"))


def offline_video(args, stereo, process):
    sources = [args.packed] if args.packed else [args.left, args.right]
    if args.video_out.resolve() in [p.resolve() for p in sources]:
        raise ValueError("Output video must differ from the input")
    caps = [cv.VideoCapture(str(p)) for p in sources]
    writer = None
    size = None
    n = 0
    start = perf_counter()
    try:
        if not all(cap.isOpened() for cap in caps):
            raise OSError("Cannot open one or more input videos")
        fps = video_fps(caps[0])
        counts = [int(cap.get(cv.CAP_PROP_FRAME_COUNT)) for cap in caps]
        if len(caps) == 2 and (abs(fps - video_fps(caps[1])) > 0.01 or
                               min(counts) > 0 and counts[0] != counts[1]):
            raise ValueError("Paired videos must have equal FPS and frame count")
        while not args.max_frames or n < args.max_frames:
            data = [cap.read() for cap in caps]
            if not all(ok for ok, _ in data):
                if any(ok for ok, _ in data) or min(counts) > n:
                    raise OSError(f"Video ended unexpectedly at frame {n + 1}")
                break
            images = [frame for _, frame in data]
            views = stereo.split(images[0]) if args.packed else images
            frame = process(views, n)
            if writer is None:
                size = frame.shape
                writer = open_writer(args.video_out, fps, frame)
            if frame.shape != size:
                raise ValueError("Input video resolution changed")
            writer.write(frame)
            n += 1
            if args.show and show(frame):
                break
        if not n:
            raise ValueError("No input frames decoded")
        print(f"Processed {n} frames, {n / max(perf_counter() - start, 1e-6):.2f} processing FPS")
    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv.destroyAllWindows()


def camera_preview(args, stereo, process):
    cap = cv.VideoCapture(args.camera, {"any": cv.CAP_ANY, "msmf": cv.CAP_MSMF, "dshow": cv.CAP_DSHOW}[args.backend])
    raw_writer = None
    if not cap.isOpened():
        cap.release()
        raise OSError(f"Camera {args.camera} cannot be opened")
    for prop, value in ((cv.CAP_PROP_FRAME_WIDTH, args.width), (cv.CAP_PROP_FRAME_HEIGHT, args.height),
                        (cv.CAP_PROP_FPS, args.fps)):
        cap.set(prop, value)

    latest = Queue(maxsize=1)
    stop, done = Event(), Event()
    errors = []
    captured = 0

    def capture():
        nonlocal captured, raw_writer
        size = None
        try:
            while not stop.is_set() and (not args.capture_frames or captured < args.capture_frames):
                ok, frame = cap.read()
                if not ok:
                    raise OSError("Camera frame acquisition failed")
                if frame.shape[1::-1] != (args.width, args.height):
                    raise ValueError(f"Camera returned {frame.shape[1]}x{frame.shape[0]}, expected {args.width}x{args.height}")
                if raw_writer is None and args.raw_out:
                    fps = cap.get(cv.CAP_PROP_FPS)
                    raw_writer = open_writer(args.raw_out, fps if np.isfinite(fps) and fps > 0 else args.fps, frame)
                    size = frame.shape
                if raw_writer is not None:
                    if frame.shape != size:
                        raise ValueError("Camera frame dimensions changed")
                    raw_writer.write(frame)
                captured += 1
                try:
                    latest.put_nowait(frame)
                except Full:
                    try:
                        latest.get_nowait()
                    except Empty:
                        pass
                    latest.put_nowait(frame)
        except Exception as error:
            errors.append(error)
        finally:
            cap.release()
            if raw_writer is not None:
                raw_writer.release()
            done.set()

    worker = Thread(target=capture, name="capture")
    worker.start()
    n = 0
    try:
        while not done.is_set() or not latest.empty():
            try:
                frame = latest.get(timeout=0.1)
            except Empty:
                continue
            fused = process(stereo.split(frame), n)
            n += 1
            if args.show and show(fused):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join()
        if args.show:
            cv.destroyAllWindows()
    if errors:
        raise errors[0]
    print(f"Captured {captured} packed frames; processed {n} preview frames")


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path)
    common.add_argument("--normal-side", choices=("left", "right"), default="left")
    common.add_argument("--calibration", type=Path, help="JSON intrinsics/extrinsics or NPZ maps")
    common.add_argument("--crop-config", type=Path, help="Packed frame crop rectangles in JSON")
    common.add_argument("--mask", nargs=2, type=Path, metavar=("LEFT_MASK", "RIGHT_MASK"))
    common.add_argument("--out", type=Path, required=True, help="Diagnostics directory (first frame for video)")
    root = argparse.ArgumentParser(description=__doc__)
    modes = root.add_subparsers(dest="mode", required=True)
    for name in ("image", "video"):
        mode = modes.add_parser(name, parents=[common])
        source = mode.add_mutually_exclusive_group(required=True)
        source.add_argument("--packed", type=Path)
        source.add_argument("--left", type=Path)
        mode.add_argument("--right", type=Path)
        if name == "video":
            mode.add_argument("--video-out", type=Path, required=True)
            mode.add_argument("--max-frames", type=int, default=0)
            mode.add_argument("--show", action="store_true")
    camera = modes.add_parser("camera", parents=[common])
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--backend", choices=("any", "dshow", "msmf"), default="msmf")
    camera.add_argument("--width", type=int, default=4000)
    camera.add_argument("--height", type=int, default=1200)
    camera.add_argument("--fps", type=float, default=30)
    camera.add_argument("--capture-frames", type=int, default=0)
    camera.add_argument("--raw-out", type=Path)
    camera.add_argument("--no-show", dest="show", action="store_false")
    return root


def main():
    command = parser()
    args = command.parse_args()
    if args.mode in ("image", "video") and bool(args.right) != bool(args.left):
        command.error("--left and --right must be supplied together")
    if args.mode == "video" and args.max_frames < 0:
        command.error("--max-frames must be nonnegative")
    if args.mode == "camera":
        if min(args.width, args.height, args.fps) <= 0 or not np.isfinite(args.fps) or args.capture_frames < 0:
            command.error("Invalid camera dimensions, FPS or frame limit")

    cv.setNumThreads(1)
    cv.ocl.setUseOpenCL(False)
    config = Config.load(args.config)
    stereo = StereoInput(args.calibration, args.crop_config, args.normal_side, args.mask)
    suppressor = GlareSuppressor(config)

    def process(views, index=0):
        normal, dark, mask = stereo.prepare(*views)
        result = suppressor.process(normal, dark, mask)
        if index == 0:
            diagnostics(args.out, normal, dark, result, config)
        return result["fused"]

    if args.mode == "image":
        views = stereo.split(read_image(args.packed)) if args.packed else (read_image(args.left), read_image(args.right))
        process(views)
    elif args.mode == "video":
        offline_video(args, stereo, process)
    else:
        camera_preview(args, stereo, process)
    print(f"Diagnostics: {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, cv.error) as error:
        raise SystemExit(f"Error: {error}") from error
