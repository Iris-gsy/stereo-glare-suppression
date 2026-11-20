# Reused from binoculars/stereo_dense_flow_alignment/stereo_align/flow_aligner.py.
import cv2
import numpy as np


def _create_dis(preset_name: str):
    presets = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    if preset_name not in presets:
        raise ValueError(f"Unsupported DIS preset: {preset_name}")

    if hasattr(cv2, "DISOpticalFlow_create"):
        dis = cv2.DISOpticalFlow_create(presets[preset_name])
    elif hasattr(cv2, "DISOpticalFlow") and hasattr(cv2.DISOpticalFlow, "create"):
        dis = cv2.DISOpticalFlow.create(presets[preset_name])
    else:
        raise RuntimeError(
            "This OpenCV build does not provide DISOpticalFlow. "
            "Install/upgrade opencv-python."
        )

    # These settings improve robustness across different camera exposures.
    if hasattr(dis, "setUseMeanNormalization"):
        dis.setUseMeanNormalization(True)
    if hasattr(dis, "setUseSpatialPropagation"):
        dis.setUseSpatialPropagation(True)
    if hasattr(dis, "setVariationalRefinementIterations"):
        dis.setVariationalRefinementIterations(5)
    return dis


def _resize_for_work(
    left: np.ndarray,
    right: np.ndarray,
    max_work_size: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    height, width = left.shape[:2]
    if max_work_size <= 0 or max(height, width) <= max_work_size:
        return left, right, 1.0, 1.0

    scale = max_work_size / float(max(height, width))
    work_width = max(16, int(round(width * scale)))
    work_height = max(16, int(round(height * scale)))

    left_small = cv2.resize(
        left,
        (work_width, work_height),
        interpolation=cv2.INTER_AREA,
    )
    right_small = cv2.resize(
        right,
        (work_width, work_height),
        interpolation=cv2.INTER_AREA,
    )
    return (
        left_small,
        right_small,
        work_width / float(width),
        work_height / float(height),
    )


def _upscale_flow(
    flow_small: np.ndarray,
    full_width: int,
    full_height: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    flow_full = cv2.resize(
        flow_small,
        (full_width, full_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    flow_full[..., 0] /= scale_x
    flow_full[..., 1] /= scale_y
    return flow_full
