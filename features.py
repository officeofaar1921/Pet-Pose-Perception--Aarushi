"""Feature extraction for the Pet Pose Perception pipeline.

The frozen submission artifact expects a 3,498-dimensional vector:
2,916 local HOG descriptors + 576 compact appearance descriptors + 6 box features.
"""
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog

FEATURE_DIM = 3498
CLASSES = ["sitting", "standing", "lying"]


def read_yolo_box(label_path: str):
    values = [float(x) for x in Path(label_path).read_text().split()]
    if len(values) < 5:
        raise ValueError(f"Invalid YOLO label: {label_path}")
    _, cx, cy, bw, bh = values[:5]
    return cx, cy, bw, bh


def crop_from_box(image, box):
    h, w = image.shape[:2]
    cx, cy, bw, bh = box
    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def appearance_features(crop):
    """Return the fixed 3,492-dimensional appearance vector."""
    crop80 = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_AREA)
    gray80 = cv2.cvtColor(crop80, cv2.COLOR_BGR2GRAY)
    local_hog = hog(
        gray80,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    ).astype(np.float32)

    # 576 compact appearance values: 24x24 grayscale after local contrast normalization.
    # Kept separate from HOG to make the feature contract explicit.
    gray24 = cv2.resize(gray80, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    compact = gray24.flatten()

    return np.concatenate([local_hog, compact]).astype(np.float32)


def make_features(image, box):
    crop = crop_from_box(image, box)
    app = appearance_features(crop)
    cx, cy, bw, bh = box
    geom = np.array(
        [bw, bh, bw / (bh + 1e-8), bw * bh, cx, cy], dtype=np.float32
    )
    x = np.concatenate([app, geom]).astype(np.float32)
    if x.shape[0] != FEATURE_DIM:
        raise RuntimeError(f"Feature dimension {x.shape[0]} != {FEATURE_DIM}")
    return x
