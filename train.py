"""Train the lightweight Extra-Trees pose classifier.

Usage (Python 3.10):
    python train.py --data /path/to/dataset --output models/model.pt
"""
import argparse
from pathlib import Path
import joblib
import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from src.features import CLASSES, make_features, read_yolo_box


def collect(data_dir, split):
    X, y = [], []
    data_dir = Path(data_dir)
    for class_id, cls in enumerate(CLASSES):
        label_dir = data_dir / cls / split / "labels"
        image_dir = data_dir / cls / split / "images"
        for lp in sorted(label_dir.glob("*.txt")):
            ip = image_dir / (lp.stem + ".jpg")
            image = cv2.imread(str(ip))
            if image is None:
                raise FileNotFoundError(ip)
            X.append(make_features(image, read_yolo_box(lp)))
            y.append(class_id)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="models/model.pt")
    args = ap.parse_args()
    X, y = collect(args.data, "train")
    model = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=12,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes": CLASSES,
            "feature_version": 1,
            "confidence_threshold": 0.52,
            "margin_threshold": 0.12,
        },
        out,
        compress=3,
    )
    print(f"Saved {out} | train samples={len(y)} | features={X.shape[1]}")

if __name__ == "__main__":
    main()
