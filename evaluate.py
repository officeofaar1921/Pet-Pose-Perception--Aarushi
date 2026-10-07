"""Evaluate the fixed 30-image test set."""
import argparse
from pathlib import Path
import csv
import joblib
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.features import CLASSES, make_features, read_yolo_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="models/model.pt")
    ap.add_argument("--csv", default="results/test_predictions.csv")
    args = ap.parse_args()
    bundle = joblib.load(args.weights)
    model = bundle["model"]
    rows, y_true, y_pred = [], [], []
    root = Path(args.data)
    for ci, cls in enumerate(CLASSES):
        for lp in sorted((root / cls / "test" / "labels").glob("*.txt")):
            ip = root / cls / "test" / "images" / (lp.stem + ".jpg")
            image = cv2.imread(str(ip))
            x = make_features(image, read_yolo_box(lp)).reshape(1, -1)
            probs = model.predict_proba(x)[0]
            pred = int(np.argmax(probs))
            y_true.append(ci); y_pred.append(pred)
            rows.append((str(ip), cls, CLASSES[pred], float(probs[pred])))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["image","ground_truth","predicted_class","confidence_score"]); w.writerows(rows)

if __name__ == "__main__":
    main()
