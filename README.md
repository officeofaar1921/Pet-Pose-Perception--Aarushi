# Pet Pose Perception — Ogmen Robotics Preliminary Engineering Task

Lightweight CPU-oriented dog pose perception pipeline for **sitting, standing, lying**, with an `unknown` state and temporal smoothing for video.

## Repository structure

```text
pet-pose-perception/
├── train.py
├── evaluate.py
├── inference_video.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   └── features.py
├── models/
│   └── model.pt
├── results/
│   ├── metrics.txt
│   └── test_predictions.csv
└── design_document/
    └── pet_pose_perception_design.pdf
```

## Python 3.10

Create a clean environment with Python 3.10:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

The supplied dataset is expected in this structure:

```text
dataset/
├── sitting/{train,test}/{images,labels}
├── standing/{train,test}/{images,labels}
└── lying/{train,test}/{images,labels}
```

The YOLO labels contain a dog bounding box followed by pose/keypoint annotations. The directory name is the pose target.

## Train

```bash
python train.py --data path/to/dataset --output models/model.pt
```

## Evaluate the fixed test set

```bash
python evaluate.py --data path/to/dataset --weights models/model.pt --csv results/test_predictions.csv
```

## Required video inference CLI

```bash
python inference_video.py --video path/to/video.mp4 --weights path/to/model.pt --output path/to/predictions.csv
```

Output headers are exactly:

```text
frame_number,predicted_class,confidence_score
```

`predicted_class` is restricted to `sitting`, `standing`, `lying`, or `unknown`.

## Robotics considerations

- **Open-set handling:** predictions below the confidence threshold or with a small top-1/top-2 margin become `unknown`.
- **Temporal noise:** a short majority-history filter reduces one-frame flicker.
- **CPU efficiency:** the deployed model is an Extra-Trees classifier over compact HOG/appearance features; it avoids a heavyweight neural detector during grading.
- **ROI proposal:** motion-based ROI is used when available, with a center-crop fallback for stationary scenes.

## Supplied evaluation

The supplied 30-image test set was evaluated before submission. The recorded result is **70.00% (21/30)**; see `results/metrics.txt` and the design PDF for details.

## Important note about `model.pt`

The supplied final artifact is a compressed joblib model package saved with the assignment-requested `.pt` filename. It is loaded by the included Python scripts with `joblib.load()`.
