"""CPU-friendly video inference.

Required CLI:
python inference_video.py --video path/to/video.mp4 --weights path/to/model.pt --output path/to/predictions.csv
"""
import argparse, csv
from collections import Counter, deque
from pathlib import Path
import joblib, cv2, numpy as np
from src.features import CLASSES, FEATURE_DIM, make_features


def proposal(frame, previous_gray):
    """Return a coarse dog ROI. Motion is preferred; center fallback avoids a heavyweight detector."""
    h,w=frame.shape[:2]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if previous_gray is not None:
        diff=cv2.absdiff(previous_gray,gray)
        diff=cv2.GaussianBlur(diff,(5,5),0)
        _,mask=cv2.threshold(diff,22,255,cv2.THRESH_BINARY)
        kernel=np.ones((7,7),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
        mask=cv2.dilate(mask,kernel,iterations=1)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c=max(contours,key=cv2.contourArea)
            x,y,bw,bh=cv2.boundingRect(c)
            if bw*bh > 0.015*w*h:
                pad=int(0.35*max(bw,bh)); x1=max(0,x-pad); y1=max(0,y-pad); x2=min(w,x+bw+pad); y2=min(h,y+bh+pad)
                return frame[y1:y2,x1:x2], (x1/w,y1/h,(x2-x1)/w,(y2-y1)/h), gray
    # center crop: large enough for typical centered pet videos
    side=int(min(h,w)*0.85); x1=(w-side)//2; y1=(h-side)//2
    return frame[y1:y1+side,x1:x1+side], (x1/w,y1/h,side/w,side/h), gray


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True)
    ap.add_argument("--weights",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    bundle=joblib.load(args.weights); model=bundle["model"]
    threshold=float(bundle.get("confidence_threshold",0.52)); margin_thr=float(bundle.get("margin_threshold",0.12))
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {args.video}")
    history=deque(maxlen=7); previous=None; rows=[]; frame_no=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame_no += 1
        roi,box,previous=proposal(frame,previous)
        try:
            # box is a normalized ROI approximation; make_features crops again, so pass full ROI box.
            feat=make_features(roi,(0.5,0.5,1.0,1.0)).reshape(1,-1)
            if feat.shape[1] != FEATURE_DIM: raise RuntimeError("Feature shape mismatch")
            probs=model.predict_proba(feat)[0]
            order=np.argsort(probs)[::-1]; best=int(order[0]); conf=float(probs[best]); margin=float(probs[order[0]]-probs[order[1]])
            if conf < threshold or margin < margin_thr:
                label="unknown"
            else:
                history.append(best)
                # Require persistence; retain prior state through one-frame noise.
                counts=Counter(history); stable,count=counts.most_common(1)[0]
                label=CLASSES[stable] if count >= 3 else CLASSES[best]
        except Exception:
            label="unknown"; conf=0.0
        rows.append((frame_no,label,round(conf,6)))
    cap.release()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["frame_number","predicted_class","confidence_score"]); w.writerows(rows)

if __name__=="__main__": main()
