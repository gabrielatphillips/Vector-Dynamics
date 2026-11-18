"""Offline rear-facing video detector/tracker. EXPERIMENTAL; no distance or warnings.

Requires: pip install ultralytics opencv-python
Run: python vision/rear_camera_pipeline.py --video path/to/rear_video.mp4
Video MUST be recorded from a rear-facing camera; output is NOT radar data.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

COLUMNS = ['frame','time_ms','track_id','class_id','class_name','confidence','x1_px','y1_px','x2_px','y2_px','center_x_px','center_y_px','frame_width_px','frame_height_px','source','range_m','bearing_deg','warning_authorized']
VEHICLES = {2, 3, 5, 7}

def run(video: Path, output: Path, model: str='yolo11n.pt', conf: float=.2, imgsz: int=640, device: str='cpu', max_frames: int=0):
    if not video.is_file():
        raise FileNotFoundError(f'Video not found: {video}')
    if not 0 < conf < 1 or imgsz < 160 or max_frames < 0:
        raise ValueError('Invalid configuration')
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit('Install video dependencies: pip install ultralytics opencv-python') from e
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError('Video FPS unavailable; timestamps cannot be trusted')
    detector = YOLO(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = detections = 0
    try:
        with output.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            while True:
                if max_frames and frames >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                # Sequential frames are essential for ByteTrack persistent identities.
                result = detector.track(frame, persist=True, tracker='bytetrack.yaml', classes=sorted(VEHICLES), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for box in boxes:
                        if box.id is None:
                            continue
                        tid = int(box.id.item())
                        cid = int(box.cls.item())
                        x1,y1,x2,y2 = [float(v) for v in box.xyxy[0].tolist()]
                        writer.writerow({'frame':frames,'time_ms':round(frames*1000/fps),'track_id':tid,'class_id':cid,
                            'class_name':result.names.get(cid,str(cid)),'confidence':round(float(box.conf.item()),5),
                            'x1_px':round(x1,2),'y1_px':round(y1,2),'x2_px':round(x2,2),'y2_px':round(y2,2),
                            'center_x_px':round((x1+x2)/2,2),'center_y_px':round((y1+y2)/2,2),
                            'frame_width_px':w,'frame_height_px':h,'source':'rear_video_unverified',
                            'range_m':'','bearing_deg':'','warning_authorized':'false'})
                        detections += 1
                frames += 1
    finally:
        cap.release()
    print(f'Processed {frames} video frames; {detections} tracked vehicle observations')
    print(f'Output: {output}')
    print('Camera-only pixels, NOT calibrated distance or lane position. NO safety warnings authorized.')
    return frames, detections

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('vision/results/rear_camera_tracks.csv'))
    p.add_argument('--model',default='yolo11n.pt')
    p.add_argument('--conf',type=float,default=.2)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args()
    run(a.video,a.output,a.model,a.conf,a.imgsz,a.device,a.max_frames)
if __name__=='__main__':main()
