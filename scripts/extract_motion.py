"""Generate the paper's YOLOv8 + DeepSORT motion features during preprocessing.

The command reads a metadata CSV with columns video_id,video_path,split,label_cls,
label_reg (seven normalised values), and optionally centre. It writes one or more
16-frame ``.pt`` clips per video containing frames, motion and detection_conf.
The detector/tracker are deliberately optional imports so the rest of the repo
can be installed and tested without downloading model weights.
"""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm
from models.motion_branch import KinematicFeatureExtractor


def load_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def parse_vec(value, n=7):
    if isinstance(value, (list,tuple)): return [float(x) for x in value]
    vals=[float(x.strip()) for x in str(value).replace(';',',').split(',') if x.strip()]
    if len(vals)!=n: raise ValueError(f'expected {n} values, got {len(vals)}: {value!r}')
    return vals

def extract_tracks(video_path, detector, tracker, extractor, clip_len, size):
    cap=cv2.VideoCapture(str(video_path)); frames=[]; motions=[]; confs=[]; trajectories={}; frame_no=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.resize(frame,(size[1],size[0]),interpolation=cv2.INTER_AREA)
        result=detector(frame, verbose=False)[0]
        detections=[]
        for box in result.boxes:
            xyxy=box.xyxy[0].detach().cpu().numpy().tolist(); score=float(box.conf[0]); cls=int(box.cls[0]) if box.cls is not None else 0
            detections.append(([xyxy[0],xyxy[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]],score,cls))
        tracks=tracker.update_tracks(detections,frame=frame)
        for tr in tracks:
            if not tr.is_confirmed(): continue
            tid=int(tr.track_id); l,t,r,b=tr.to_ltrb(); trajectories.setdefault(tid,[]).append([l,t,r,b])
        feat,_=extractor.extract(trajectories,size)
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); motions.append(feat); confs.append(max((d[1] for d in detections),default=0.0)); frame_no+=1
    cap.release(); return frames,motions,confs

def main(args):
    try:
        from ultralytics import YOLO
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError as e:
        raise SystemExit('Install ultralytics and deep-sort-realtime from requirements.txt before running extraction') from e
    detector=YOLO(args.weights); extractor=KinematicFeatureExtractor(); out=Path(args.output)
    for row in tqdm(load_rows(args.metadata),desc='videos'):
        split=row.get('split','train'); vid=row['video_id']; frames,motion,conf=extract_tracks(row['video_path'],detector,DeepSort(max_age=30,n_init=2),extractor,args.clip_len,(args.height,args.width))
        label_cls=int(row['label_cls']); label_reg=torch.tensor(parse_vec(row['label_reg']),dtype=torch.float32); centre=row.get('centre') or row.get('hospital')
        target=out/split/'clips'; target.mkdir(parents=True,exist_ok=True)
        for start in range(0,len(frames)-args.clip_len+1,args.clip_len):
            payload={'frames':torch.from_numpy(np.stack(frames[start:start+args.clip_len])).contiguous(),'motion':torch.from_numpy(np.stack(motion[start:start+args.clip_len])).float(),'detection_conf':torch.tensor(conf[start:start+args.clip_len],dtype=torch.float32),'label_cls':label_cls,'label_reg':label_reg}
            if centre: payload['centre']=centre
            torch.save(payload,target/f'{vid}_{start:06d}.pt')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--metadata',required=True); p.add_argument('--output',required=True); p.add_argument('--weights',default='yolov8n.pt'); p.add_argument('--clip-len',type=int,default=16); p.add_argument('--height',type=int,default=480); p.add_argument('--width',type=int,default=640); main(p.parse_args())
