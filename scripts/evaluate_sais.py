"""Evaluate SAIS and emit paper-style calibration/error analysis."""
import argparse, json
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from models.sais import SAISBaseline
from data import ArthroscopyDataset, collate_fn
from utils.metrics import SkillMetrics

@torch.no_grad()
def main(args):
    cfg=yaml.safe_load(Path(args.config).read_text()); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=ArthroscopyDataset(args.data,'val',args.config); loader=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=False,collate_fn=collate_fn,num_workers=0)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False); model=SAISBaseline(cfg,stride=ckpt.get('stride',1)).to(device); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    yt=[]; yp=[]; ys=[]; ps=[]; probs=[]; ids=[]
    for b in loader:
        o=model(b['appearance'].to(device),b['motion'].to(device)); pr=torch.softmax(o['cls_logits'],-1).cpu().numpy()
        yt.extend(b['label_cls'].tolist()); yp.extend(pr.argmax(1).tolist()); probs.extend(pr.tolist()); ys.extend(b['label_reg'][:,6].tolist()); ps.extend(o['total_score'].squeeze(1).cpu().tolist()); ids.extend(b['clip_ids'])
    yt,yp,ys,ps=np.asarray(yt),np.asarray(yp),np.asarray(ys),np.asarray(ps); probs=np.asarray(probs)
    metrics=SkillMetrics.compute_all(yt,yp,ys,ps,probs); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    result={'metrics':metrics,'per_sample':[{'clip_id':i,'true_cls':int(t),'pred_cls':int(p),'true_grs':float(s),'pred_grs':float(q),'confidence':float(max(c)),'high_confidence_error':bool(t!=p and max(c)>=.8),'error_span':int(abs(t-p))} for i,t,p,s,q,c in zip(ids,yt,yp,ys,ps,probs)]}
    (out/'sais_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); main(p.parse_args())
