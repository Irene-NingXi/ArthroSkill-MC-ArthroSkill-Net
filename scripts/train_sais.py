"""Train the SAIS-style appearance-only baseline.

Usage: python scripts/train_sais.py --config configs/config.yaml --data ./processed --output ./outputs/sais
"""
import argparse, json, random
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from models.sais import SAISBaseline
from data import ArthroscopyDataset, collate_fn


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); yt=[]; yp=[]; ys=[]; ps=[]
    for b in loader:
        o=model(b['appearance'].to(device), b['motion'].to(device))
        yt += b['label_cls'].tolist(); yp += o['cls_logits'].argmax(1).cpu().tolist()
        ys += b['label_reg'][:,6].tolist(); ps += o['total_score'].squeeze(1).cpu().tolist()
    yt,yp,ys,ps=map(np.asarray,(yt,yp,ys,ps))
    return {'accuracy':float((yt==yp).mean()), 'grs_mae':float(np.abs(ys-ps).mean()),
            'pearson_r':float(np.corrcoef(ys,ps)[0,1]) if len(ys)>1 and ys.std()>0 and ps.std()>0 else 0.0}

def main(args):
    cfg=yaml.safe_load(Path(args.config).read_text()); seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() and cfg['training'].get('device')=='auto' else 'cpu')
    train=ArthroscopyDataset(args.data,'train',args.config); val=ArthroscopyDataset(args.data,'val',args.config)
    tr=DataLoader(train,batch_size=cfg['training']['batch_size'],shuffle=True,collate_fn=collate_fn,num_workers=0)
    va=DataLoader(val,batch_size=cfg['training']['batch_size'],shuffle=False,collate_fn=collate_fn,num_workers=0)
    model=SAISBaseline(cfg,stride=args.stride).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); best=-1; history=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        for b in tr:
            o=model(b['appearance'].to(device),b['motion'].to(device)); loss=model.compute_loss(o,b['label_cls'].to(device),b['label_reg'].to(device))['total']
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m=evaluate(model,va,device); m['epoch']=epoch; history.append(m)
        if m['accuracy']>best: best=m['accuracy']; torch.save({'model_state_dict':model.state_dict(),'config':cfg,'stride':args.stride,'metrics':m},out/'sais_best_model.pth')
        print(f"epoch {epoch}: accuracy={m['accuracy']:.4f} mae={m['grs_mae']:.4f} r={m['pearson_r']:.4f}")
    (out/'train_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--stride',type=int,default=1); p.add_argument('--seed',type=int,default=3407); main(p.parse_args())
