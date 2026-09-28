"""Train/evaluate the paper's 3D-ResNet-18 or Video-Swin-only baselines."""
import argparse, json
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from data import ArthroscopyDataset, collate_fn
from models.baselines import ResNet3DBaseline, VideoSwinOnly

def main(a):
 c=yaml.safe_load(Path(a.config).read_text()); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); cls=ResNet3DBaseline if a.model=='resnet3d' else VideoSwinOnly; m=cls(c,pretrained=a.pretrained).to(dev) if a.model=='resnet3d' else cls(c).to(dev)
 tr=DataLoader(ArthroscopyDataset(a.data,'train',a.config),batch_size=c['training']['batch_size'],shuffle=True,collate_fn=collate_fn,num_workers=0); va=DataLoader(ArthroscopyDataset(a.data,'val',a.config),batch_size=c['training']['batch_size'],shuffle=False,collate_fn=collate_fn,num_workers=0); opt=torch.optim.AdamW(m.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 for ep in range(1,a.epochs+1):
  m.train()
  for b in tr:
   o=m(b['appearance'].to(dev)); loss=torch.nn.functional.cross_entropy(o['cls_logits'],b['label_cls'].to(dev))+torch.nn.functional.smooth_l1_loss(o['total_score'],b['label_reg'][:,6:7].to(dev)); opt.zero_grad(); loss.backward(); opt.step()
  m.eval(); yt=[]; yp=[]
  with torch.no_grad():
   for b in va: yt+=b['label_cls'].tolist(); yp+=m(b['appearance'].to(dev))['cls_logits'].argmax(1).cpu().tolist()
  acc=float((np.asarray(yt)==np.asarray(yp)).mean()); print(f'epoch {ep}: val_accuracy={acc:.4f}'); torch.save({'model_state_dict':m.state_dict(),'config':c,'accuracy':acc},out/f'{a.model}_best.pth')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--model',choices=['resnet3d','swin'],required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--pretrained',action='store_true'); main(p.parse_args())
