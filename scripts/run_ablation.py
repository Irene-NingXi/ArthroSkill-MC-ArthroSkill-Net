"""Run the five paper ablations on a fixed train/validation split.

Variants: appearance_only, motion_only, naive_concat, unmasked, full.  The
wrapper preserves the paper's shared training and loss protocol while changing
only the requested information path.  ``naive_concat`` uses a learned
concatenation projection; the other variants reuse the production branches.
"""
import argparse, json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader
from data import ArthroscopyDataset, collate_fn
from models import ArthroSkillMC

class AblationModel(nn.Module):
    def __init__(self, config, variant):
        super().__init__(); self.variant=variant; self.base=ArthroSkillMC(config); dim=config['model']['motion']['output_dim']; app=config['model']['appearance']['output_dim']
        self.concat=nn.Sequential(nn.Linear(app+dim,dim),nn.LayerNorm(dim),nn.ReLU()) if variant=='naive_concat' else None
    def forward(self, appearance, motion, detection_conf=None):
        b=self.base; af=b.appearance_branch(appearance); mf,mask=b.motion_branch(motion,detection_conf)
        seq=motion.permute(0,2,1)
        for layer in b.motion_branch.encoder.cnn: seq=layer(seq)
        seq=seq.permute(0,2,1)
        if self.variant=='appearance_only': fused=b.fusion.appearance_residual(af); attn=None
        elif self.variant=='motion_only': fused=mf; attn=None
        elif self.variant=='naive_concat': fused=self.concat(torch.cat([af,mf],dim=-1)); attn=None
        else:
            if self.variant=='unmasked': mask=torch.ones_like(mask)
            fused,attn=b.fusion(af,seq,mask)
        cls, dims, total=b.head(fused)
        return {'cls_logits':cls,'dim_scores':dims,'total_score':total,'attn_weights':attn,'occlusion_mask':mask}
    def compute_loss(self,*args): return self.base.compute_loss(*args)

def main(a):
    c=yaml.safe_load(Path(a.config).read_text()); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=DataLoader(ArthroscopyDataset(a.data,'train',a.config),batch_size=c['training']['batch_size'],shuffle=True,collate_fn=collate_fn,num_workers=0); va=DataLoader(ArthroscopyDataset(a.data,'val',a.config),batch_size=c['training']['batch_size'],shuffle=False,collate_fn=collate_fn,num_workers=0)
    m=AblationModel(c,a.variant).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']); best=-1; rows=[]; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for ep in range(1,a.epochs+1):
        m.train()
        for batch in tr:
            o=m(batch['appearance'].to(dev),batch['motion'].to(dev),batch['detection_conf'].to(dev)); loss=m.compute_loss(o,batch['label_cls'].to(dev),batch['label_reg'].to(dev))['total']; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval(); yt=[]; yp=[]; ys=[]; ps=[]
        with torch.no_grad():
            for batch in va:
                o=m(batch['appearance'].to(dev),batch['motion'].to(dev),batch['detection_conf'].to(dev)); yt.extend(batch['label_cls'].tolist()); yp.extend(o['cls_logits'].argmax(1).cpu().tolist()); ys.extend(batch['label_reg'][:,6].tolist()); ps.extend(o['total_score'].squeeze(1).cpu().tolist())
        acc=float((np.asarray(yt)==np.asarray(yp)).mean()); mae=float(np.abs(np.asarray(ys)-np.asarray(ps)).mean()); row={'epoch':ep,'accuracy':acc,'grs_mae':mae}; rows.append(row); print(f'{a.variant} epoch {ep}: accuracy={acc:.4f} mae={mae:.4f}')
        if acc>best: best=acc; torch.save({'model_state_dict':m.state_dict(),'variant':a.variant,'config':c,'metrics':row},out/f'{a.variant}_best.pth')
    (out/f'{a.variant}_history.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--variant',choices=['appearance_only','motion_only','naive_concat','unmasked','full'],required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--seed',type=int,default=3407); main(p.parse_args())
