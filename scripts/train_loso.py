import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset, Subset
from models import ArthroSkillMC
from data import ArthroscopyDataset, collate_fn

def train_epoch(model, loader, optimizer, device):
	model.train()
	for batch in loader:
		outputs = model(batch["appearance"].to(device), batch["motion"].to(device))
		loss = model.compute_loss(outputs, batch["label_cls"].to(device), batch["label_reg"].to(device))["total"]
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

def evaluate(model, loader, device):
	model.eval(); truth_cls=[]; pred_cls=[]; truth_score=[]; pred_score=[]
	with torch.no_grad():
		for batch in loader:
			outputs = model(batch["appearance"].to(device), batch["motion"].to(device))
			truth_cls.extend(batch["label_cls"].numpy().tolist()); pred_cls.extend(outputs["cls_logits"].argmax(1).cpu().numpy().tolist())
			truth_score.extend(batch["label_reg"][:,6].numpy().tolist()); pred_score.extend(outputs["total_score"].squeeze(-1).cpu().numpy().tolist())
	y=np.asarray(truth_score); p=np.asarray(pred_score); r=float(np.corrcoef(y,p)[0,1]) if len(y)>1 and y.std()>0 and p.std()>0 else 0.0
	metrics={"n":len(y),"accuracy":float(np.mean(np.asarray(truth_cls)==np.asarray(pred_cls))),"pearson_r":r,"grs_mae":float(np.mean(np.abs(y-p)))}
	return metrics, truth_cls, pred_cls, y, p

def main():
	ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--data",required=True); ap.add_argument("--output",required=True); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
	random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
	with open(args.config,encoding="utf-8") as handle: config=yaml.safe_load(handle)
	train_set=ArthroscopyDataset(args.data,"train",args.config); val_set=ArthroscopyDataset(args.data,"val",args.config); combined=ConcatDataset([train_set,val_set]); groups={c:[] for c in ("A","B","C")}
	for offset, dataset in ((0,train_set),(len(train_set),val_set)):
		for local_index, sample in enumerate(dataset.samples):
			data=torch.load(sample["clip_path"],weights_only=False); centre=data.get("centre")
			if centre is None: raise KeyError(f"Missing centre in {sample['clip_path']}; add centre ('A', 'B', or 'C') as documented in data/README.md")
			if centre not in groups: raise ValueError(f"Unknown centre {centre!r}; expected A, B, or C")
			groups[centre].append(offset+local_index)
	if any(not indices for indices in groups.values()): raise ValueError("LOSO requires clips for centres A, B, and C")
	device=torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"]=="auto" else "cpu"); rows=[]; pooled_truth=[]; pooled_pred=[]; pooled_y=[]; pooled_p=[]; tr=config["training"]
	for held_out in ("A","B","C"):
		train_indices=[i for c,indices in groups.items() if c!=held_out for i in indices]; test_indices=groups[held_out]; bs=tr["batch_size"]
		train_loader=DataLoader(Subset(combined,train_indices),batch_size=bs,shuffle=True,collate_fn=collate_fn,num_workers=0); test_loader=DataLoader(Subset(combined,test_indices),batch_size=bs,shuffle=False,collate_fn=collate_fn,num_workers=0)
		model=ArthroSkillMC(config).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=tr["learning_rate"],weight_decay=tr["weight_decay"])
		for _ in range(tr["num_epochs"]): train_epoch(model,train_loader,optimizer,device)
		metrics,truth,pred,y,p=evaluate(model,test_loader,device); rows.append({"fold":held_out,"test_centre":held_out,**metrics}); pooled_truth.extend(truth); pooled_pred.extend(pred); pooled_y.extend(y.tolist()); pooled_p.extend(p.tolist())
	pooled={"fold":"pooled","test_centre":"all","n":len(pooled_truth),"accuracy":float(np.mean(np.asarray(pooled_truth)==np.asarray(pooled_pred))),"pearson_r":float(np.corrcoef(np.asarray(pooled_y),np.asarray(pooled_p))[0,1]) if len(pooled_y)>1 and np.std(pooled_y)>0 and np.std(pooled_p)>0 else 0.0,"grs_mae":float(np.mean(np.abs(np.asarray(pooled_y)-np.asarray(pooled_p))))}; rows.append(pooled)
	output=Path(args.output)/"loso"; output.mkdir(parents=True,exist_ok=True); (output/"loso_results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8"); print("fold test_centre n accuracy pearson_r grs_mae"); [print(r) for r in rows]

if __name__ == "__main__":
	main()
