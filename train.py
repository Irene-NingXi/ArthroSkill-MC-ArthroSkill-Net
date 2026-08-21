"""
Training script.
Corresponds to paper Methods: "Training was implemented in MMAction2... AdamW at a learning rate of 1e-4, 
batch size 4, cosine-annealing schedule, and data augmentation"
Note: This implementation uses standard PyTorch (no hard dependency on MMAction2 specific APIs).
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

from models import ArthroSkillMC
from data import ArthroscopyDataset, collate_fn
from utils import SkillMetrics


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, num_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        appearance = batch["appearance"].to(device)
        motion = batch["motion"].to(device)
        labels_cls = batch["label_cls"].to(device)
        labels_reg = batch["label_reg"].to(device)

        optimizer.zero_grad()
        outputs = model(appearance, motion)
        losses = model.compute_loss(outputs, labels_cls, labels_reg)
        loss = losses["total"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += losses["cls"].item()
        total_reg_loss += losses["reg"].item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cls": f"{losses['cls'].item():.4f}",
            "reg": f"{losses['reg'].item():.4f}"
        })

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "reg_loss": total_reg_loss / n
    }


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    all_preds_cls = []
    all_labels_cls = []
    all_preds_reg = []
    all_labels_reg = []
    all_probs = []

    for batch in tqdm(dataloader, desc="[Validate]"):
        appearance = batch["appearance"].to(device)
        motion = batch["motion"].to(device)
        labels_cls = batch["label_cls"]
        labels_reg = batch["label_reg"]

        outputs = model(appearance, motion)

        probs = torch.softmax(outputs["cls_logits"], dim=-1)
        preds_cls = torch.argmax(probs, dim=-1).cpu().numpy()
        preds_total = outputs["total_score"].squeeze(-1).cpu().numpy()

        all_preds_cls.extend(preds_cls)
        all_labels_cls.extend(labels_cls.numpy())
        all_preds_reg.extend(preds_total)
        all_labels_reg.extend(labels_reg[:, 6].numpy())
        all_probs.extend(probs.cpu().numpy())

    metrics = SkillMetrics.compute_all(
        np.array(all_labels_cls),
        np.array(all_preds_cls),
        np.array(all_labels_reg),
        np.array(all_preds_reg),
        np.array(all_probs)
    )

    return metrics


def main(config_path, data_dir, output_dir):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] == "auto" else "cpu")
    print(f"Device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ArthroscopyDataset(data_dir, split="train", config_path=config_path)
    val_dataset = ArthroscopyDataset(data_dir, split="val", config_path=config_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn
    )

    model = ArthroSkillMC(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = build_optimizer(
        model,
        config["training"]["learning_rate"],
        config["training"]["weight_decay"]
    )
    scheduler = build_scheduler(
        optimizer,
        config["training"]["num_epochs"],
        config["training"]["warmup_epochs"]
    )

    best_acc = 0.0
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (cls={train_metrics['cls_loss']:.4f}, reg={train_metrics['reg_loss']:.4f})")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Pearson r: {val_metrics['pearson_r']:.4f}")
        print(f"  Val MAE: {val_metrics['mae']:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics,
                "config": config
            }, output_dir / "best_model.pth")
            print(f"  -> Saved best model (acc={best_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArthroSkill-MC")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="./processed")
    parser.add_argument("--output", default="./outputs")
    args = parser.parse_args()
    main(args.config, args.data, args.output)
