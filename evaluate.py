"""
Evaluation script.
Corresponds to paper Results: classification accuracy, regression metrics, 
confusion matrix, calibration analysis, error structure analysis.
"""
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json

from models import ArthroSkillMC
from data import ArthroscopyDataset, collate_fn
from utils import SkillMetrics


def evaluate(model, dataloader, device, config):
    model.eval()

    all_preds_cls = []
    all_labels_cls = []
    all_preds_reg = []
    all_labels_reg = []
    all_probs = []
    all_attn_weights = []
    all_occlusion_masks = []
    all_clip_ids = []

    with torch.no_grad():
        for batch in dataloader:
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
            all_attn_weights.extend(outputs["attn_weights"].cpu().numpy())
            all_occlusion_masks.extend(outputs["occlusion_mask"].cpu().numpy())
            all_clip_ids.extend(batch["clip_ids"])

    y_true_cls = np.array(all_labels_cls)
    y_pred_cls = np.array(all_preds_cls)
    y_true_reg = np.array(all_labels_reg)
    y_pred_reg = np.array(all_preds_reg)
    y_prob = np.array(all_probs)

    metrics = SkillMetrics.compute_all(y_true_cls, y_pred_cls, y_true_reg, y_pred_reg, y_prob)

    print("=" * 60)
    print("ArthroSkill-MC Evaluation Results")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Pearson r: {metrics['pearson_r']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    print(f"\nError Structure:")
    print(f"  Adjacent-grade: {metrics['error_structure']['adjacent']} ({metrics['error_structure']['adjacent_ratio']*100:.1f}%)")
    print(f"  Cross-grade: {metrics['error_structure']['cross_grade']}")
    print(f"  Total errors: {metrics['error_structure']['total_errors']}")
    if 'calibration_error' in metrics:
        print(f"\nCalibration Error (ECE): {metrics['calibration_error']:.4f}")
    print("=" * 60)

    results = {
        "metrics": metrics,
        "per_sample": []
    }

    for i in range(len(y_true_cls)):
        results["per_sample"].append({
            "clip_id": all_clip_ids[i],
            "true_cls": int(y_true_cls[i]),
            "pred_cls": int(y_pred_cls[i]),
            "true_reg": float(y_true_reg[i]),
            "pred_reg": float(y_pred_reg[i]),
            "probabilities": y_prob[i].tolist(),
            "is_error": bool(y_true_cls[i] != y_pred_cls[i]),
            "error_span": int(abs(y_true_cls[i] - y_pred_cls[i]))
        })

    return results


def main(config_path, data_dir, checkpoint_path, output_dir):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ArthroscopyDataset(data_dir, split="val", config_path=config_path)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], 
                           shuffle=False, collate_fn=collate_fn)

    model = ArthroSkillMC(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model: {checkpoint_path}")

    results = evaluate(model, dataloader, device, config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ArthroSkill-MC")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="./processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./outputs")
    args = parser.parse_args()
    main(args.config, args.data, args.checkpoint, args.output)
