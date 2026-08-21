"""
Dataset definition.
Corresponds to paper Methods: "Each video was decoded at 25 fps, rescaled to 640x480, 
and divided into non-overlapping 16-frame clips."
"""
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import yaml
from pathlib import Path


class ArthroscopyDataset(Dataset):
    """
    Arthroscopic surgery video dataset.
    Input: preprocessed 16-frame clips (.pt or .npy)
    Output: {
        'appearance': [C, T, H, W] video tensor,
        'motion': [T, D] kinematic feature sequence (optional, placeholder if not pre-computed),
        'label_cls': int skill level (0=novice, 1=intermediate, 2=expert),
        'label_reg': [7] 6 GRS dimensions + total score (normalised to 0-1)
    }
    """
    def __init__(self, data_dir, split="train", config_path="configs/config.yaml", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.clip_len = self.cfg["data"]["clip_len"]
        self.img_size = self.cfg["data"]["img_size"]
        self.num_classes = self.cfg["data"]["num_classes"]

        self.samples = self._load_samples()
        print(f"[{split}] Loaded {len(self.samples)} samples")

    def _load_samples(self):
        split_dir = self.data_dir / self.split
        clips_dir = split_dir / "clips"

        samples = []
        if not clips_dir.exists():
            return samples

        for clip_file in sorted(clips_dir.glob("*.pt")):
            samples.append({
                "clip_path": str(clip_file),
                "clip_id": clip_file.stem
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.load(sample["clip_path"], weights_only=False)

        appearance = data["frames"]
        if appearance.dim() == 4 and appearance.shape[-1] == 3:
            appearance = appearance.permute(3, 0, 1, 2)

        motion = data.get("motion", torch.zeros(self.clip_len, 6))
        label_cls = data["label_cls"]
        label_reg = data["label_reg"]

        if self.transform and self.split == "train":
            appearance = self._augment(appearance)

        return {
            "appearance": appearance.float(),
            "motion": motion.float(),
            "label_cls": torch.tensor(label_cls, dtype=torch.long),
            "label_reg": label_reg.float(),
            "clip_id": sample["clip_id"]
        }

    def _augment(self, x):
        """Basic data augmentation: random horizontal flip."""
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[-1])
        return x


def collate_fn(batch):
    """Custom collate for batching."""
    appearance = torch.stack([b["appearance"] for b in batch])
    motion = torch.stack([b["motion"] for b in batch])
    label_cls = torch.stack([b["label_cls"] for b in batch])
    label_reg = torch.stack([b["label_reg"] for b in batch])
    clip_ids = [b["clip_id"] for b in batch]

    return {
        "appearance": appearance,
        "motion": motion,
        "label_cls": label_cls,
        "label_reg": label_reg,
        "clip_ids": clip_ids
    }
