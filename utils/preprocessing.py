"""
Data preprocessing utilities.
Corresponds to paper Methods: "Each video was decoded at 25 fps, rescaled to 640x480, 
and divided into non-overlapping 16-frame clips. Frames showing severe fluid turbidity 
or out-of-body views were removed using brightness and blur thresholds, 
and per-video intensity normalisation was applied."
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


class VideoPreprocessor:
    """
    Video preprocessing: decode -> clip -> filter bad frames -> normalise.
    """
    def __init__(self, clip_len=16, img_size=(480, 640), fps=25,
                 brightness_thresh=30, blur_thresh=100):
        self.clip_len = clip_len
        self.img_size = img_size  # (H, W)
        self.fps = fps
        self.brightness_thresh = brightness_thresh
        self.blur_thresh = blur_thresh

    def process_video(self, video_path, output_dir):
        """
        Process a single video.
        Args:
            video_path: path to raw video
            output_dir: output directory
        Returns:
            num_clips: number of valid clips generated
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return 0

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        valid_indices = []

        for i in tqdm(range(total_frames), desc=f"Reading {video_path.name}"):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            if brightness < self.brightness_thresh:
                continue

            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < self.blur_thresh:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            valid_indices.append(i)

        cap.release()

        if len(frames) < self.clip_len:
            print(f"Video {video_path.name}: insufficient valid frames ({len(frames)} < {self.clip_len})")
            return 0

        # Per-video intensity normalisation
        frames = np.array(frames, dtype=np.float32)
        mean = frames.mean(axis=(0, 1, 2), keepdims=True)
        std = frames.std(axis=(0, 1, 2), keepdims=True) + 1e-6
        frames = (frames - mean) / std

        # Split into non-overlapping 16-frame clips
        num_clips = 0
        for start in range(0, len(frames) - self.clip_len + 1, self.clip_len):
            clip = frames[start:start + self.clip_len]
            clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float()

            clip_name = f"{video_path.stem}_clip{num_clips:04d}.pt"
            torch.save({
                "frames": clip_tensor,
                "start_frame": valid_indices[start],
                "end_frame": valid_indices[min(start + self.clip_len - 1, len(valid_indices)-1)]
            }, output_dir / clip_name)
            num_clips += 1

        print(f"Video {video_path.name}: {len(frames)} valid frames -> {num_clips} clips")
        return num_clips

    def process_directory(self, input_dir, output_dir):
        """Batch process all videos in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
        video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in video_exts]

        print(f"Found {len(video_files)} video files")
        total_clips = 0
        for vf in video_files:
            out_subdir = output_dir / vf.stem
            n = self.process_video(vf, out_subdir)
            total_clips += n

        print(f"Preprocessing complete, {total_clips} clips generated")
        return total_clips
