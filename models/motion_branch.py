"""
Motion branch: YOLOv8 detection + DeepSORT tracking + 6 kinematic features + 1D-CNN + BiLSTM
Corresponds to paper Methods Motion branch section.
"""
import torch
import torch.nn as nn
import numpy as np


class KinematicFeatureExtractor:
    """
    Extract 6 kinematic features from instrument trajectories.
    Paper: "From the resulting trajectories we computed six kinematic feature sequences"
    """
    def __init__(self, fov_diagonal=800, pause_vel_ratio=0.05):
        self.fov_diagonal = fov_diagonal
        self.pause_vel_ratio = pause_vel_ratio
        self.epsilon = 1e-3

    def extract(self, trajectories, frame_shape):
        """Extract 6-D feature vector from single-frame tracking results."""
        if not trajectories:
            return np.zeros(6), False

        H, W = frame_shape
        fov_diag = np.sqrt(H**2 + W**2)
        features = []

        # 1. Normalised path length
        total_path = 0.0
        for inst_id, traj in trajectories.items():
            if len(traj) >= 2:
                centers = [self._box_center(box) for box in traj]
                for j in range(1, len(centers)):
                    total_path += np.linalg.norm(np.array(centers[j]) - np.array(centers[j-1]))
        path_length = total_path / (fov_diag + self.epsilon)
        features.append(path_length)

        # 2. Average velocity
        velocities = []
        for inst_id, traj in trajectories.items():
            if len(traj) >= 2:
                centers = [self._box_center(box) for box in traj]
                for j in range(1, len(centers)):
                    dist = np.linalg.norm(np.array(centers[j]) - np.array(centers[j-1]))
                    velocities.append(dist)
        avg_vel = np.mean(velocities) if velocities else 0.0
        features.append(avg_vel)

        # 3. Motion smoothness (jerk)
        jerk_smoothness = 0.0
        for inst_id, traj in trajectories.items():
            if len(traj) >= 4:
                centers = [self._box_center(box) for box in traj]
                vels = []
                for j in range(1, len(centers)):
                    vels.append(np.array(centers[j]) - np.array(centers[j-1]))
                accs = []
                for j in range(1, len(vels)):
                    accs.append(vels[j] - vels[j-1])
                jerks = []
                for j in range(1, len(accs)):
                    jerks.append(np.linalg.norm(accs[j] - accs[j-1]))
                if jerks:
                    jerk_smoothness += np.mean(jerks)
        smoothness = 1.0 / (jerk_smoothness + self.epsilon)
        features.append(smoothness)

        # 4. Pause ratio
        pause_ratio = 0.0
        if velocities:
            peak = max(velocities) if max(velocities) > 0 else 1.0
            threshold = self.pause_vel_ratio * peak
            is_pause = [v < threshold for v in velocities]
            pause_ratio = sum(is_pause) / len(is_pause) if is_pause else 0.0
        features.append(pause_ratio)

        # 5. Bimanual coordination
        c_bi = 0.0
        inst_ids = list(trajectories.keys())
        if len(inst_ids) >= 2:
            traj1 = trajectories[inst_ids[0]]
            traj2 = trajectories[inst_ids[1]]
            if len(traj1) >= 2 and len(traj2) >= 2:
                centers1 = [self._box_center(box) for box in traj1]
                centers2 = [self._box_center(box) for box in traj2]
                v1 = [np.linalg.norm(np.array(centers1[j]) - np.array(centers1[j-1])) 
                      for j in range(1, len(centers1))]
                v2 = [np.linalg.norm(np.array(centers2[j]) - np.array(centers2[j-1])) 
                      for j in range(1, len(centers2))]
                min_len = min(len(v1), len(v2))
                v1_arr = np.array(v1[-min_len:])
                v2_arr = np.array(v2[-min_len:])
                if len(v1_arr) > 1 and np.std(v1_arr) > 0 and np.std(v2_arr) > 0:
                    c_bi = np.corrcoef(v1_arr, v2_arr)[0, 1]
                    c_bi = max(0.0, c_bi)
        features.append(c_bi)

        # 6. Scope stability
        all_centers = []
        for traj in trajectories.values():
            if traj:
                all_centers.append(self._box_center(traj[-1]))
        if len(all_centers) >= 2:
            centers_arr = np.array(all_centers)
            var = np.var(centers_arr, axis=0).mean()
            scope_stability = 1.0 / (var + self.epsilon)
        else:
            scope_stability = 0.0
        features.append(scope_stability)

        return np.array(features, dtype=np.float32), True

    @staticmethod
    def _box_center(box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


class MotionEncoder(nn.Module):
    """
    Motion feature encoder: 3-layer 1D-CNN + BiLSTM.
    Paper: kernel sizes 7,5,3; channel widths 64,128,256; BiLSTM hidden=128.
    """
    def __init__(self, input_dim=6, cnn_channels=[64, 128, 256], 
                 cnn_kernels=[7, 5, 3], lstm_hidden=128, lstm_layers=1, 
                 output_dim=256, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = input_dim
        for out_ch, k in zip(cnn_channels, cnn_kernels):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.proj = nn.Sequential(
            nn.Linear(lstm_hidden * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_forward = h_n[0]
        h_backward = h_n[1]
        h = torch.cat([h_forward, h_backward], dim=-1)
        out = self.proj(h)
        return out


class MotionBranch(nn.Module):
    """
    Motion branch wrapper.
    Input: [B, T, 6] pre-computed kinematic feature sequences.
    Output: [B, 256] motion feature vector + [B, T] occlusion mask.
    """
    def __init__(self, input_dim=6, output_dim=256, conf_threshold=0.5, **encoder_kwargs):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.encoder = MotionEncoder(input_dim=input_dim, output_dim=output_dim, **encoder_kwargs)

    def forward(self, motion_features, detection_conf=None):
        if detection_conf is not None:
            occlusion_mask = (detection_conf >= self.conf_threshold).float()
        else:
            occlusion_mask = torch.ones(motion_features.shape[0], motion_features.shape[1],
                                        device=motion_features.device)
        motion_vec = self.encoder(motion_features)
        return motion_vec, occlusion_mask
