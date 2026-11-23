import os
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class contact_dataset(Dataset):
    """Sliding-window proprioceptive dataset for 16-class quadruped foot contact estimation.

    Loads time-series proprioceptive measurements (IMU linear acceleration, angular velocity,
    joint positions, joint velocities, and joint torques; 48 channels) and integer contact
    states in [0..15] (4-bit encoding across FL, FR, RL, RR feet).

    If `data_path` or `label_path` is missing on disk, generates realistic synthetic
    quadruped trot/walk proprioceptive sequences so training and evaluation run out-of-the-box.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        label_path: Optional[str] = None,
        window_size: int = 150,
        device: str = "cpu",
        input_channels: int = 48,
        stride: int = 1,
        synthetic_steps: int = 4000,
        seed: int = 42,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.stride = max(1, int(stride))
        self.input_channels = int(input_channels)
        self.device = device

        if data_path and label_path and os.path.exists(data_path) and os.path.exists(label_path):
            raw_x = np.load(data_path).astype(np.float32)
            raw_y = np.load(label_path).astype(np.int64).reshape(-1)
            if raw_x.shape[1] > self.input_channels:
                raw_x = raw_x[:, : self.input_channels]
            self.is_synthetic = False
        else:
            raw_x, raw_y = self._generate_synthetic_gait(
                num_steps=synthetic_steps,
                num_channels=self.input_channels,
                seed=seed,
            )
            self.is_synthetic = True

        if len(raw_x) != len(raw_y):
            min_len = min(len(raw_x), len(raw_y))
            raw_x, raw_y = raw_x[:min_len], raw_y[:min_len]

        if len(raw_x) < self.window_size:
            raise ValueError(f"Sequence length ({len(raw_x)}) is shorter than window_size ({self.window_size}).")

        self.x = torch.from_numpy(raw_x)
        self.y = torch.from_numpy(raw_y)
        self.indices = list(range(self.window_size - 1, len(self.x), self.stride))

    @staticmethod
    def _generate_synthetic_gait(
        num_steps: int = 4000, num_channels: int = 48, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        t = np.linspace(0, num_steps / 1000.0 * 2.0 * np.pi * 2.5, num_steps, dtype=np.float32)

        # Simulate diagonal trot phases for [FL, FR, RL, RR]
        phases = np.stack([
            np.sin(t),              # FL
            np.sin(t + np.pi),      # FR
            np.sin(t + np.pi),      # RL
            np.sin(t),              # RR
        ], axis=1)
        contacts = (phases > -0.15).astype(np.int64)
        labels = (
            contacts[:, 0] * 8
            + contacts[:, 1] * 4
            + contacts[:, 2] * 2
            + contacts[:, 3] * 1
        )

        # Build 48 proprioceptive channels correlated with contact phases + sensor noise
        features = np.zeros((num_steps, num_channels), dtype=np.float32)
        for ch in range(num_channels):
            leg_idx = ch % 4
            freq_mult = 1.0 + 0.25 * (ch // 12)
            features[:, ch] = (
                0.8 * phases[:, leg_idx] * freq_mult
                + 0.35 * contacts[:, leg_idx]
                + rng.normal(0.0, 0.12, size=num_steps).astype(np.float32)
            )
        return features, labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end_idx = self.indices[idx]
        start_idx = end_idx - self.window_size + 1
        x_win = self.x[start_idx : end_idx + 1]   # (W, C)
        y_end = self.y[end_idx]                   # scalar class in [0..15]
        y_seq = self.y[start_idx : end_idx + 1]   # (W,) per-timestep labels
        return x_win, y_end, y_seq
