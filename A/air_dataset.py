"""A 组输出的轻量样本接口，供 C、D 组模型共同使用。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class AirWindowDataset:
    """根据起报小时提取历史 48 小时和未来 24 小时 PM2.5。"""

    def __init__(self, cache_dir: str | Path, origins: np.ndarray, history: int = 48, horizon: int = 24):
        self.cache_dir = Path(cache_dir)
        self.air = np.load(self.cache_dir / "air.npy", mmap_mode="r")
        self.origins = np.asarray(origins, dtype=np.int64)
        self.history = history
        self.horizon = horizon
        if self.origins.size and (self.origins.min() < history - 1 or self.origins.max() + horizon >= len(self.air)):
            raise ValueError("起报小时没有完整的历史窗口或未来标签窗口")

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        origin = int(self.origins[index])
        history = np.asarray(self.air[origin - self.history + 1 : origin + 1], dtype=np.float32)
        target = np.asarray(self.air[origin + 1 : origin + self.horizon + 1, :, 0], dtype=np.float32)
        return {
            "origin_index": origin,
            "history": history,
            "history_mask": np.isfinite(history),
            "target_pm25": target,
            "target_mask": np.isfinite(target),
        }


def load_train_normalization(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with (Path(cache_dir) / "train_statistics.json").open(encoding="utf-8") as handle:
        stats = json.load(handle)
    channels = ["PM25", "PM10", "NO2", "CO", "O3", "SO2"]
    mean = np.asarray([stats[name]["mean"] for name in channels], dtype=np.float32)
    std = np.asarray([stats[name]["std"] for name in channels], dtype=np.float32)
    return mean, std


def split_origins(history: int = 48, horizon: int = 24) -> dict[str, np.ndarray]:
    """返回未来标签完整属于分区的起报索引。

    验证和测试允许使用边界前的合法历史，因为真实部署时这些观测在起报时已经存在。
    """
    boundaries = {
        "train": (0, 24 * (365 + 366) - 1),
        "validation": (24 * (365 + 366), 24 * (365 + 366 + 365) - 1),
        "test": (24 * (365 + 366 + 365), 35064 - 1),
    }
    return {
        name: np.arange(history - 1 if start == 0 else start - 1, end - horizon + 1, dtype=np.int64)
        for name, (start, end) in boundaries.items()
    }
