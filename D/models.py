"""地理近邻图和轻量时空图卷积网络。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

FORECAST_HORIZON = 24
EARTH_RADIUS_KM = 6371.0088


def _haversine_matrix(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    lon = np.radians(longitude)
    lat = np.radians(latitude)
    delta_lon = lon[:, None] - lon[None, :]
    delta_lat = lat[:, None] - lat[None, :]
    hav = np.sin(delta_lat / 2.0) ** 2
    hav += np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(delta_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def make_graph(
    stations: pd.DataFrame, k: int = 4, self_only: bool = False
) -> np.ndarray:
    """按经纬度建立对称 k 近邻图，并做行归一化。

    行列顺序严格沿用 ``stations`` 当前顺序。地理边使用高斯距离权重，
    高斯尺度为全部入选边距离的中位数。每个节点均包含自环。
    """

    required = {"longitude", "latitude"}
    if not required.issubset(stations.columns):
        raise ValueError("stations 必须包含 longitude 和 latitude")
    count = len(stations)
    if count == 0:
        raise ValueError("stations 不能为空")
    if self_only:
        return np.eye(count, dtype=np.float32)
    if not 1 <= k < count:
        raise ValueError("k 必须在 1 到站点数减 1 之间")

    coordinates = stations[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("构图前必须移除缺少坐标的站点")
    distances = _haversine_matrix(coordinates[:, 0], coordinates[:, 1])
    # 将自身显式排除，避免重复坐标使自身不在排序首位。
    neighbor_distances = distances.copy()
    np.fill_diagonal(neighbor_distances, np.inf)
    nearest = np.argsort(neighbor_distances, axis=1)[:, :k]
    selected = np.zeros((count, count), dtype=bool)
    selected[np.arange(count)[:, None], nearest] = True
    selected |= selected.T
    edge_distances = distances[selected]
    scale = float(np.median(edge_distances[edge_distances > 0]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    adjacency = np.zeros_like(distances, dtype=np.float64)
    adjacency[selected] = np.exp(-np.square(distances[selected] / scale))
    np.fill_diagonal(adjacency, 1.0)
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    return adjacency.astype(np.float32)


class _CausalGatedTemporal(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.filter_conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=self.padding, dilation=dilation
        )
        self.gate_conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=self.padding, dilation=dilation
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filtered = torch.tanh(self.filter_conv(x)[..., :-self.padding])
        gated = torch.sigmoid(self.gate_conv(x)[..., :-self.padding])
        return x + self.dropout(filtered * gated)


class STGCN(nn.Module):
    """先逐站提取时间特征，再按邻接矩阵传播的轻量 STGCN。"""

    def __init__(
        self,
        in_channels: int,
        adjacency: np.ndarray | torch.Tensor,
        future_channels: int = 0,
        hidden_channels: int = 12,
    ) -> None:
        super().__init__()
        graph = torch.as_tensor(adjacency, dtype=torch.float32)
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("adjacency 必须是方阵")
        if in_channels < 1 or future_channels < 0:
            raise ValueError("通道数必须有效")
        self.future_channels = future_channels
        self.register_buffer("adjacency", graph)
        self.input_projection = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.temporal = nn.Sequential(
            _CausalGatedTemporal(hidden_channels, dilation=1),
            _CausalGatedTemporal(hidden_channels, dilation=2),
            _CausalGatedTemporal(hidden_channels, dilation=4),
            _CausalGatedTemporal(hidden_channels, dilation=8),
            _CausalGatedTemporal(hidden_channels, dilation=16),
        )
        self.self_projection = nn.Linear(hidden_channels, hidden_channels)
        self.graph_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.horizon_embedding = nn.Parameter(
            torch.empty(FORECAST_HORIZON, hidden_channels)
        )
        nn.init.normal_(self.horizon_embedding, std=0.02)
        self.future_projection = (
            nn.Linear(future_channels, hidden_channels) if future_channels else None
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self, x: torch.Tensor, future: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x 必须是 [batch, time, stations, channels]")
        batch, history, stations, channels = x.shape
        if stations != self.adjacency.shape[0]:
            raise ValueError("x 的站点数与邻接矩阵不一致")
        values = torch.nan_to_num(x).permute(0, 2, 3, 1)
        values = values.reshape(batch * stations, channels, history)
        temporal = self.temporal(self.input_projection(values))[..., -1]
        temporal = temporal.reshape(batch, stations, -1)
        propagated = torch.einsum("nm,bmh->bnh", self.adjacency, temporal)
        context = torch.relu(
            self.self_projection(temporal) + self.graph_projection(propagated)
        )
        decoded = context[:, :, None, :] + self.horizon_embedding[None, None, :, :]

        if self.future_projection is not None:
            if future is None:
                raise ValueError("模型设置了 future_channels，forward 必须传入 future")
            expected = (batch, stations, FORECAST_HORIZON, self.future_channels)
            if tuple(future.shape) != expected:
                raise ValueError(f"future 形状应为 {expected}")
            decoded = decoded + self.future_projection(torch.nan_to_num(future))
        elif future is not None:
            raise ValueError("future_channels=0 时不应传入 future")
        return self.output(decoded).squeeze(-1)
