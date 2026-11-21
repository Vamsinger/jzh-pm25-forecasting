"""简单基线和逐站共享参数的时间卷积网络。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


FORECAST_HORIZON = 24


def baselines(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回持久性预测和日周期朴素预测。

    参数 ``history`` 的形状为 ``[批次, 历史时间, 站点]``，可以包含 NaN。
    持久性预测使用 48 小时窗口内最近的有效值。日周期预测将历史最后
    24 小时逐点平移到未来 24 小时，缺失值保持为 NaN。
    """

    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("history 必须是 [batch, time, stations] 三维数组")
    if values.shape[1] < FORECAST_HORIZON:
        raise ValueError("日周期基线至少需要 24 小时历史")

    valid = np.isfinite(values)
    time_index = np.arange(values.shape[1], dtype=np.int32)[None, :, None]
    latest_index = np.where(valid, time_index, -1).max(axis=1)
    safe_index = np.maximum(latest_index, 0)
    latest = np.take_along_axis(values, safe_index[:, None, :], axis=1)[:, 0, :]
    latest[latest_index < 0] = np.nan

    persistence = np.repeat(latest[:, :, None], FORECAST_HORIZON, axis=2)
    daily = values[:, -FORECAST_HORIZON:, :].transpose(0, 2, 1).copy()
    return persistence, daily


class _CausalGatedBlock(nn.Module):
    """不读取未来位置的门控一维卷积块。"""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.padding = padding
        self.filter_conv = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding=padding
        )
        self.gate_conv = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding=padding
        )
        self.dropout = nn.Dropout(0.1)

    def _trim(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., :-self.padding]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self._trim(self.filter_conv(values)))
        gated = gated * torch.sigmoid(self._trim(self.gate_conv(values)))
        return values + self.dropout(gated)


class TCN(nn.Module):
    """所有站点共享参数、站点之间互不串扰的轻量 TCN。

    ``x`` 为 ``[batch, 48, stations, in_channels]``。若使用未来天气，
    ``future`` 为 ``[batch, stations, 24, future_channels]``。输出始终为
    ``[batch, stations, 24]``。
    """

    def __init__(
        self,
        in_channels: int,
        future_channels: int = 0,
        hidden_channels: int = 12,
    ) -> None:
        super().__init__()
        if in_channels < 1 or future_channels < 0:
            raise ValueError("通道数必须有效")
        self.future_channels = future_channels
        self.input_projection = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.temporal = nn.Sequential(
            _CausalGatedBlock(hidden_channels, dilation=1),
            _CausalGatedBlock(hidden_channels, dilation=2),
            _CausalGatedBlock(hidden_channels, dilation=4),
            _CausalGatedBlock(hidden_channels, dilation=8),
            _CausalGatedBlock(hidden_channels, dilation=16),
        )
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
        batch, _, stations, _ = x.shape
        values = torch.nan_to_num(x).permute(0, 2, 3, 1)
        values = values.reshape(batch * stations, x.shape[3], x.shape[1])
        context = self.temporal(self.input_projection(values))[..., -1]
        context = context.reshape(batch, stations, -1)
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
