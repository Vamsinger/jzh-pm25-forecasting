"""无需测试框架即可运行的接口验收。"""

import numpy as np
import torch

from C.models import TCN, baselines
from D.models import STGCN, make_graph


def main() -> None:
    history = np.full((2, 48, 5), np.nan, dtype=np.float32)
    history[:, -25:, :] = np.arange(25, dtype=np.float32)[None, :, None]
    persistence, daily = baselines(history)
    assert persistence.shape == daily.shape == (2, 5, 24)
    assert np.all(persistence == 24)
    assert np.all(daily[:, :, -1] == 24)

    import pandas as pd

    stations = pd.DataFrame(
        {"longitude": [120.0, 120.1, 120.2, 120.3, 120.4], "latitude": [30] * 5}
    )
    graph = make_graph(stations, k=2)
    assert graph.shape == (5, 5)
    np.testing.assert_allclose(graph.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(make_graph(stations, self_only=True), np.eye(5))
    duplicate_stations = stations.copy()
    duplicate_stations.loc[1, ["longitude", "latitude"]] = duplicate_stations.loc[
        0, ["longitude", "latitude"]
    ]
    duplicate_graph = make_graph(duplicate_stations, k=2)
    assert np.all((duplicate_graph > 0).sum(axis=1) >= 3)  # 2 个邻站和自环

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 48, 5, 8, device=device)
    future = torch.randn(2, 5, 24, 3, device=device)
    for model in (TCN(8, 3), STGCN(8, graph, 3)):
        model = model.to(device)
        output = model(x, future)
        assert output.shape == (2, 5, 24)
        output.mean().backward()

    # TCN 不应让一个站点的改动影响另一个站点。
    tcn = TCN(8).to(device).eval()
    clean = torch.zeros(1, 48, 5, 8, device=device)
    changed = clean.clone()
    changed[:, :, 0, :] = 1
    with torch.no_grad():
        clean_output = tcn(clean)
        changed_output = tcn(changed)
    torch.testing.assert_close(clean_output[:, 1:], changed_output[:, 1:])

    # 只有自环时，STGCN 也不应发生跨站传播。
    self_graph = make_graph(stations, self_only=True)
    stgcn = STGCN(8, self_graph).to(device).eval()
    with torch.no_grad():
        clean_output = stgcn(clean)
        changed_output = stgcn(changed)
    torch.testing.assert_close(clean_output[:, 1:], changed_output[:, 1:])
    print(f"接口验收通过，运行设备：{device}")


if __name__ == "__main__":
    main()
