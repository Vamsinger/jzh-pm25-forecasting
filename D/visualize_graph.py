"""生成站点地理近邻图预览和构图审计表。"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from D.models import make_graph


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    station_path = root / "cache" / "A" / "stations.csv"
    output_dir = Path(__file__).resolve().parent
    stations = pd.read_csv(
        station_path,
        dtype={"station_id": "string", "city_id": "string", "district_id": "string"},
    )
    usable = stations.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)
    k = json.loads((root / "E/results/frozen_M2.json").read_text())[0]["k"]
    graph = make_graph(usable, k=k)

    figure, axis = plt.subplots(figsize=(7.2, 6.2), dpi=160)
    lon = usable["longitude"].to_numpy()
    lat = usable["latitude"].to_numpy()
    rows, columns = np.where((graph > 0) & ~np.eye(len(graph), dtype=bool))
    for row, column in zip(rows, columns):
        if row < column:
            axis.plot(
                [lon[row], lon[column]],
                [lat[row], lat[column]],
                color="#8ba7b5",
                linewidth=0.25,
                alpha=0.3,
                zorder=1,
            )
    axis.scatter(lon, lat, s=7, c="#c23b22", alpha=0.85, zorder=2)
    axis.set(title=f"Forecasting network: {len(usable)} stations, {k} nearest neighbors", xlabel="Longitude", ylabel="Latitude")
    axis.grid(alpha=0.16)
    figure.tight_layout()
    figure.savefig(output_dir / "station_graph.png")
    plt.close(figure)

    audit = pd.DataFrame(
        {
            "metric": [
                "model_station_count",
                "usable_coordinate_count",
                "excluded_coordinate_count",
                "k",
                "directed_nonzero_off_diagonal",
                "row_sum_min",
                "row_sum_max",
            ],
            "value": [
                len(stations),
                len(usable),
                len(stations) - len(usable),
                k,
                int(((graph > 0) & ~np.eye(len(graph), dtype=bool)).sum()),
                float(graph.sum(axis=1).min()),
                float(graph.sum(axis=1).max()),
            ],
        }
    )
    audit.to_csv(output_dir / "graph_audit.csv", index=False, encoding="utf-8-sig")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()

