"""导出一个可以人工核对的真实 48→24 小时样本。"""

from pathlib import Path

import numpy as np
import pandas as pd

from air_dataset import AirWindowDataset


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "A"
OUTPUT = Path(__file__).resolve().parent / "outputs"
ORIGIN = pd.Timestamp("2017-01-03 00:00:00")
START = pd.Timestamp("2015-01-01 00:00:00")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    stations = pd.read_csv(CACHE / "stations.csv", dtype=str)
    origin_index = int((ORIGIN - START) / pd.Timedelta(hours=1))
    item = AirWindowDataset(CACHE, np.asarray([origin_index]))[0]
    # 选择这个起报时刻历史和未来真值最完整的站，避免人为挑固定站导致空样例。
    completeness = item["history_mask"].sum(axis=(0, 2)) + item["target_mask"].sum(axis=0)
    station_index = int(np.argmax(completeness))
    station_id = stations.iloc[station_index]["station_id"]
    history_times = pd.date_range(ORIGIN - pd.Timedelta(hours=47), ORIGIN, freq="h")
    target_times = pd.date_range(ORIGIN + pd.Timedelta(hours=1), periods=24, freq="h")
    history = pd.DataFrame(item["history"][:, station_index], columns=["PM25", "PM10", "NO2", "CO", "O3", "SO2"])
    history.insert(0, "time", history_times)
    history.insert(0, "station_id", station_id)
    history.to_csv(OUTPUT / "example_history_48h.csv", index=False, encoding="utf-8-sig")
    target = pd.DataFrame({
        "station_id": station_id,
        "time": target_times,
        "horizon": np.arange(1, 25),
        "PM25": item["target_pm25"][:, station_index],
        "target_valid": item["target_mask"][:, station_index],
    })
    target.to_csv(OUTPUT / "example_target_24h.csv", index=False, encoding="utf-8-sig")
    print(f"样例站点 {station_id}，起报时刻 {ORIGIN}")


if __name__ == "__main__":
    main()
