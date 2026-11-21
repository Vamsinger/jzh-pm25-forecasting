"""核验 A 组张量、样本切分和原始记录的一致性。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from air_dataset import AirWindowDataset, split_origins


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "A"
DATA = ROOT / "JZH_Air_2014_2018"
OUTPUT = Path(__file__).resolve().parent / "outputs"
START = pd.Timestamp("2015-01-01 00:00:00")
CHANNELS = ["PM25", "PM10", "NO2", "CO", "O3", "SO2"]


def source_sample(station_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    matches = []
    for chunk in pd.read_csv(
        DATA / "data" / "airquality.csv.gz",
        compression="gzip",
        dtype={"station_id": "string"},
        parse_dates=["time"],
        chunksize=250_000,
    ):
        use = (chunk["station_id"] == station_id) & chunk["time"].between(start, end)
        matches.append(chunk.loc[use, ["time", *CHANNELS]])
    return pd.concat(matches, ignore_index=True).sort_values("time")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    tensor = np.load(CACHE / "air.npy", mmap_mode="r")
    stations = pd.read_csv(CACHE / "stations.csv", dtype=str)
    assert tensor.shape == (35064, 212, 6)
    assert stations["station_id"].is_monotonic_increasing
    assert stations[["longitude", "latitude"]].notna().all().all()

    splits = split_origins()
    expected_bounds = {
        "train": (47, 17519),
        "validation": (17543, 26279),
        "test": (26303, 35039),
    }
    for name, origins in splits.items():
        assert (int(origins[0]), int(origins[-1])) == expected_bounds[name]
        AirWindowDataset(CACHE, origins[[0, -1]])

    origin = pd.Timestamp("2017-01-03 00:00:00")
    history = pd.read_csv(OUTPUT / "example_history_48h.csv", dtype={"station_id": str}, parse_dates=["time"])
    target = pd.read_csv(OUTPUT / "example_target_24h.csv", dtype={"station_id": str}, parse_dates=["time"])
    station_id = history.loc[0, "station_id"]
    raw = source_sample(station_id, history["time"].min(), target["time"].max())
    generated = history[["time", *CHANNELS]]
    merged = generated.merge(raw, on="time", how="left", suffixes=("_generated", "_raw"), validate="one_to_one")
    for channel in CHANNELS:
        generated_name = f"{channel}_generated"
        raw_name = f"{channel}_raw"
        if generated_name in merged and raw_name in merged:
            assert np.allclose(merged[generated_name], merged[raw_name], equal_nan=True)
    target_check = target[["time", "PM25"]].merge(
        raw[["time", "PM25"]], on="time", how="left", suffixes=("_generated", "_raw"), validate="one_to_one"
    )
    assert np.allclose(target_check["PM25_generated"], target_check["PM25_raw"], equal_nan=True)

    report = {
        "status": "PASS",
        "tensor_shape": list(tensor.shape),
        "station_count": len(stations),
        "split_origin_bounds": {name: [int(values[0]), int(values[-1])] for name, values in splits.items()},
        "source_checked_station": station_id,
        "source_checked_period": [str(history["time"].min()), str(target["time"].max())],
        "source_rows_checked": len(raw),
        "origin": str(origin),
    }
    with (OUTPUT / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
