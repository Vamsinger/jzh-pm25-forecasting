"""把原始空气质量 CSV 整理成模型共享的逐小时张量。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ["PM25", "PM10", "NO2", "CO", "O3", "SO2"]
START = pd.Timestamp("2015-01-01 00:00:00")
END = pd.Timestamp("2018-12-31 23:00:00")
TRAIN_END = pd.Timestamp("2016-12-31 23:00:00")
N_HOURS = len(pd.date_range(START, END, freq="h"))


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project / "JZH_Air_2014_2018")
    parser.add_argument("--output", type=Path, default=project / "cache" / "A")
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def select_stations(aq_path: Path, metadata_path: Path, chunksize: int) -> pd.DataFrame:
    """选择训练期至少有一个 PM2.5 真值且有经纬度的站点。"""
    station_ids: set[str] = set()
    for chunk in pd.read_csv(
        aq_path,
        compression="gzip",
        usecols=["station_id", "time", "PM25"],
        dtype={"station_id": "string"},
        parse_dates=["time"],
        chunksize=chunksize,
    ):
        valid = chunk["time"].between(START, TRAIN_END) & chunk["PM25"].notna()
        station_ids.update(chunk.loc[valid, "station_id"].dropna().astype(str))

    metadata = pd.read_csv(
        metadata_path,
        dtype={"station_id": "string", "district_id": "string", "city_id": "string"},
    )
    keep = metadata["station_id"].isin(station_ids) & metadata[["longitude", "latitude"]].notna().all(axis=1)
    columns = ["station_id", "district_id", "city_id", "longitude", "latitude"]
    return metadata.loc[keep, columns].sort_values("station_id").reset_index(drop=True)


def build_tensor(aq_path: Path, stations: pd.DataFrame, output: Path, chunksize: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    station_to_col = {station_id: index for index, station_id in enumerate(stations["station_id"])}
    shape = (N_HOURS, len(stations), len(CHANNELS))
    sums_path = output / "air_sums.tmp.npy"
    counts_path = output / "air_counts.tmp.npy"
    sums = np.lib.format.open_memmap(sums_path, mode="w+", dtype="float32", shape=shape)
    counts = np.lib.format.open_memmap(counts_path, mode="w+", dtype="uint8", shape=shape)
    key_counts = np.zeros(shape[:2], dtype=np.uint8)
    rows_read = rows_in_tensor = invalid_time_rows = 0

    for chunk in pd.read_csv(
        aq_path,
        compression="gzip",
        dtype={"city_id": "string", "station_id": "string"},
        parse_dates=["time"],
        chunksize=chunksize,
    ):
        rows_read += len(chunk)
        hour = ((chunk["time"] - START) / pd.Timedelta(hours=1)).to_numpy()
        exact_hour = np.isfinite(hour) & (hour == np.floor(hour))
        in_range = exact_hour & (hour >= 0) & (hour < N_HOURS)
        station_col = chunk["station_id"].map(station_to_col)
        use = in_range & station_col.notna().to_numpy()
        invalid_time_rows += int((~exact_hour).sum())
        if not use.any():
            continue
        time_index = hour[use].astype(np.int64)
        station_index = station_col.to_numpy()[use].astype(np.int64)
        rows_in_tensor += len(time_index)
        np.add.at(key_counts, (time_index, station_index), 1)
        for channel_index, channel in enumerate(CHANNELS):
            values = pd.to_numeric(chunk.loc[use, channel], errors="coerce").to_numpy(dtype=np.float32)
            valid = np.isfinite(values)
            np.add.at(sums[:, :, channel_index], (time_index[valid], station_index[valid]), values[valid])
            np.add.at(counts[:, :, channel_index], (time_index[valid], station_index[valid]), 1)

    tensor_path = output / "air.npy"
    tensor = np.lib.format.open_memmap(tensor_path, mode="w+", dtype="float32", shape=shape)
    tensor[:] = np.nan
    for channel_index in range(len(CHANNELS)):
        channel_counts = counts[:, :, channel_index]
        observed = channel_counts > 0
        tensor[:, :, channel_index][observed] = (
            sums[:, :, channel_index][observed] / channel_counts[observed]
        )
    tensor.flush()
    del tensor, sums, counts, channel_counts, observed
    gc.collect()
    sums_path.unlink()
    counts_path.unlink()
    return {
        "source_rows": rows_read,
        "selected_rows": rows_in_tensor,
        "invalid_non_hour_rows": invalid_time_rows,
        "duplicate_station_time_extra_rows": int(np.maximum(key_counts.astype(np.int16) - 1, 0).sum()),
        "duplicate_policy": "同一站点同一小时存在多行时，对每个非空污染物取均值",
    }


def longest_missing_run(observed: np.ndarray) -> int:
    missing = ~observed
    if not missing.any():
        return 0
    padded = np.concatenate(([False], missing, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return int((ends - starts).max())


def write_audits(stations: pd.DataFrame, output: Path, build_audit: dict) -> None:
    tensor = np.load(output / "air.npy", mmap_mode="r")
    times = pd.date_range(START, END, freq="h")
    train = times <= TRAIN_END
    years = times.year.to_numpy()
    rows = []
    for index, station in stations.iterrows():
        pm25 = tensor[:, index, 0]
        row = {
            "station_id": station["station_id"],
            "city_id": station["city_id"],
            "observed_pm25_hours": int(np.isfinite(pm25).sum()),
            "pm25_coverage": float(np.isfinite(pm25).mean()),
            "longest_pm25_missing_run_hours": longest_missing_run(np.isfinite(pm25)),
        }
        for year in range(2015, 2019):
            row[f"pm25_hours_{year}"] = int(np.isfinite(pm25[years == year]).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "station_coverage.csv", index=False, encoding="utf-8-sig")

    train_values = tensor[train]
    # float64 累加避免几百万个 float32 观测求和时积累明显舍入误差。
    means = np.nanmean(train_values, axis=(0, 1), dtype=np.float64)
    stds = np.nanstd(train_values, axis=(0, 1), dtype=np.float64)
    counts = np.isfinite(train_values).sum(axis=(0, 1)).astype(int)
    statistics = {
        channel: {"mean": means[i], "std": stds[i], "observed_count": int(counts[i])}
        for i, channel in enumerate(CHANNELS)
    }
    with (output / "train_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, ensure_ascii=False, indent=2)

    metadata = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "channels": CHANNELS,
        "start": str(START),
        "end": str(END),
        "train_period": [str(START), str(TRAIN_END)],
        "validation_period": ["2017-01-01 00:00:00", "2017-12-31 23:00:00"],
        "test_period": ["2018-01-01 00:00:00", "2018-12-31 23:00:00"],
        "station_selection": "训练期至少有一条 PM25 真值，且元数据含经纬度",
        **build_audit,
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    aq_path = args.data_root / "data" / "airquality.csv.gz"
    stations = select_stations(aq_path, args.data_root / "metadata" / "stations.csv", args.chunksize)
    args.output.mkdir(parents=True, exist_ok=True)
    stations.to_csv(args.output / "stations.csv", index=False, encoding="utf-8-sig")
    build_audit = build_tensor(aq_path, stations, args.output, args.chunksize)
    write_audits(stations, args.output, build_audit)
    print(f"完成：{N_HOURS} 小时 × {len(stations)} 站 × {len(CHANNELS)} 个污染物")


if __name__ == "__main__":
    main()
