"""把气象观测和天气预报对齐到 A 组确定的站点与小时轴。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


START = pd.Timestamp("2015-01-01 00:00:00")
END = pd.Timestamp("2018-12-31 23:00:00")
N_HOURS = int((END - START) / pd.Timedelta(hours=1)) + 1
MET_FEATURES = ["Temperature", "Humidity", "WindSpeed"]
FORECAST_FEATURES = ["Up_Temperature", "Bottom_Temperature", "WindStrength", "forecast_age_hours"]


def hour_index(values: pd.Series) -> np.ndarray:
    return ((values - START) / pd.Timedelta(hours=1)).astype("int32").to_numpy()


def load_stations(path: Path) -> pd.DataFrame:
    stations = pd.read_csv(
        path,
        dtype={"station_id": "string", "district_id": "string", "city_id": "string"},
    )
    required = {"station_id", "district_id", "city_id"}
    missing = required.difference(stations.columns)
    if missing:
        raise ValueError(f"站点文件缺少字段: {sorted(missing)}")
    if stations["station_id"].duplicated().any():
        raise ValueError("站点文件 station_id 不唯一")
    return stations.reset_index(drop=True)


def read_meteorology(path: Path, regions: set[str]) -> pd.DataFrame:
    frames = []
    usecols = ["region_id", "region_level", "time", *MET_FEATURES]
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"region_id": "string", "region_level": "category"},
        parse_dates=["time"],
        chunksize=500_000,
    ):
        keep = chunk["region_id"].isin(regions) & chunk["time"].between(START, END)
        frames.append(chunk.loc[keep])
    return pd.concat(frames, ignore_index=True)


def region_hour_matrix(group: pd.DataFrame, features: list[str]) -> np.ndarray:
    # 非整点观测到下一个整点才视作可用，避免把 01:30 的信息放进 01:00。
    available_hour = group["time"].dt.ceil("h")
    grouped = group.assign(available_hour=available_hour).groupby("available_hour", sort=False, observed=True)[features].mean()
    out = np.full((N_HOURS, len(features)), np.nan, dtype="float32")
    valid = grouped.index.to_series().between(START, END).to_numpy()
    idx = ((grouped.index[valid] - START) / pd.Timedelta(hours=1)).astype("int32")
    out[idx] = grouped.iloc[np.flatnonzero(valid)].to_numpy(dtype="float32")
    return out


def build_meteorology(raw: pd.DataFrame, stations: pd.DataFrame, output: Path) -> dict:
    out = np.lib.format.open_memmap(
        output, mode="w+", dtype="float32", shape=(N_HOURS, len(stations), len(MET_FEATURES))
    )
    out[:] = np.nan
    matrices: dict[tuple[str, str], np.ndarray] = {}
    duplicate_rows = int(raw.duplicated(["region_level", "region_id", "time"], keep=False).sum())
    for (level, region), group in raw.groupby(["region_level", "region_id"], observed=True, sort=False):
        matrices[(str(level), str(region))] = region_hour_matrix(group, MET_FEATURES)

    fallback = 0
    district_used = 0
    for station_idx, row in stations.iterrows():
        city = matrices.get(("city", str(row.city_id)))
        district = matrices.get(("district", str(row.district_id)))
        if city is not None:
            out[:, station_idx, :] = city
        if district is not None:
            mask = np.isfinite(district)
            current = out[:, station_idx, :]
            current[mask] = district[mask]
            out[:, station_idx, :] = current
            district_used += 1
        else:
            fallback += 1
    out.flush()
    availability = np.isfinite(out).mean(axis=(0, 1)).tolist()
    del out
    return {
        "features": MET_FEATURES,
        "duplicate_rows_before_hourly_mean": duplicate_rows,
        "stations_with_district_data": district_used,
        "stations_city_only": fallback,
        "availability": dict(zip(MET_FEATURES, availability)),
        "non_hourly_source_rows_ceiled_forward": int((raw["time"].dt.minute.ne(0) | raw["time"].dt.second.ne(0)).sum()),
        "alignment_rule": "整点记录保留原时刻；非整点记录向上对齐到下一个整点，保证该小时开始前记录已经产生",
    }


def read_forecast(path: Path, regions: set[str]) -> pd.DataFrame:
    frames = []
    cols = ["region_id", "region_level", "time", "time_period_start", "Up_Temperature", "Bottom_Temperature", "WindStrength"]
    for chunk in pd.read_csv(
        path,
        usecols=cols,
        dtype={"region_id": "string", "region_level": "category"},
        parse_dates=["time", "time_period_start"],
        chunksize=500_000,
    ):
        keep = chunk["region_id"].isin(regions) & chunk["time"].between(START, END)
        frames.append(chunk.loc[keep])
    return pd.concat(frames, ignore_index=True)


def build_region_forecast(group: pd.DataFrame) -> np.ndarray:
    """建立 [起报小时, 预测时距1..24, 特征]，同一目标时刻使用最新已发布记录。"""
    data = group.copy()
    data = data[data["time_period_start"].dt.minute.eq(0) & data["time"].dt.minute.eq(0)]
    data = data[data["time_period_start"].between(START + pd.Timedelta(hours=1), END + pd.Timedelta(hours=24))]
    data = data.sort_values(["time_period_start", "time"]).drop_duplicates(
        ["time_period_start", "time"], keep="last"
    )
    result = np.full((N_HOURS, 24, len(FORECAST_FEATURES)), np.nan, dtype="float32")
    target_idx = hour_index(data["time_period_start"])
    issue_idx = hour_index(data["time"])
    next_issue = data.groupby("time_period_start", sort=False)["time"].shift(-1)
    next_issue_idx = hour_index(next_issue.fillna(data["time_period_start"]))
    values = data[["Up_Temperature", "Bottom_Temperature", "WindStrength"]].to_numpy(dtype="float32")
    # 每个目标时刻最多查看 24 个预测时距。对每个时距一次性筛选该起报时刻最新可用的一版。
    for horizon in range(1, 25):
        origins = target_idx - horizon
        valid = (
            (origins >= 0)
            & (origins < N_HOURS)
            & (issue_idx <= origins)
            & (origins < next_issue_idx)
        )
        if not valid.any():
            continue
        result[origins[valid], horizon - 1, :3] = values[valid]
        result[origins[valid], horizon - 1, 3] = origins[valid] - issue_idx[valid]
    return result


def build_forecast(raw: pd.DataFrame, stations: pd.DataFrame, output: Path) -> dict:
    shape = (N_HOURS, len(stations), 24, len(FORECAST_FEATURES))
    out = np.lib.format.open_memmap(output, mode="w+", dtype="float32", shape=shape)
    out[:] = np.nan
    grouped = {(str(a), str(b)): g for (a, b), g in raw.groupby(["region_level", "region_id"], observed=True, sort=False)}
    cache: dict[tuple[str, str], np.ndarray] = {}
    district_uses = stations["district_id"].astype(str).value_counts().to_dict()

    def matrix(key: tuple[str, str]) -> np.ndarray | None:
        if key not in grouped:
            return None
        if key not in cache:
            cache[key] = build_region_forecast(grouped[key])
        return cache[key]

    district_used = 0
    for station_idx, row in stations.iterrows():
        city = matrix(("city", str(row.city_id)))
        if city is not None:
            out[:, station_idx, :, :] = city
        district = matrix(("district", str(row.district_id)))
        if district is not None:
            current = out[:, station_idx, :, :]
            # 区县任一核心温度字段可用时，用区县整条记录覆盖城市记录。
            valid = np.isfinite(district[:, :, 0]) | np.isfinite(district[:, :, 1])
            current[valid] = district[valid]
            out[:, station_idx, :, :] = current
            district_used += 1
        district_key = ("district", str(row.district_id))
        district_uses[str(row.district_id)] -= 1
        if district_uses[str(row.district_id)] == 0:
            cache.pop(district_key, None)
    out.flush()
    matched = np.zeros(24, dtype="int64")
    age_values: list[tuple[np.ndarray, np.ndarray]] = []
    chunk_hours = 24 * 31
    for start in range(0, N_HOURS, chunk_hours):
        block = out[start : min(N_HOURS, start + chunk_hours)]
        matched += np.isfinite(block[:, :, :, 0]).sum(axis=(0, 1))
        age_block = block[:, :, :, 3]
        finite = age_block[np.isfinite(age_block)]
        if finite.size:
            age_values.append(np.unique(finite, return_counts=True))
    match_by_horizon = (matched / (N_HOURS * len(stations))).tolist()
    age_counts: dict[float, int] = {}
    for values, counts in age_values:
        for value, count in zip(values, counts):
            age_counts[float(value)] = age_counts.get(float(value), 0) + int(count)
    ordered_ages = sorted(age_counts.items())
    total_age = sum(count for _, count in ordered_ages)
    midpoint = (total_age - 1) // 2
    cumulative = 0
    median_age = None
    for value, count in ordered_ages:
        cumulative += count
        if cumulative > midpoint:
            median_age = value
            break
    age_summary = {
        "min": ordered_ages[0][0] if ordered_ages else None,
        "median": median_age,
        "max": ordered_ages[-1][0] if ordered_ages else None,
    }
    del out
    return {
        "features": FORECAST_FEATURES,
        "stations_with_district_forecast": district_used,
        "match_rate_by_horizon": {str(i + 1): rate for i, rate in enumerate(match_by_horizon)},
        "forecast_age_hours": age_summary,
        "matching_rule": "目标小时与 time_period_start 精确相等；对同一目标小时选择不晚于起报时刻的最新发布记录；区县优先、城市回退",
    }


def write_trace_example(stations: pd.DataFrame, met_path: Path, forecast_path: Path, output: Path) -> None:
    met = np.load(met_path, mmap_mode="r")
    forecast = np.load(forecast_path, mmap_mode="r")
    # 优先选择起报时刻同时有实测天气的样例，便于人工核对整条链路。
    valid_forecast = np.isfinite(forecast[:, :, :, 0])
    valid_met = np.isfinite(met[:, :, 0])
    candidates = np.argwhere(valid_forecast & valid_met[:, :, None])
    if not len(candidates):
        candidates = np.argwhere(valid_forecast)
    if not len(candidates):
        raise RuntimeError("没有可追溯的预报匹配样例")
    origin, station_idx, horizon_idx = candidates[len(candidates) // 2]
    station = stations.iloc[int(station_idx)]
    origin_time = START + pd.Timedelta(hours=int(origin))
    target_time = origin_time + pd.Timedelta(hours=int(horizon_idx) + 1)
    forecast_values = forecast[origin, station_idx, horizon_idx]
    forecast_age = forecast_values[3]
    issued_time = origin_time - pd.Timedelta(hours=float(forecast_age)) if np.isfinite(forecast_age) else None
    record = {
        "station_id": str(station.station_id),
        "district_id": str(station.district_id),
        "city_id": str(station.city_id),
        "forecast_origin": str(origin_time),
        "horizon": int(horizon_idx) + 1,
        "target_time": str(target_time),
        "forecast_issued_time": str(issued_time) if issued_time is not None else None,
        "source_lookup_order": [f"district:{station.district_id}", f"city:{station.city_id}"],
        "meteorology_at_origin": dict(zip(MET_FEATURES, [None if not np.isfinite(x) else float(x) for x in met[origin, station_idx]])),
        "forecast_features": dict(zip(FORECAST_FEATURES, [None if not np.isfinite(x) else float(x) for x in forecast_values])),
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "JZH_Air_2014_2018")
    parser.add_argument("--stations", type=Path, default=Path(__file__).resolve().parents[1] / "cache/A/stations.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "cache/B")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stations = load_stations(args.stations)
    regions = set(stations["district_id"].dropna().astype(str)) | set(stations["city_id"].dropna().astype(str))

    met_raw = read_meteorology(args.data_root / "data/meteorology.csv.gz", regions)
    met_audit = build_meteorology(met_raw, stations, args.output / "met.npy")
    del met_raw
    forecast_raw = read_forecast(args.data_root / "data/weather_forecast.csv.gz", regions)
    forecast_audit = build_forecast(forecast_raw, stations, args.output / "forecast.npy")
    source_audit = {
        "forecast_rows": len(forecast_raw),
        "duplicate_keys": int(forecast_raw.duplicated(["region_level", "region_id", "time", "time_period_start"], keep=False).sum()),
        "publication_hours": sorted(forecast_raw["time"].dt.hour.dropna().unique().astype(int).tolist()),
        "target_minute_counts": forecast_raw["time_period_start"].dt.minute.value_counts().sort_index().astype(int).to_dict(),
    }
    del forecast_raw
    audit = {
        "time_axis": {"start": str(START), "end": str(END), "hours": N_HOURS},
        "station_count": len(stations),
        "meteorology": met_audit,
        "forecast": forecast_audit,
        "source_forecast_audit": source_audit,
    }
    (args.output / "weather_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_trace_example(stations, args.output / "met.npy", args.output / "forecast.npy", args.output / "trace_example.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

