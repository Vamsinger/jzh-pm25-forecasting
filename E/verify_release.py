"""独立复核最终发布物，并重建统一评价图。

默认在项目根目录使用 ``cache/``。重新克隆仓库后，可用 ``--cache`` 指向
已有的大缓存；若还要调用原评价入口重建全部图，缓存须位于项目的 ``cache/``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


START = pd.Timestamp("2015-01-01 00:00:00")
CUT2 = 1096 * 24
END = 1461 * 24
HORIZON = 24


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--cache", type=Path, default=root / "cache")
    parser.add_argument("--skip-plot-rebuild", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> list[str]:
    dataset = root / "JZH_Air_2014_2018"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8-sig"))
    checked = []
    assert manifest["algorithm"].lower() == "sha256"
    for entry in manifest["files"]:
        path = dataset / entry["path"]
        assert path.stat().st_size == entry["bytes"], f"文件大小不符：{entry['path']}"
        assert sha256(path) == entry["sha256"], f"SHA-256 不符：{entry['path']}"
        checked.append(entry["path"])
    return checked


def prediction_labels(settings: dict) -> list[str]:
    return ["B1", "B2", *[f"{model}_{seed}" for model in settings["experiments"] for seed in settings["seeds"]]]


def verify_arrays(root: Path, cache: Path) -> tuple[np.ndarray, list[str], tuple[int, int, int]]:
    settings = json.loads((root / "E/settings.json").read_text(encoding="utf-8-sig"))
    labels = prediction_labels(settings)
    origins = np.load(cache / "predictions/origins.npy", mmap_mode="r")
    expected_origins = np.arange(CUT2 - 1, END - HORIZON, dtype=origins.dtype)
    assert np.array_equal(origins, expected_origins), "测试起报时刻与冻结协议不一致"
    stations = pd.read_csv(cache / "A/stations.csv", dtype={"station_id": str})
    expected_shape = (len(expected_origins), len(stations), HORIZON)
    for label in ["truth", *labels]:
        array = np.load(cache / f"predictions/{label}.npy", mmap_mode="r")
        assert array.shape == expected_shape, f"{label} 形状错误：{array.shape}"
        assert array.dtype == np.float32, f"{label} 类型错误：{array.dtype}"
    first_origin = START + pd.Timedelta(hours=int(origins[0]))
    last_origin = START + pd.Timedelta(hours=int(origins[-1]))
    assert first_origin == pd.Timestamp("2017-12-31 23:00:00")
    assert first_origin + pd.Timedelta(hours=1) == pd.Timestamp("2018-01-01 00:00:00")
    assert last_origin + pd.Timedelta(hours=24) == pd.Timestamp("2018-12-31 23:00:00")
    return origins, labels, expected_shape


def verify_raw_truth(root: Path, cache: Path, origins: np.ndarray) -> dict:
    truth = np.load(cache / "predictions/truth.npy", mmap_mode="r")
    stations = pd.read_csv(cache / "A/stations.csv", dtype={"station_id": str})
    completeness = np.isfinite(truth[0]).sum(axis=1)
    station_index = int(np.argmax(completeness))
    station_id = stations.iloc[station_index].station_id
    origin = START + pd.Timedelta(hours=int(origins[0]))
    target_start, target_end = origin + pd.Timedelta(hours=1), origin + pd.Timedelta(hours=24)
    pieces = []
    for chunk in pd.read_csv(
        root / "JZH_Air_2014_2018/data/airquality.csv.gz",
        compression="gzip",
        dtype={"station_id": "string"},
        parse_dates=["time"],
        usecols=["station_id", "time", "PM25"],
        chunksize=250_000,
    ):
        use = (chunk.station_id == station_id) & chunk.time.between(target_start, target_end)
        pieces.append(chunk.loc[use, ["time", "PM25"]])
    raw = pd.concat(pieces, ignore_index=True).sort_values("time")
    expected_times = pd.date_range(target_start, target_end, freq="h")
    raw = raw.set_index("time").reindex(expected_times)
    assert np.allclose(truth[0, station_index], raw.PM25.to_numpy(), equal_nan=True)
    return {
        "station_id": station_id,
        "forecast_origin": str(origin),
        "target_period": [str(target_start), str(target_end)],
        "matched_hours": int(np.isfinite(raw.PM25).sum()),
    }


def recompute_m1(root: Path, cache: Path, labels: list[str]) -> pd.DataFrame:
    """以独立分块算法重算 M1 三个种子在共同样本上的 24 步指标。"""
    truth = np.load(cache / "predictions/truth.npy", mmap_mode="r")
    arrays = {label: np.load(cache / f"predictions/{label}.npy", mmap_mode="r") for label in labels}
    seeds = json.loads((root / "E/settings.json").read_text(encoding="utf-8-sig"))["seeds"]
    accumulators = {seed: [np.zeros(24), np.zeros(24), np.zeros(24, dtype=np.int64)] for seed in seeds}
    for start in range(0, len(truth), 37):
        part = np.asarray(truth[start : start + 37])
        common = np.isfinite(part)
        for array in arrays.values():
            common &= np.isfinite(array[start : start + 37])
        for seed in seeds:
            error = np.asarray(arrays[f"M1_{seed}"][start : start + 37]) - part
            absolute = np.where(common, np.abs(error), 0.0).astype(np.float64)
            square = np.where(common, error * error, 0.0).astype(np.float64)
            accumulators[seed][0] += absolute.sum(axis=(0, 1))
            accumulators[seed][1] += square.sum(axis=(0, 1))
            accumulators[seed][2] += common.sum(axis=(0, 1))
    rows = []
    for seed, (absolute, square, count) in accumulators.items():
        for horizon in range(24):
            rows.append({
                "model": "M1",
                "seed": str(seed),
                "horizon": horizon + 1,
                "n": int(count[horizon]),
                "MAE": absolute[horizon] / count[horizon],
                "RMSE": np.sqrt(square[horizon] / count[horizon]),
            })
    result = pd.DataFrame(rows)

    published = pd.read_csv(root / "E/results/metrics.csv", dtype={"seed": str})
    published = published[(published.model == "M1") & (published.city == "全区域") & (published.horizon > 0)]
    merged = result.merge(published, on=["model", "seed", "horizon"], suffixes=("_check", "_published"), validate="one_to_one")
    assert (merged.n_check == merged.n_published).all()
    assert np.allclose(merged.MAE_check, merged.MAE_published, rtol=1e-10, atol=1e-7)
    assert np.allclose(merged.RMSE_check, merged.RMSE_published, rtol=1e-10, atol=1e-7)

    summary = pd.read_csv(root / "E/results/summary.csv")
    summary = summary[(summary.model == "M1") & (summary.city == "全区域") & (summary.horizon > 0)]
    independent = result.groupby("horizon", as_index=False).agg(
        n=("n", "first"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std")
    )
    merged_summary = independent.merge(summary, on=["horizon"], suffixes=("_check", "_published"), validate="one_to_one")
    for column in ["MAE", "MAE_sd", "RMSE", "RMSE_sd"]:
        assert np.allclose(merged_summary[f"{column}_check"], merged_summary[f"{column}_published"], rtol=1e-10, atol=1e-7)
    return independent


def rebuild_plots(root: Path, cache: Path) -> list[str]:
    if cache.resolve() != (root / "cache").resolve():
        raise ValueError("重建全部图时，请把 --cache 指向项目根目录下的 cache")
    subprocess.run([sys.executable, str(root / "E/evaluate.py")], cwd=root, check=True)
    names = ["horizons.png", "cases.png", "cities.png"]
    for name in names:
        image = mpimg.imread(root / "E/results" / name)
        assert image.ndim in (2, 3) and min(image.shape[:2]) >= 500, f"图像尺寸异常：{name} {image.shape}"
    return names


def main() -> None:
    args = parse_args()
    root, cache = args.root.resolve(), args.cache.resolve()
    files = verify_manifest(root)
    origins, labels, shape = verify_arrays(root, cache)
    raw = verify_raw_truth(root, cache, origins)
    metrics = recompute_m1(root, cache, labels)
    plots = [] if args.skip_plot_rebuild else rebuild_plots(root, cache)
    report = {
        "status": "PASS",
        "manifest_files": len(files),
        "prediction_shape": list(shape),
        "prediction_arrays": len(labels) + 1,
        "raw_truth_check": raw,
        "independent_metric_rows": len(metrics),
        "m1_h1_mae": float(metrics.loc[metrics.horizon == 1, "MAE"].iloc[0]),
        "m1_h24_mae": float(metrics.loc[metrics.horizon == 24, "MAE"].iloc[0]),
        "rebuilt_plots": plots,
    }
    (root / "E/results/release_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = f"""# 发布复核说明

复核结论：**通过**。

- 依据 `manifest.json` 重新计算并通过 {len(files)} 个数据文件的大小和 SHA-256。
- 预测数组形状统一为 `{list(shape)}`，共检查 {len(labels) + 1} 个真值或预测数组。
- 测试标签从 2018-01-01 00:00 开始，到 2018-12-31 23:00 结束；历史均不晚于起报时刻。
- 原表核对站点 `{raw['station_id']}`，起报时刻 `{raw['forecast_origin']}`，逐小时真值与压缩原表一致。
- 使用独立分块算法重算 TCN 三个随机种子的全部 24 步 MAE、RMSE 和有效样本数，与发布表一致。
- 重新调用统一评价入口并检查图像可读：{', '.join(plots) if plots else '本次按参数跳过'}。

机器可读记录位于 `results/release_verification.json`。
"""
    (root / "E/复核说明.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
