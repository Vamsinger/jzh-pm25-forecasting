"""生成 PM2.5 站点覆盖率和逐年有效小时数图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "A"
OUTPUT = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    coverage = pd.read_csv(CACHE / "station_coverage.csv", dtype={"station_id": str, "city_id": str})
    yearly = [coverage[f"pm25_hours_{year}"].sum() for year in range(2015, 2019)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(coverage["pm25_coverage"] * 100, bins=12, color="#2878b5", edgecolor="white")
    axes[0].set(xlabel="PM2.5 coverage (%)", ylabel="Stations", title="Station coverage, 2015–2018")
    axes[1].bar([str(year) for year in range(2015, 2019)], yearly, color="#55a868")
    axes[1].set(xlabel="Year", ylabel="Observed station-hours", title="Valid PM2.5 observations")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    fig.tight_layout()
    fig.savefig(OUTPUT / "pm25_data_quality.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
