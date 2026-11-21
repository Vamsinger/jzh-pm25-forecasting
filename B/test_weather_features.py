"""B 组对齐逻辑的最小回归测试。"""

import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_weather_features as weather


def test_forecast_uses_latest_available_release() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2015-01-01 08:00", "2015-01-01 20:00", "2015-01-01 08:00"]),
            "time_period_start": pd.to_datetime(["2015-01-02 08:00", "2015-01-02 08:00", "2015-01-01 20:00"]),
            "Up_Temperature": [10, 20, 5],
            "Bottom_Temperature": [0, 1, 2],
            "WindStrength": [3, 4, 5],
        }
    )
    result = weather.build_region_forecast(frame)
    assert result[19, 12, 0] == 10
    assert result[19, 12, 3] == 11
    assert result[20, 11, 0] == 20
    assert result[20, 11, 3] == 0
    assert result[9, 10, 0] == 5


def test_non_hourly_observation_is_not_moved_backwards() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2015-01-01 01:30", "2015-01-01 02:00"]),
            "Temperature": [1, 3],
            "Humidity": [2, 4],
            "WindSpeed": [3, 5],
        }
    )
    result = weather.region_hour_matrix(frame, weather.MET_FEATURES)
    assert np.isnan(result[1]).all()
    np.testing.assert_allclose(result[2], [2, 3, 4])


if __name__ == "__main__":
    test_forecast_uses_latest_available_release()
    test_non_hourly_observation_is_not_moved_backwards()
    print("2 tests passed")

