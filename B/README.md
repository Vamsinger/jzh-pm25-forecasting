# B 组：气象信息对齐

本文件夹负责把“实际观测到的天气”和“预测当时已经发布的天气预报”接到 A 组的站点小时轴上。最重要的原则很简单：模型不能偷看未来。

## 输入与输出

脚本读取原始气象表、天气预报表和 `cache/A/stations.csv`。输出在 `cache/B/`：

- `met.npy`：`[小时, 站点, 3]`，三个字段依次是温度、湿度、风速。
- `forecast.npy`：`[起报小时, 站点, 24, 4]`，四个字段依次是最高温、最低温、风力、预报发布后经过的小时数。
- `weather_audit.json`：覆盖率、24 个预测时距的匹配率和原始预报审计。
- `trace_example.json`：一个可以追溯的真实匹配样例。

数组中的缺失值保留为 `NaN`。区县数据可用时优先使用区县数据，对应小时没有区县记录时回退到城市记录。

## 如何运行

在仓库根目录执行：

```powershell
python B/build_weather_features.py
```

脚本只把 `time_period_start` 与目标小时完全相同的预报用于该小时，并从同一目标小时的多次预报中选择起报时刻以前最新发布的一次。这个规则容易复查，也避免把时段含义猜错。

## 后续模型怎样读取

```python
import numpy as np

met = np.load("cache/B/met.npy", mmap_mode="r")
forecast = np.load("cache/B/forecast.npy", mmap_mode="r")
```

两个数组都支持内存映射，不需要一次性装入内存。站点维度严格跟随 A 组 `stations.csv` 的行序。

