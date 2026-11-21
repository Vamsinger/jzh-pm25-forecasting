# 融合气象信息的江浙沪 PM2.5 短期时空预测

给定 212 个站过去 48 小时的污染物和当时可获得的天气，逐小时预测未来 24 小时 PM2.5。本项目使用 2015～2018 年江浙沪课程数据。训练使用 2015～2016 年，验证使用 2017 年，所有模型定型后在 2018 年测试。允许使用起报时已经发生的上年度历史，未来 24 个标签必须落在同一分区。

## 2018 年的测试结果

25 个城市、212 个站点共有 24,330,653 个共同可评价的预测目标。跨全部 24 步，完整模型 MAE 为 15.616，持久性基线为 16.946。单独加入已发布预报，MAE 从 15.625 降到 15.616，幅度小于三个随机种子的波动。第 1 小时持久性基线更准，第 24 小时完整模型更准。达到训练期 PM2.5 第 90 百分位时，持久性基线也优于完整模型。详见 [按场景统计](E/results/scenarios.csv)和[消融曲线](E/results/ablation.png)。

## 从哪里开始看

| 想了解什么 | 去哪里看 |
|---|---|
| 分工、正在做的事、实际起止时间 | [总体流程进展](总体流程进展.md)；每组目录下还有个人记录 |
| 原始数据与作业要求 | [完整数据](JZH_Air_2014_2018/)、[课程摘录](选题与课程要求.md)、[五人组任务清单](五人组任务清单.md) |
| 污染物如何清洗，缺失时怎么处理 | [A · 空气质量](A/README.md) |
| 气象记录与预报怎样匹配到站点和小时 | [B · 气象对齐](B/README.md) |
| 如何用本站历史预测 | [C · 基线和 TCN](C/README.md) |
| 邻站怎样参与预测 | [D · 站点图和 STGCN](D/README.md) |
| 模型怎么比、怎样复核结果 | [E · 统一评价](E/README.md)、[完整指标](E/results/summary.csv)、[真实预测样例](E/results/shanghai_predictions.csv) |
| 阅读报告或准备答辩 | [实验报告与答辩材料](E/交付材料/) |

## 怎样重跑

命令均在仓库根目录执行。实验环境使用 Python 3.10.20、PyTorch 2.7.1（本机为 CUDA 12.8 构建）、NumPy 2.2.6、pandas 2.3.3 和 Matplotlib 3.10.9。本次实际训练使用 RTX 5080。安装适配设备的 PyTorch 后，用 `python -m pip install -r requirements.txt` 补齐依赖。首次运行请为输入缓存和逐样本预测预留约 10 GB 以上磁盘空间。

```powershell
python A/prepare_airquality.py
python A/make_example.py
python A/plot_data_quality.py
python A/validate_airquality.py
python B/build_weather_features.py --data-root JZH_Air_2014_2018 --stations cache/A/stations.csv --output cache/B
python -m C.test_models
python -m E.experiment train
python -m D.visualize_graph
python -m E.experiment test
python -m E.evaluate
python -m E.sample
python -m E.scenarios
python -m E.verify_release
```

仓库提供正式运行的轻量权重、配置与日志。已有权重时可从重建数据直接进入 `test`，再用 `evaluate` 重建结果表和图。要从头训练，可在一个新工作副本中先把 `C/runs`、`D/runs` 改名备份。逐起报时刻的完整预测在本地 `cache/predictions`，使用 `E/results/shanghai_predictions.csv` 可直接查看一个展开的例子。

## 数据与署名

课程包来自 AirMeteorology.7z 的江浙沪子集。文件摘要、原始压缩包指纹和数据文件校验值都在数据目录。数据源未标明时区和部分变量单位，报告按原始时间与浓度数值解释。模型参考文献及与原实现的差别见 C、D 两份说明。A～E 是五个工作包，本仓库代码、实验、文稿及执行记录由 Codex 协助生成和实际运行；真人成员姓名、贡献核实和口头演练由小组完成。


