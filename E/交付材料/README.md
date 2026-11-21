# 最终交付材料

本文件夹提供两份可以直接下载和提交的文件：

- `实验报告.pdf`：10 页中文实验报告。
- `答辩幻灯片.pptx`：10 页、16:9、可继续编辑的答辩幻灯片。

报告和幻灯片都读取 `E/results/` 中的真实测试结果。生成脚本找不到主结果表或评价图时会直接停止，不会用占位数字代替。

## 一次重新生成

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File E/交付材料/generate_all.ps1
```

`generate_report.py` 使用 ReportLab 生成 PDF。`generate_presentation.mjs` 使用 Codex 桌面端随附的 Artifact Tool 生成 PPTX，并检查页数、字体、版式、原生表格和文件结构。`generate_all.ps1` 中记录了本机运行时路径。如果 Codex 桌面端更新了版本，只需改这个脚本中的 `RuntimeRoot` 和 `PresentationSkill`。

## 结论怎么读

A2 的整体 MAE 数值最低，为 15.616。它只比 A1 低 0.008，而且 RMSE 略高，这一差异小于三个随机种子的波动。因此材料把结论写成“天气预报在当前训练预算下没有显示稳定增益”。

训练期 PM2.5 第 90 百分位对应原始数值 100。测试真值达到这一阈值时，A2 的 MAE 为 50.755，高于持久性基线的 40.699。这个失败情景已经放入报告和答辩稿。

## 交付前还要做什么

五位成员需要共同核对结论和本人贡献，随后线下完成计时答辩演练。PPTX 已通过程序化导入与渲染检查，但仍建议在实际答辩电脑的 PowerPoint 中播放一遍，确认学校电脑上的字体和投影比例。
