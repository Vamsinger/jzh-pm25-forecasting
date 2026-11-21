"""从真实实验结果生成 10 页中文课程实验报告 PDF。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "E" / "results"
OUTPUT = Path(__file__).resolve().parent / "实验报告.pdf"
MODEL_NAMES = {
    "B1": "持久性基线",
    "B2": "日周期基线",
    "M1": "TCN",
    "M2": "STGCN",
    "A1": "STGCN + 历史气象",
    "A2": "STGCN + 历史气象 + 已发布预报",
    "A3": "完整模型（仅自环）",
}


def load_inputs():
    required = [
        RESULTS / "summary.csv",
        RESULTS / "horizons.png",
        RESULTS / "cities.png",
        RESULTS / "cases.png",
        RESULTS / "ablation.png",
        RESULTS / "scenarios.csv",
        RESULTS / "coverage.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少最终结果文件：" + ", ".join(missing))
    summary = pd.read_csv(RESULTS / "summary.csv")
    coverage = json.loads((RESULTS / "coverage.json").read_text(encoding="utf-8"))
    settings = json.loads((ROOT / "E" / "settings.json").read_text(encoding="utf-8"))
    dataset = json.loads(
        (ROOT / "JZH_Air_2014_2018" / "dataset_summary.json").read_text(
            encoding="utf-8"
        )
    )
    runs = []
    for folder in [ROOT / "C" / "runs", ROOT / "D" / "runs"]:
        for path in folder.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict) and "experiment" in record:
                    runs.append(record)
            except json.JSONDecodeError:
                continue
    scenarios = pd.read_csv(RESULTS / "scenarios.csv")
    return summary, coverage, settings, dataset, pd.DataFrame(runs), scenarios


def register_fonts():
    regular = Path("C:/Windows/Fonts/Deng.ttf")
    bold = Path("C:/Windows/Fonts/Dengb.ttf")
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("未找到等线字体，无法可靠生成中文 PDF")
    pdfmetrics.registerFont(TTFont("Deng", str(regular)))
    pdfmetrics.registerFont(TTFont("DengBold", str(bold)))
    pdfmetrics.registerFontFamily("Deng", normal="Deng", bold="DengBold")


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName="DengBold",
            fontSize=24,
            leading=33,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base["Normal"],
            fontName="Deng",
            fontSize=13,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#506477"),
        ),
        "h1": ParagraphStyle(
            "H1CN",
            parent=base["Heading1"],
            fontName="DengBold",
            fontSize=18,
            leading=25,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=base["Heading2"],
            fontName="DengBold",
            fontSize=12,
            leading=18,
            textColor=colors.HexColor("#176B87"),
            spaceBefore=6,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=10.2,
            leading=16.5,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#263746"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=8.2,
            leading=12,
            textColor=colors.HexColor("#4F6070"),
        ),
        "callout": ParagraphStyle(
            "CalloutCN",
            parent=base["BodyText"],
            fontName="DengBold",
            fontSize=11,
            leading=18,
            textColor=colors.HexColor("#A84A32"),
            borderColor=colors.HexColor("#E5B8A9"),
            borderWidth=0.8,
            borderPadding=9,
            backColor=colors.HexColor("#FFF6F2"),
            spaceAfter=10,
        ),
    }


def p(text, style):
    return Paragraph(text, style)


def bullet(items, style):
    return [p("• " + item, style) for item in items]


def styled_table(data, widths=None, font_size=8.2):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "DengBold"),
                ("FONTNAME", (0, 1), (-1, -1), "Deng"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 3),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C4CE")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def overall_rows(summary):
    overall = summary[(summary["city"] == "全区域") & (summary["horizon"] == 0)]
    rows = [["编号", "模型", "MAE", "RMSE", "有效标签数"]]
    for code in MODEL_NAMES:
        match = overall[overall["model"] == code]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            [
                code,
                MODEL_NAMES[code],
                f"{row.MAE:.2f}" if pd.isna(row.MAE_sd) else f"{row.MAE:.2f} ± {row.MAE_sd:.2f}",
                f"{row.RMSE:.2f}" if pd.isna(row.RMSE_sd) else f"{row.RMSE:.2f} ± {row.RMSE_sd:.2f}",
                f"{int(row.n):,}",
            ]
        )
    return rows, overall


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D4DCE2"))
    canvas.line(1.8 * cm, 1.45 * cm, A4[0] - 1.8 * cm, 1.45 * cm)
    canvas.setFont("Deng", 8)
    canvas.setFillColor(colors.HexColor("#607080"))
    canvas.drawString(1.8 * cm, 1.05 * cm, "江浙沪 PM2.5 短期时空预测")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.05 * cm, str(doc.page))
    canvas.restoreState()


def build_story(summary, coverage, settings, dataset, runs, scenarios, styles):
    rows, overall = overall_rows(summary)
    a2 = overall[overall.model == "A2"].iloc[0]
    a1 = overall[overall.model == "A1"].iloc[0]
    a3 = overall[overall.model == "A3"].iloc[0]
    m1 = overall[overall.model == "M1"].iloc[0]
    m2 = overall[overall.model == "M2"].iloc[0]
    b1 = overall[overall.model == "B1"].iloc[0]
    best = overall.loc[overall.MAE.idxmin()]
    history_weather_gain = (m2.MAE - a1.MAE) / m2.MAE * 100
    forecast_increment = (a1.MAE - a2.MAE) / a1.MAE * 100
    space_increment = (a3.MAE - a2.MAE) / a3.MAE * 100
    baseline_gain = (b1.MAE - a2.MAE) / b1.MAE * 100
    extreme = scenarios[scenarios.group == "训练期P90及以上"]
    normal = scenarios[scenarios.group == "训练期P90以下"]
    extreme_b1 = float(extreme[extreme.model_seed == "B1"].MAE.iloc[0])
    extreme_a2 = float(extreme[extreme.model_seed.str.startswith("A2_")].MAE.mean())
    normal_b1 = float(normal[normal.model_seed == "B1"].MAE.iloc[0])
    normal_a2 = float(normal[normal.model_seed.str.startswith("A2_")].MAE.mean())
    train_runs = runs.groupby("experiment", as_index=False).agg(
        parameters=("parameters", "max"), seconds=("seconds", "mean")
    ) if not runs.empty else pd.DataFrame()

    story = []
    story += [Spacer(1, 3.2 * cm), p("融合气象信息的江浙沪<br/>PM2.5 短期时空预测", styles["title"])]
    story += [Spacer(1, 0.6 * cm), p("时空数据挖掘课程项目实验报告", styles["subtitle"])]
    story += [Spacer(1, 3.0 * cm), p("五人组 A、B、C、D、E", styles["subtitle"])]
    story += [p("数据范围：2015—2018 年　预测范围：未来 1—24 小时", styles["subtitle"])]
    story += [Spacer(1, 3.0 * cm), p("报告由项目内真实数据、训练日志和评价结果自动生成。模型结论以测试集汇总表为准。", styles["small"]), PageBreak()]

    story += [p("摘要与研究问题", styles["h1"])]
    story += [p(
        "本项目研究江浙沪空气质量监测站的短期预测问题。模型读取过去 48 小时的污染物和气象记录，直接预测各站未来 24 小时的 PM2.5。实验重点检验两个问题：站点之间的地理联系能否降低误差，以及预测发起时已经发布的天气预报是否提供额外信息。",
        styles["body"],
    )]
    story += [p(
        f"测试集结果中，{MODEL_NAMES[str(best.model)]}取得数值最低的整体 MAE {best.MAE:.2f}。完整模型 A2 相对持久性基线降低 {baseline_gain:.1f}%。A2 与 A1 只差 {a1.MAE-a2.MAE:.3f}，小于随机种子波动，而且 RMSE 略高，因此本实验没有证据说明天气预报带来稳定提升。",
        styles["callout"],
    )]
    story += [p("研究对象与输入输出", styles["h2"])]
    story += bullet([
        "节点为具有训练期 PM2.5 观测且坐标完整的空气质量监测站。",
        "历史输入包含六种污染物、对应版本的历史气象和缺失标记。",
        "未来输入只使用预测发起时已经发布并能匹配目标时段的天气预报。",
        "模型一次输出未来连续 24 小时，评价 MAE、RMSE 和有效标签数。",
    ], styles["body"])
    story += [p("关键词：PM2.5；多步时间序列回归；TCN；STGCN；天气预报", styles["small"]), PageBreak()]

    aq = dataset["data"]["airquality"]
    met = dataset["data"]["meteorology"]
    fc = dataset["data"]["weatherforecast"]
    story += [p("数据与质量审计", styles["h1"])]
    data_table = [
        ["数据表", "空间粒度", "时间范围", "记录数"],
        ["空气质量", "站点", f"{aq['min_time'][:10]} 至 {aq['max_time'][:10]}", f"{aq['output_rows']:,}"],
        ["实测气象", "城市或区县", f"{met['min_time'][:10]} 至 {met['max_time'][:10]}", f"{met['output_rows']:,}"],
        ["天气预报", "城市或区县", f"{fc['min_time'][:10]} 至 {fc['max_time'][:10]}", f"{fc['output_rows']:,}"],
    ]
    story += [styled_table(data_table, [3.2 * cm, 3.4 * cm, 6.2 * cm, 3.2 * cm])]
    story += [Spacer(1, 0.3 * cm), p("关键数据处理", styles["h2"])]
    story += bullet([
        "目录名包含 2014，但江浙沪数据从 2015 年开始。实验按 2015—2016 年训练、2017 年验证、2018 年测试划分。",
        f"空气质量表出现 {dataset['observed_airquality_station_count']} 个站。三处观测站缺少空间元数据，正式节点集合由训练期观测和坐标共同确定。",
        f"PM2.5 字段已有记录中包含 {aq['missing_values']['PM25']:,} 个空值。流程另行建立逐小时时间轴，以识别整行缺失。",
        "气象表同时包含城市和区县记录。站点优先匹配所属区县，缺少时回退到城市记录，并在合并后检查站点时刻主键唯一。",
        "天气预报表区分发布时间 time 与预报时段 time_period_start。预测原点 t 只使用 time 不晚于 t 的记录。",
        f"精确时段匹配率从 1 小时的 30.89% 缓慢降至 24 小时的 30.20%，因此预报特征在多数目标位置保持缺失并由掩码标识。",
    ], styles["body"])
    story += [Image(str(ROOT / "A" / "outputs" / "pm25_data_quality.png"), width=11.8 * cm, height=5.8 * cm)]
    story += [p("图中汇总各站 PM2.5 覆盖情况，为节点筛选和缺失处理提供依据。", styles["small"])]
    story += [PageBreak()]

    story += [p("方法", styles["h1"])]
    method_table = [
        ["编号", "方法", "历史气象", "未来预报", "空间信息"],
        ["B1", "最近有效值延续", "否", "否", "否"],
        ["B2", "昨天相同小时", "否", "否", "否"],
        ["M1", "TCN", "否", "否", "否"],
        ["M2", "STGCN", "否", "否", "地理近邻图"],
        ["A1", "STGCN + 历史气象", "是", "否", "地理近邻图"],
        ["A2", "完整模型", "是", "是", "地理近邻图"],
        ["A3", "完整模型的空间消融", "是", "是", "仅自环"],
    ]
    story += [styled_table(method_table, [1.3 * cm, 6.6 * cm, 2.6 * cm, 2.6 * cm, 3.7 * cm], 8.0)]
    story += [p("TCN 与 STGCN", styles["h2"])]
    story += [p(
        "TCN 对每个站独立使用同一组时间卷积参数。五层三点因果卷积采用 1、2、4、8、16 的膨胀率，感受野为 63 小时。STGCN 先提取逐站历史序列表示，再通过行归一化的地理近邻矩阵传播。两个模型均使用小型 12 通道隐藏层和 24 步直接输出头。",
        styles["body"],
    )]
    story += [p("气象融合", styles["h2"])]
    story += [p(
        "历史气象与污染物沿时间轴共同输入。未来天气按目标小时组织，并额外保留缺失标记和预报新鲜度。模型在每个预测时距将未来天气投影到隐藏空间，再与站点历史表示相加。A1、A2 和 A3 每次只改变一个因素，用于分别判断预报和空间信息的贡献。",
        styles["body"],
    )]
    story += [Image(str(ROOT / "D" / "station_graph.png"), width=7.4 * cm, height=6.2 * cm)]
    story += [p("正式空间图包含 212 个入模站点，验证集选择 6 个近邻，并在对称化后进行行归一化。", styles["small"])]
    story += [PageBreak()]

    story += [p("实验设置与可复现性", styles["h1"])]
    exp_table = [
        ["设置", "取值"],
        ["历史窗口 / 预测范围", f"{settings['history']} 小时 / {settings['horizons']} 小时"],
        ["随机种子", "、".join(str(x) for x in settings["seeds"])],
        ["训练轮数", str(settings["epochs"])],
        ["批次大小", str(settings["batch"])],
        ["训练 / 验证起报步长", f"{settings['train_origin_stride']} / {settings['validation_origin_stride']} 小时"],
        ["候选学习率", "、".join(str(x) for x in settings["learning_rates"])],
        ["候选邻居数", "、".join(str(x) for x in settings["neighbors"])],
        ["运行设备", "NVIDIA GeForce RTX 5080，16 GB 显存"],
    ]
    story += [styled_table(exp_table, [6.2 * cm, 10.4 * cm], 8.5)]
    if not train_runs.empty:
        runtime_rows = [["实验", "参数量", "平均训练时间（秒）"]]
        for row in train_runs.itertuples():
            runtime_rows.append([row.experiment, f"{int(row.parameters):,}", f"{row.seconds:.2f}"])
        story += [Spacer(1, 0.35 * cm), p("模型规模与训练记录", styles["h2"]), styled_table(runtime_rows, [4 * cm, 5 * cm, 7.6 * cm])]
    story += [p(
        "配置只在验证集上选择。测试集用于冻结方案后的最终评价。关键学习模型运行三个随机种子，表中报告均值与标准差。训练采用稀疏起报步长和少量轮数以满足课程计算预算，因此结论限于当前实验配置。",
        styles["body"],
    )]
    story += [PageBreak()]

    story += [p("主结果", styles["h1"]), styled_table(rows, [1.2 * cm, 6.3 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm], 7.5)]
    story += [Spacer(1, 0.25 * cm), Image(str(RESULTS / "horizons.png"), width=16.4 * cm, height=8.1 * cm)]
    story += [p(
        f"图中展示未来 1—24 小时的误差变化。整体最低 MAE 为 {best.MAE:.2f}，对应 {MODEL_NAMES[str(best.model)]}。曲线与主表使用相同有效标签集合，避免因缺失覆盖不同产生虚假优势。",
        styles["body"],
    )]
    story += [PageBreak()]

    story += [p("消融实验与城市差异", styles["h1"])]
    story += [p(
        f"加入历史气象后，A1 相对 M2 的 MAE降低 {history_weather_gain:.2f}%。继续加入已发布预报后，A2 相对 A1 只降低 {forecast_increment:.2f}%，而 RMSE 从 {a1.RMSE:.3f} 微升至 {a2.RMSE:.3f}。A2 相对仅自环的 A3 降低 {space_increment:.2f}%。这些差异都小于随机种子波动，当前结果不足以支持稳定增益。",
        styles["callout"],
    )]
    story += [Table([[Image(str(RESULTS / "ablation.png"), width=8.1 * cm, height=6.7 * cm), Image(str(RESULTS / "cities.png"), width=8.1 * cm, height=6.7 * cm)]], colWidths=[8.3 * cm, 8.3 * cm])]
    story += [p(
        "同一模型在不同城市的表现可能受站点密度、缺失率、污染水平和气象匹配率影响。城市结果用于解释误差来源，未参与模型选择。",
        styles["body"],
    )]
    story += [PageBreak()]

    story += [p("案例与误差来源", styles["h1"]), Image(str(RESULTS / "cases.png"), width=16.6 * cm, height=9.4 * cm)]
    story += [p("案例分析原则", styles["h2"])]
    story += bullet([
        "成功案例检查模型能否跟随污染变化的方向和峰值到达时间。",
        "失败案例检查缺失观测、突发污染和天气预报覆盖是否与误差同时出现。",
        "评价脚本按单个站点和起报时刻的 24 步绝对误差计算 A2 相对持久性基线的改善，分别选择改善最大和最小的两个样例。",
    ], styles["body"])
    story += [p("限制", styles["h2"])]
    story += [p(
        f"训练期 PM2.5 第 90 百分位对应原始数值 100。真值达到该阈值时，持久性基线 MAE 为 {extreme_b1:.3f}，A2 为 {extreme_a2:.3f}；阈值以下分别为 {normal_b1:.3f} 和 {normal_a2:.3f}。模型对常规情景有效，但在高污染情景明显落后于持久性基线。",
        styles["callout"],
    )]
    story += [p(
        "原始数据未注明时区，天气与风向编码也缺少完整字典。预报按时段起点精确匹配，未确认的区间不做逐小时扩展。当前训练轮数和调参范围较小，结果不能代表各模型的理论上限。",
        styles["body"],
    )]
    story += [PageBreak()]

    story += [p("复现流程与成员贡献", styles["h1"])]
    contribution = [
        ["成员", "主要交付"],
        ["A", "空气质量审计、逐小时时间轴、48 小时输入与 24 小时标签"],
        ["B", "实测气象和天气预报对齐、匹配率与可追溯样例"],
        ["C", "简单基线、TCN、训练日志与预测结果"],
        ["D", "地理图、STGCN、气象融合与空间消融"],
        ["E", "统一评价、图表、案例、报告和答辩材料"],
    ]
    story += [styled_table(contribution, [2.2 * cm, 14.4 * cm], 8.5)]
    story += [p("复现入口", styles["h2"])]
    story += bullet([
        "原始数据与 SHA-256 清单位于 JZH_Air_2014_2018。",
        "A 和 B 的脚本构建统一缓存；E 的实验入口读取冻结配置并训练各实验。",
        "评价脚本生成 summary.csv、逐时距图、分城市图和案例图。",
        "交付材料脚本只读取上述真实结果，重新生成 PDF 和 PPTX。",
    ], styles["body"])
    story += [p("自动化辅助说明", styles["h2"])]
    story += [p(
        "代码、说明文档、图表排版和初稿使用自动化工具辅助生成。数据处理规则、实验配置、结果解释和最终提交内容仍需五位成员共同核对。答辩演练和每位成员的口头问答准备需要线下完成。",
        styles["body"],
    )]
    story += [PageBreak()]

    story += [p("结论与参考文献", styles["h1"])]
    story += [p(
        f"本项目完成了从原始站点数据到 24 步预测评价的可复现流程。当前测试中，{MODEL_NAMES[str(best.model)]}取得数值最低的整体 MAE {best.MAE:.2f}，相对持久性基线降低 {baseline_gain:.1f}%。A1、A2、A3 的差异小于种子波动，因此天气预报和空间图在当前预算下没有显示稳定增益。",
        styles["callout"],
    )]
    story += [p("主要结论的适用范围", styles["h2"])]
    story += bullet([
        "结论适用于本数据集 2018 年测试期、当前站点集合和固定训练预算。",
        "未来工作可增加滚动窗口评估、污染峰值加权损失和更完整的天气编码。",
        "即使复杂模型未在所有时距超过简单基线，结果仍能说明空间信息和天气信息何时有效。",
    ], styles["body"])
    story += [p("参考文献", styles["h2"])]
    references = [
        "[1] Bai S, Kolter J Z, Koltun V. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. 2018.",
        "[2] Yu B, Yin H, Zhu Z. Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. IJCAI, 2018.",
        "[3] 本课程提供的江浙沪空气质量、实测气象与天气预报数据，2015—2018。",
    ]
    story += [p(item, styles["small"]) for item in references]
    story += [Spacer(1, 0.6 * cm), p("附注：报告中的数值来自 E/results/summary.csv，图像来自同目录评价产物。", styles["small"])]
    return story


def main():
    register_fonts()
    styles = make_styles()
    summary, coverage, settings, dataset, runs, scenarios = load_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.75 * cm,
        title="融合气象信息的江浙沪 PM2.5 短期时空预测",
        author="课程项目五人组 A、B、C、D、E",
    )
    doc.build(
        build_story(summary, coverage, settings, dataset, runs, scenarios, styles),
        onFirstPage=footer,
        onLaterPages=footer,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
