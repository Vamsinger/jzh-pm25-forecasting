// 从真实评价结果生成 10 页可编辑 PPTX，并通过 Artifact Tool 终检。
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..", "..");
const resultsDir = path.join(root, "E", "results");
const buildDir = path.join(scriptDir, ".build");
const receiptDir = path.join(root, "E", ".presentation-validation");
const renderDir = path.join(buildDir, "slides");
const stableOutput = path.join(scriptDir, "答辩幻灯片.pptx");
const runtimePython = process.env.RUNTIME_PYTHON;
const runtimeNodeModules = process.env.RUNTIME_NODE_MODULES;
const skillDir = process.env.PRESENTATION_SKILL_DIR;

if (!runtimePython || !runtimeNodeModules || !skillDir) {
  throw new Error("必须设置 RUNTIME_PYTHON、RUNTIME_NODE_MODULES 和 PRESENTATION_SKILL_DIR");
}
const { Presentation, PresentationFile } = await import(pathToFileURL(
  path.join(runtimeNodeModules, "@oai", "artifact-tool", "dist", "artifact_tool.mjs")
).href);
const { finalizePresentation, makeNativeBulletParagraphs } = await import(pathToFileURL(
  path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")
).href);

const COLORS = {
  navy: "#17324D",
  teal: "#176B87",
  rust: "#A84A32",
  ink: "#263746",
  gray: "#5D6B78",
  pale: "#F2F6F8",
  warm: "#FFF5F0",
  white: "#FFFFFF",
};
const FONT = "Microsoft YaHei";
const MODEL_NAMES = {
  B1: "持久性基线",
  B2: "日周期基线",
  M1: "TCN",
  M2: "STGCN",
  A1: "STGCN + 历史气象",
  A2: "STGCN + 历史气象 + 预报",
  A3: "完整模型（仅自环）",
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

async function readBytes(filePath) {
  const buffer = await fs.readFile(filePath);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? "18pt",
    bold: options.bold ?? false,
    color: options.color ?? COLORS.ink,
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: "shrinkText",
  };
  return shape;
}

function addTitle(slide, title, page) {
  addText(slide, title, { left: 70, top: 42, width: 1060, height: 58 }, {
    fontSize: "32pt",
    bold: true,
    color: COLORS.navy,
  });
  addText(slide, String(page).padStart(2, "0"), { left: 1165, top: 48, width: 55, height: 34 }, {
    fontSize: "13pt",
    bold: true,
    color: COLORS.teal,
  });
}

function addBody(slide, lines, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = makeNativeBulletParagraphs(lines, {
    marginLeftPoints: 18,
    hangingPoints: 9,
    spaceAfterPoints: 7,
  });
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? "18pt",
    color: options.color ?? COLORS.ink,
    autoFit: "shrinkText",
  };
  return shape;
}

function addTable(slide, values, position, widths, fontSize = 16) {
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: position.left,
    top: position.top,
    width: position.width,
    height: position.height,
    columnWidths: widths,
    values,
  });
  table.borders.assign({ style: "solid", fill: "#BAC6CF", width: 1 });
  table.styleOptions = { headerRow: true, bandedRows: true };
  for (let row = 0; row < values.length; row += 1) {
    for (let column = 0; column < values[0].length; column += 1) {
      const cell = table.getCell(row, column);
      cell.fill = row === 0 ? COLORS.navy : (row % 2 === 0 ? COLORS.pale : COLORS.white);
      cell.text.style = {
        typeface: FONT,
        fontSize: `${fontSize}pt`,
        bold: row === 0,
        color: row === 0 ? COLORS.white : COLORS.ink,
      };
    }
  }
  return table;
}

async function addImage(slide, imagePath, position, alt, fit = "contain") {
  slide.images.add({
    blob: await readBytes(imagePath),
    contentType: "image/png",
    alt,
    fit,
    position,
  });
}

async function main() {
  const required = ["summary.csv", "horizons.png", "cities.png", "cases.png", "ablation.png", "scenarios.csv", "coverage.json"];
  for (const name of required) {
    await fs.access(path.join(resultsDir, name));
  }
  const rows = parseCsv(await fs.readFile(path.join(resultsDir, "summary.csv"), "utf8"));
  const overall = rows.filter((row) => row.city === "全区域" && Number(row.horizon) === 0);
  const byCode = Object.fromEntries(overall.map((row) => [row.model, row]));
  const best = overall.reduce((left, right) => Number(left.MAE) < Number(right.MAE) ? left : right);
  const a2 = byCode.A2;
  const m2 = byCode.M2;
  const b1 = byCode.B1;
  const a1 = byCode.A1;
  const a3 = byCode.A3;
  const forecastIncrement = ((Number(a1.MAE) - Number(a2.MAE)) / Number(a1.MAE) * 100).toFixed(2);
  const historyWeatherGain = ((Number(m2.MAE) - Number(a1.MAE)) / Number(m2.MAE) * 100).toFixed(2);
  const spaceIncrement = ((Number(a3.MAE) - Number(a2.MAE)) / Number(a3.MAE) * 100).toFixed(2);
  const baselineGain = ((Number(b1.MAE) - Number(a2.MAE)) / Number(b1.MAE) * 100).toFixed(1);
  const coverage = JSON.parse(await fs.readFile(path.join(resultsDir, "coverage.json"), "utf8"));
  const scenarioRows = parseCsv(await fs.readFile(path.join(resultsDir, "scenarios.csv"), "utf8"));
  const extreme = scenarioRows.filter((row) => row.group === "训练期P90及以上");
  const normal = scenarioRows.filter((row) => row.group === "训练期P90以下");
  const average = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const extremeB1 = Number(extreme.find((row) => row.model_seed === "B1").MAE);
  const extremeA2 = average(extreme.filter((row) => row.model_seed.startsWith("A2_")).map((row) => Number(row.MAE)));
  const normalB1 = Number(normal.find((row) => row.model_seed === "B1").MAE);
  const normalA2 = average(normal.filter((row) => row.model_seed.startsWith("A2_")).map((row) => Number(row.MAE)));
  const settings = JSON.parse(await fs.readFile(path.join(root, "E", "settings.json"), "utf8"));
  const dataset = JSON.parse(await fs.readFile(path.join(root, "JZH_Air_2014_2018", "dataset_summary.json"), "utf8"));

  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // 1. 封面
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    await addImage(slide, path.join(root, "D", "station_graph.png"), { left: 675, top: 0, width: 605, height: 720 }, "江浙沪站点近邻图", "cover");
    addText(slide, "融合气象信息的江浙沪\nPM2.5 短期时空预测", { left: 70, top: 145, width: 570, height: 190 }, { fontSize: "40pt", bold: true, color: COLORS.navy });
    addText(slide, "未来 1—24 小时站点级预测", { left: 74, top: 365, width: 500, height: 50 }, { fontSize: "22pt", color: COLORS.teal });
    addText(slide, "时空数据挖掘课程项目\n五人组 A、B、C、D、E", { left: 74, top: 500, width: 460, height: 90 }, { fontSize: "17pt", color: COLORS.gray });
  }

  // 2. 任务与数据
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "研究问题与数据范围", 2);
    addText(slide, "输入", { left: 72, top: 142, width: 190, height: 40 }, { fontSize: "20pt", bold: true, color: COLORS.teal });
    addBody(slide, ["过去 48 小时污染物", "历史实测气象与缺失标记", "起报时已经发布的天气预报"], { left: 72, top: 187, width: 490, height: 190 });
    addText(slide, "输出与评价", { left: 72, top: 420, width: 250, height: 40 }, { fontSize: "20pt", bold: true, color: COLORS.teal });
    addBody(slide, ["各站未来连续 24 小时 PM2.5", "MAE、RMSE 与有效标签数", "逐时距和分城市分析"], { left: 72, top: 465, width: 490, height: 165 });
    await addImage(slide, path.join(root, "A", "outputs", "pm25_data_quality.png"), { left: 610, top: 130, width: 590, height: 455 }, "PM2.5 数据质量统计");
    addText(slide, `${dataset.city_count} 个城市，空气质量表 ${dataset.data.airquality.output_rows.toLocaleString()} 条记录`, { left: 670, top: 605, width: 490, height: 42 }, { fontSize: "16pt", color: COLORS.gray });
  }

  // 3. 数据边界
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "数据对齐与信息边界", 3);
    addTable(slide, [
      ["数据", "空间粒度", "可使用的时间"],
      ["空气质量", "监测站", "起报时刻及以前"],
      ["实测气象", "区县优先，城市回退", "起报时刻及以前"],
      ["天气预报", "区县或城市", "发布时间不晚于起报时刻"],
      ["PM2.5 标签", "监测站", "未来 1—24 小时，仅用于监督与评价"],
    ], { left: 74, top: 137, width: 1130, height: 280 }, [260, 390, 480], 16);
    addText(slide, "时间划分", { left: 74, top: 462, width: 200, height: 40 }, { fontSize: "20pt", bold: true, color: COLORS.teal });
    addBody(slide, ["2015—2016 年训练", "2017 年验证与选择配置", "2018 年冻结方案后测试"], { left: 74, top: 510, width: 430, height: 135 });
    addText(slide, "预报对齐", { left: 610, top: 462, width: 200, height: 40 }, { fontSize: "20pt", bold: true, color: COLORS.teal });
    addBody(slide, ["区分发布时间 time 与目标时段 time_period_start", "未确认的预报区间不做逐小时扩展", "保留预报新鲜度和缺失标记"], { left: 610, top: 510, width: 585, height: 145 });
  }

  // 4. 模型矩阵
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "模型与消融矩阵", 4);
    addTable(slide, [
      ["编号", "模型", "历史气象", "未来预报", "站点图"],
      ["B1", "持久性", "", "", ""],
      ["B2", "日周期", "", "", ""],
      ["M1", "TCN", "", "", ""],
      ["M2", "STGCN", "", "", "地理图"],
      ["A1", "STGCN + 历史气象", "有", "", "地理图"],
      ["A2", "完整模型", "有", "有", "地理图"],
      ["A3", "空间消融", "有", "有", "仅自环"],
    ], { left: 90, top: 130, width: 1100, height: 445 }, [120, 400, 190, 190, 200], 15);
    addText(slide, "TCN 和 STGCN 使用相同的 48 小时输入、24 步输出和评价样本。五层因果卷积覆盖 63 小时。A1、A2、A3 每次只改变一个因素。", { left: 100, top: 600, width: 1060, height: 70 }, { fontSize: "17pt", color: COLORS.gray });
  }

  // 5. 实验设置
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "实验设置", 5);
    addTable(slide, [
      ["项目", "设置"],
      ["历史与预测", `${settings.history} 小时 / ${settings.horizons} 小时`],
      ["随机种子", settings.seeds.join("、")],
      ["轮数与批次", `${settings.epochs} 轮 / ${settings.batch}`],
      ["训练起报步长", `${settings.train_origin_stride} 小时`],
      ["候选学习率", settings.learning_rates.join("、")],
      ["候选邻居数", settings.neighbors.join("、")],
      ["运行设备", "NVIDIA GeForce RTX 5080，16 GB"],
    ], { left: 90, top: 130, width: 690, height: 455 }, [280, 410], 16);
    addText(slide, "公平比较", { left: 850, top: 150, width: 250, height: 40 }, { fontSize: "22pt", bold: true, color: COLORS.teal });
    addBody(slide, ["所有模型使用相同时间划分", "归一化统计只来自训练集", "验证集选择学习率与邻接规模", "测试结果不再用于调参", "三个随机种子报告均值和波动"], { left: 850, top: 215, width: 355, height: 330 }, { fontSize: "17pt" });
    addText(slide, "当前训练预算较小，结论限于本次固定配置。", { left: 850, top: 585, width: 340, height: 60 }, { fontSize: "16pt", color: COLORS.rust });
  }

  // 6. 主结果表
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "整体测试结果", 6);
    const resultValues = [["编号", "模型", "MAE", "RMSE", "有效标签数"]];
    for (const code of Object.keys(MODEL_NAMES)) {
      const row = byCode[code];
      if (!row) continue;
      const mae = row.MAE_sd === "" ? Number(row.MAE).toFixed(2) : `${Number(row.MAE).toFixed(2)} ± ${Number(row.MAE_sd).toFixed(2)}`;
      const rmse = row.RMSE_sd === "" ? Number(row.RMSE).toFixed(2) : `${Number(row.RMSE).toFixed(2)} ± ${Number(row.RMSE_sd).toFixed(2)}`;
      resultValues.push([code, MODEL_NAMES[code], mae, rmse, Number(row.n).toLocaleString()]);
    }
    addTable(slide, resultValues, { left: 55, top: 120, width: 1170, height: 430 }, [100, 420, 220, 220, 210], 14);
    addText(slide, `${MODEL_NAMES[best.model]}取得数值最低整体 MAE ${Number(best.MAE).toFixed(2)}`, { left: 90, top: 585, width: 650, height: 55 }, { fontSize: "22pt", bold: true, color: COLORS.rust });
    addText(slide, `完整模型 A2 相对持久性基线降低 ${baselineGain}%`, { left: 780, top: 590, width: 420, height: 45 }, { fontSize: "18pt", color: COLORS.teal });
  }

  // 7. 逐时距曲线
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "未来 1—24 小时误差", 7);
    await addImage(slide, path.join(resultsDir, "horizons.png"), { left: 70, top: 112, width: 1140, height: 500 }, "各模型逐时距误差");
    addText(slide, "各曲线使用相同的站点、起报时刻和有效标签集合。", { left: 150, top: 630, width: 980, height: 42 }, { fontSize: "17pt", color: COLORS.gray });
  }

  // 8. 城市与消融
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "消融与重点情景", 8);
    await addImage(slide, path.join(resultsDir, "ablation.png"), { left: 45, top: 115, width: 790, height: 505 }, "气象、预报和空间消融结果");
    addText(slide, "消融结论", { left: 875, top: 150, width: 260, height: 45 }, { fontSize: "22pt", bold: true, color: COLORS.teal });
    addBody(slide, [
      `历史气象 A1 相对 M2 降低 ${historyWeatherGain}%`,
      `天气预报 A2 相对 A1 降低 ${forecastIncrement}%`,
      `空间传播 A2 相对 A3 降低 ${spaceIncrement}%`,
      "上述差异均小于随机种子波动",
      "消融差异均小于随机种子波动",
    ], { left: 875, top: 215, width: 330, height: 305 }, { fontSize: "17pt" });
    addText(slide, `高污染真值达到训练期 P90 时\n持久性 MAE ${extremeB1.toFixed(2)}\nA2 MAE ${extremeA2.toFixed(2)}\n\nP90 以下\n持久性 ${normalB1.toFixed(2)}，A2 ${normalA2.toFixed(2)}`, { left: 875, top: 510, width: 330, height: 155 }, { fontSize: "16pt", color: COLORS.rust });
  }

  // 9. 案例与限制
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "成功、失败案例与限制", 9);
    await addImage(slide, path.join(resultsDir, "cases.png"), { left: 45, top: 115, width: 790, height: 505 }, "测试集成功与失败案例");
    addText(slide, "解释重点", { left: 870, top: 140, width: 260, height: 45 }, { fontSize: "22pt", bold: true, color: COLORS.teal });
    addBody(slide, ["按 24 步 MAE 相对持久性的改善排序", "左图选择改善最大样例", "右图选择改善最小样例", "同时检查峰值、缺失与预报覆盖"], { left: 870, top: 200, width: 335, height: 190 }, { fontSize: "16pt" });
    addText(slide, "数据限制", { left: 870, top: 420, width: 260, height: 45 }, { fontSize: "22pt", bold: true, color: COLORS.teal });
    addBody(slide, ["原始时区未注明", "天气和风向编码字典不完整", "预报只按确认的时段起点匹配", "训练轮数与调参预算有限"], { left: 870, top: 475, width: 335, height: 175 }, { fontSize: "16pt" });
  }

  // 10. 结论与分工
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addTitle(slide, "结论与成员贡献", 10);
    addText(slide, `当前测试中，${MODEL_NAMES[best.model]}取得数值最低整体 MAE ${Number(best.MAE).toFixed(2)}`, { left: 75, top: 125, width: 1110, height: 60 }, { fontSize: "25pt", bold: true, color: COLORS.rust });
    addText(slide, "结论", { left: 75, top: 235, width: 200, height: 40 }, { fontSize: "21pt", bold: true, color: COLORS.teal });
    addBody(slide, ["完整模型完成污染物、实测气象、天气预报和空间图融合", "天气预报与空间图的差异小于随机种子波动", "结论适用于 2018 年测试期和当前训练预算"], { left: 75, top: 285, width: 545, height: 200 }, { fontSize: "17pt" });
    addText(slide, "分工", { left: 680, top: 235, width: 200, height: 40 }, { fontSize: "21pt", bold: true, color: COLORS.teal });
    addBody(slide, ["A：空气质量与样本", "B：气象与预报对齐", "C：基线与 TCN", "D：空间图与 STGCN", "E：评价、报告与答辩材料"], { left: 680, top: 285, width: 485, height: 235 }, { fontSize: "17pt" });
    addText(slide, "代码与文稿使用自动化工具辅助。五位成员需共同核对结论，并线下完成答辩演练。", { left: 110, top: 595, width: 1060, height: 55 }, { fontSize: "16pt", color: COLORS.gray });
  }

  await fs.mkdir(renderDir, { recursive: true });
  await fs.mkdir(receiptDir, { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(renderDir, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await preview.arrayBuffer()));
  }

  const candidatePath = path.join(buildDir, "candidate.pptx");
  await fs.mkdir(buildDir, { recursive: true });
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
  const revision = Date.now();
  const validatedPath = path.join(buildDir, `validated-${revision}.pptx`);
  const result = await finalizePresentation({
    explicitTotalSlideCount: 10,
    workspaceDir: root,
    candidatePath,
    finalPath: validatedPath,
    pythonExecutable: runtimePython,
    integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
    layoutArgs: [
      "--expected-slide-size-emu", "12192000,6858000",
      "--validate-bullet-geometry",
      "--validate-heading-fit",
      ...[3, 4, 5, 6].flatMap((number) => ["--require-native-table-slide", String(number)]),
    ],
    requiredNativeTableOwnerSlides: [3, 4, 5, 6],
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(receiptDir, `validation-${revision}.json`),
  });
  await fs.copyFile(validatedPath, stableOutput);
  console.log(JSON.stringify({ stableOutput, result, coverage }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
