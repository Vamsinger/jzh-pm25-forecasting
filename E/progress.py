"""登记真实任务状态，重新生成五份个人记录和一份总表。"""
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "E" / "progress.json"
ROLES = dict(zip("ABCDE", ["空气质量与样本", "气象匹配", "基线与时间模型", "站点图与时空模型", "统一评价与交付"]))

def record(member, task, status, evidence="", at=None):
    clock = at or datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    rows = json.loads(DB.read_text(encoding="utf8")) if DB.exists() else []
    row = next((r for r in rows if r["member"] == member and r["task"] == task), None)
    if row is None:
        row = dict(member=member, task=task, start=clock, end="")
        rows.append(row)
    row.update(status=status, evidence=evidence, updated=clock)
    row["end"] = clock if status == "已完成" else ""
    DB.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    def table(items):
        lines = ["| 分工 | 任务 | 状态 | 开始（北京时间） | 完成（北京时间） | 产物或说明 |", "|---|---|---|---|---|---|"]
        for r in items:
            lines.append("| "+" | ".join(r[k] for k in ["member","task","status","start","end","evidence"])+" |")
        return "\n".join(lines)
    top = ["# 总体流程进展", "", "时间均为北京时间。A～E 代表任务归属，记录的是本次自动执行和验收的时间。", "", "## 当前分工", "", "| 分工 | 当前工作或最近验收 | 状态 | 最近更新 |", "|---|---|---|---|"]
    for who, role in ROLES.items():
        own = [r for r in rows if r["member"] == who]
        current = [r for r in own if r["status"] != "已完成"] or own[-1:]
        for r in current:
            top.append(f"| [{who} · {role}]({who}/任务进度.md) | {r['task']} | {r['status']} | {r['updated']} |")
        (ROOT/who/"任务进度.md").write_text(f"# {who} · {role}\n\n"+table(own)+"\n",encoding="utf8")
    (ROOT/"总体流程进展.md").write_text("\n".join(top)+"\n\n## 阶段记录\n\n"+table(rows)+"\n",encoding="utf8")
if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for key in ["member","task","status","evidence"]: p.add_argument(key)
    args=p.parse_args()
    record(**vars(args))

