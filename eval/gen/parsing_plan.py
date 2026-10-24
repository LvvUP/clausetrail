"""解析准确率抽检方案：生成分层抽样表和标注模板。python -m eval.gen.parsing_plan"""

import csv
import sys

from eval.common import DATASETS

STRATA = {
    "native": (
        {"寿险": 10, "重疾": 10, "医疗": 10, "意外": 5, "财产险": 5},
        "官网公开的原生 PDF 条款（data/corpus_manifest.csv 中的人身险来源 + 行业协会财产险库）",
    ),
    "scanned": (
        {"寿险": 6, "重疾": 6, "医疗": 6, "意外": 6, "财产险": 6},
        "扫描件：公开来源里没有找到真实的扫描条款，需要打印后扫描，或使用标注为合成的扫描件",
    ),
    "complex_table": (
        {"重疾": 10, "医疗": 10, "寿险": 5, "财产险": 5},
        "含费率表、给付比例表、病种表的条款",
    ),
}
SELECTION = "按 stratum × 险种分层，从来源池中用固定种子 20261019 随机抽取"
METRICS = "标题层级正确率; 正文字符 F1; 表格单元格准确率（无表格时不计）"
PASS_RULE = "三项同时达标才算该文档解析正确；建议阈值 0.95 / 0.98 / 0.95，可在报告中调整"
COLUMNS = [
    "sample_id",
    "stratum",
    "product_line",
    "source_pool",
    "selection_rule",
    "file_url",
    "metrics",
    "pass_rule",
    "annotator",
    "status",
]


def rows() -> list[dict]:
    out = []
    for stratum, (lines, pool) in STRATA.items():
        for line, n in lines.items():
            for _ in range(n):
                out.append(
                    {
                        "sample_id": f"ps-{len(out) + 1:03d}",
                        "stratum": stratum,
                        "product_line": line,
                        "source_pool": pool,
                        "selection_rule": SELECTION,
                        "file_url": "",
                        "metrics": METRICS,
                        "pass_rule": PASS_RULE,
                        "annotator": "",
                        "status": "待抽样",
                    }
                )
    return out


HUMAN_COLUMNS = ("file_url", "annotator", "status")


def write() -> int:
    path = DATASETS / "parsing_sample_plan.csv"
    data = rows()
    # 重新生成时保留人工填写的列（抽中的文件、标注人、进度），按 sample_id 对回去，不清掉已经做的抽检
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            old = {r["sample_id"]: r for r in csv.DictReader(f)}
        for r in data:
            prev = old.get(r["sample_id"], {})
            r.update({c: prev[c] for c in HUMAN_COLUMNS if prev.get(c)})
    DATASETS.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(data)
    return len(data)


if __name__ == "__main__":
    print(f"parsing_sample_plan.csv: {write()} 行")
    sys.exit(0)
