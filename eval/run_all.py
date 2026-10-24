"""完整评测：python -m eval.run_all [--skip-chunking]

顺序：开发集标定 → 检索 → 改写 → 引用 → 切分；报告写到 eval/reports/。
"""

import argparse
import asyncio
import sys

from clausetrail.services import Services
from eval import run_chunking, run_citation, run_retrieval, run_rewrite
from eval.common import DATASETS, REPORTS, backends, header, run_with_services, write_text


async def run(services: Services, skip_chunking: bool) -> dict:
    summary = {
        "backends": backends(services),
        "calibration": await run_retrieval.calibrate(services),
        "retrieval": await run_retrieval.evaluate(services),
        "rewrite": await run_rewrite.evaluate(services),
        "citation": await run_citation.evaluate(services),
    }
    if not skip_chunking:
        summary["chunking"] = await asyncio.to_thread(run_chunking.evaluate)
    write_text(REPORTS / "summary.md", render(summary))
    return summary


def render(s: dict) -> str:
    ret, term = s["retrieval"]["overall"], s["retrieval"]["by_type"].get("term")
    docs = s["retrieval"]["documents"] + s["calibration"]["dev"]["documents"]
    dev = s["calibration"]["dev"]
    queries = (
        s["retrieval"]["queries"]
        + s["rewrite"]["queries"]
        + s["citation"]["queries"]
        + dev["answerable"]
        + dev["unanswerable"]
    )

    # 四种配置都列出来，不只挑「纯向量 → 混合 + 精排」这一对
    def four(m: dict, key: str) -> str:
        return " / ".join(f"{m[c][key]:.4f}" for c in ("dense", "bm25", "rrf", "rerank"))

    configs = "纯向量 / 纯 BM25 / RRF / RRF + 精排"
    rows = [
        ("检索", f"MRR@10：{configs}", four(ret, "mrr@10")),
        ("检索", f"Recall@5：{configs}", four(ret, "recall@5")),
    ]
    if term:
        rows.append(("检索", f"术语问题 MRR@10：{configs}", four(term, "mrr@10")))
    rows += [
        (
            "拒答",
            "开发集标定的阈值 τ / RRF k",
            f"{s['calibration']['refusal_threshold']} / {s['calibration']['rrf_k']}",
        ),
        (
            "改写",
            "改写准确率 / 话题判断准确率",
            f"{s['rewrite']['rewrite_accuracy']:.2%} / {s['rewrite']['label_accuracy']:.2%}",
        ),
        ("引用", "作答问题中引用片段包含 gold 原句", f"{s['citation']['gold_supported_rate']:.2%}"),
    ]
    if "chunking" in s:
        f, st = s["chunking"]["overall"]["fixed"], s["chunking"]["overall"]["structure"]
        rows += [
            ("切分", "句子截断率：固定长度 → 结构感知", f"{f['sentence']['rate']:.2%} → {st['sentence']['rate']:.2%}"),
            ("切分", "列表截断率：固定长度 → 结构感知", f"{f['list']['rate']:.2%} → {st['list']['rate']:.2%}"),
        ]
    lines = [header("评测汇总", docs, queries, s["backends"]), "| 项目 | 指标 | 结果 |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines += ["", "明细：retrieval.md · rewrite.md · citation.md · chunking.md · calibration.json"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="在样例上跑完整评测")
    parser.add_argument("--skip-chunking", action="store_true", help="跳过切分评测（它要重新解析样例，较慢）")
    args = parser.parse_args()
    if not (DATASETS / "retrieval_test.jsonl").exists():
        print("还没有评测集，先运行 python -m eval.gen.synthesize")
        return 1
    return run_with_services(lambda services: run(services, args.skip_chunking), "embedder", "reranker", "llm")


if __name__ == "__main__":
    sys.exit(main())
