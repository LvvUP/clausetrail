"""检查开发集与测试集是否泄漏：python -m eval.gen.check_leakage"""

import sys

from eval.common import DATASETS, read_jsonl
from eval.metrics import jaccard, norm, shingles

NEAR_DUPLICATE = 0.9
TEST_FILES = ("retrieval_test", "rewrite_test", "citation_sample")


def _docs(row: dict) -> set[str]:
    docs = {g["doc_id"] for g in row.get("gold", [])}
    return docs | ({row["doc_id"]} if row.get("doc_id") else set())


def check(dev: list[dict], tests: dict[str, list[dict]]) -> list[str]:
    """文档 ID 不相交、gold 句子不重叠、问题不近似重复（字符 3-gram Jaccard ≥ 0.9），qid 和问题在各集合内不重复。"""
    problems = []
    dev_docs = set().union(*(_docs(r) for r in dev)) if dev else set()
    dev_spans = {norm(g["span"]) for r in dev for g in r.get("gold", [])}
    dev_shingles = [(r["qid"], shingles(r["query"])) for r in dev]
    for name, rows in {"dev": dev, **tests}.items():
        if len({r["qid"] for r in rows}) != len(rows):
            problems.append(f"{name}: qid 重复")
        # 多轮样本的同一句追问可以接在不同的上一问后面，按「历史 + 追问」判重
        queries = [" / ".join([*(h["question"] for h in r.get("history", [])), r["query"]]) for r in rows]
        # 一句 gold 原文只能对应一道题，否则就是换个问法重复出题
        golds = [(g["doc_id"], norm(g["span"])) for r in rows for g in r.get("gold", [])]
        if len(set(golds)) != len(golds):
            problems.append(f"{name}: 同一句 gold 原文对应了多道题（{len(golds) - len(set(golds))} 处）")
        if len(set(queries)) != len(queries):
            problems.append(f"{name}: 有完全相同的问题")
    for name, rows in tests.items():
        for r in rows:
            if _docs(r) & dev_docs:
                problems.append(f"{name}/{r['qid']}: 与开发集共享文档")
            if any(norm(g["span"]) in dev_spans for g in r.get("gold", [])):
                problems.append(f"{name}/{r['qid']}: gold 句子出现在开发集")
            sh = shingles(r["query"])
            for qid, other in dev_shingles:
                if jaccard(sh, other) >= NEAR_DUPLICATE:
                    problems.append(f"{name}/{r['qid']}: 与开发集 {qid} 近似重复")
    return problems


def main() -> int:
    dev = read_jsonl(DATASETS / "dev.jsonl")
    tests = {n: read_jsonl(DATASETS / f"{n}.jsonl") for n in TEST_FILES if (DATASETS / f"{n}.jsonl").exists()}
    problems = check(dev, tests)
    for p in problems[:20]:
        print(p)
    print(f"{len(problems)} 个问题" if problems else "未发现泄漏")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
