"""检索评测：先在开发集上标定 RRF k 和拒答阈值，再在测试集上比较四种配置。python -m eval.run_retrieval"""

import json
import sys

from clausetrail.retrieval.fusion import rrf
from clausetrail.retrieval.pipeline import Retriever
from clausetrail.services import Services
from eval.common import (
    DATASETS,
    REPORTS,
    backends,
    ensure_indexed,
    gold_docs,
    header,
    read_jsonl,
    run_with_services,
    write_json,
    write_text,
)
from eval.metrics import best_threshold, is_hit, ranking_metrics

CONFIGS = {"dense": "纯向量", "bm25": "纯 BM25", "rrf": "混合（RRF）", "rerank": "混合 + 精排"}
TYPE_NAMES = {"general": "普通问题", "term": "术语问题"}
SUBTYPE_NAMES = {
    "section": "章节题",
    "detail": "细节题",
    "exclusion": "责任免除题",
    "definition": "术语定义题",
    "llm": "大模型出题",
}
K_SWEEP = (10, 30, 60, 100)


async def collect(services: Services, rows: list[dict]) -> list[dict]:
    """每条 query 走一遍线上同一套检索，记录各路排序和每个片段是否命中 gold。"""
    retriever = Retriever(services)
    runs = []
    for r in rows:
        res = await retriever.retrieve(r["query"])
        chunks = services.store.get_chunks(list(set(res.dense_ids) | set(res.bm25_ids)))
        runs.append(
            {
                "qid": r["qid"],
                "type": r["type"],
                "subtype": r.get("subtype", r["type"]),
                "hit": {cid: is_hit(c.doc_id, c.text, r["gold"]) for cid, c in chunks.items()},
                "dense": res.dense_ids,
                "bm25": res.bm25_ids,
                "rrf": res.fused_ids,
                "rerank": [c.chunk.chunk_id for c in res.candidates],
                "top_score": res.top_score,
            }
        )
    return runs


def _hits(run: dict, ids: list[str]) -> list[bool]:
    return [run["hit"].get(cid, False) for cid in ids]


def _sweep(runs: list[dict], k: int) -> float:
    fused = [[cid for cid, _, _ in rrf({"dense": r["dense"], "bm25": r["bm25"]}, k=k)] for r in runs]
    return ranking_metrics([_hits(r, ids) for r, ids in zip(runs, fused, strict=True)])["mrr@10"]


async def calibrate(services: Services) -> dict:
    settings = services.settings
    dev = read_jsonl(DATASETS / "dev.jsonl")
    ensure_indexed(services, dev)
    runs = await collect(services, dev)
    answerable = [run for run, r in zip(runs, dev, strict=True) if r["gold"]]
    sweep = {k: _sweep(answerable, k) for k in K_SWEEP}
    default = settings.rrf_k
    # MRR 并列时保留默认 k：开发集很小，没有明显收益就不改默认值
    best_k = max(sweep, key=lambda k: (sweep[k], k == default))
    # k 变了，进入精排的前 20 条也会变，精排最高分要用新 k 重新跑一遍再拿来标定 τ
    if best_k != settings.rrf_k:
        settings.rrf_k = best_k
        runs = await collect(services, dev)
    # τ 只用开发集拟合：可回答问题和无关问题各自的精排最高分，测试集不参与
    ans = [run["top_score"] for run, r in zip(runs, dev, strict=True) if r["gold"]]
    una = [run["top_score"] for run, r in zip(runs, dev, strict=True) if not r["gold"]]
    fit = best_threshold(ans, una)
    settings.refusal_threshold = fit["threshold"] if fit else settings.refusal_threshold
    calibration = {
        "rrf_k": best_k,
        "rrf_k_sweep_mrr@10": {str(k): v for k, v in sweep.items()},
        "refusal_threshold": settings.refusal_threshold,
        "refusal_fit": fit,
        "dev": {"answerable": len(ans), "unanswerable": len(una), "documents": len(gold_docs(dev))},
        "backends": backends(services),
        "kb_version": services.db.kb_version(),
    }
    write_json(REPORTS / "calibration.json", calibration)
    return calibration


def apply_calibration(services: Services) -> dict | None:
    """测试脚本只读标定结果，不再调参。"""
    path = REPORTS / "calibration.json"
    if not path.exists():
        return None
    cal = json.loads(path.read_text(encoding="utf-8"))
    # 标定结果只对标定时那组模型有效：换了模型还沿用旧的 τ 和 k，拒答率和引用指标都会失真
    if cal.get("backends") != backends(services):
        print(f"[评测] calibration.json 是用 {cal.get('backends')} 标定的，和当前模型不一致，不套用；请先重新标定")
        return None
    services.settings.rrf_k, services.settings.refusal_threshold = cal["rrf_k"], cal["refusal_threshold"]
    return cal


async def evaluate(services: Services) -> dict:
    rows = read_jsonl(DATASETS / "retrieval_test.jsonl")
    ensure_indexed(services, rows)
    cal = apply_calibration(services)
    runs = await collect(services, rows)

    def metrics(subset: list[dict]) -> dict:
        return {c: ranking_metrics([_hits(r, r[c]) for r in subset]) for c in CONFIGS}

    tau = services.settings.refusal_threshold
    result = {
        "queries": len(rows),
        "documents": len(gold_docs(rows)),
        "backends": backends(services),
        "rrf_k": services.settings.rrf_k,
        "refusal_threshold": tau,
        "calibrated": cal is not None,
        "overall": metrics(runs),
        "by_type": {t: metrics([r for r in runs if r["type"] == t]) for t in sorted({r["type"] for r in runs})},
        "by_subtype": {
            t: metrics([r for r in runs if r["subtype"] == t]) for t in sorted({r["subtype"] for r in runs})
        },
        "false_refusal_rate": round(sum(r["top_score"] < tau for r in runs) / len(runs), 4) if runs else 0.0,
    }
    write_json(REPORTS / "retrieval.json", result)
    write_text(REPORTS / "retrieval.md", render(result))
    return result


def render(res: dict) -> str:
    cal = "（在开发集上标定，见 calibration.json）" if res["calibrated"] else "（未标定，使用配置默认值）"
    lines = [
        header("检索评测", res["documents"], res["queries"], res["backends"]),
        f"RRF k = {res['rrf_k']}，拒答阈值 τ = {res['refusal_threshold']}{cal}",
        "",
        "命中规则：召回片段与 gold 来自同一份文档，且（去掉空白后）包含 gold 原句；Recall@k 记前 k 名里是否命中。",
        "",
    ]
    groups = [
        ("全部", res["overall"]),
        *((TYPE_NAMES.get(t, t), m) for t, m in res["by_type"].items()),
        *((f"其中：{SUBTYPE_NAMES.get(t, t)}", m) for t, m in res.get("by_subtype", {}).items()),
    ]
    for title, m in groups:
        lines += [f"## {title}", "", "| 配置 | n | MRR@10 | Recall@5 | Recall@10 |", "| --- | --- | --- | --- | --- |"]
        for c, name in CONFIGS.items():
            x = m[c]
            lines.append(f"| {name} | {x['n']} | {x['mrr@10']:.4f} | {x['recall@5']:.4f} | {x['recall@10']:.4f} |")
        lines.append("")
    lines.append(f"测试集上精排最高分低于 τ 的比例（可回答问题被误拒）：{res['false_refusal_rate']:.2%}")
    return "\n".join(lines) + "\n"


async def run(services: Services) -> dict:
    await calibrate(services)
    return await evaluate(services)


if __name__ == "__main__":
    sys.exit(run_with_services(run, "embedder", "reranker"))
