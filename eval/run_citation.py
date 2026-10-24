"""引用评测：python -m eval.run_citation

自动指标是近似值：作答的问题里，被引用的片段至少有一条包含 gold 原句的比例。
人工抽检在 reports/citation_review.jsonl 的 human_ok 字段里填 true / false。
"""

import contextlib
import sys
from collections import Counter

from clausetrail.answer.citations import MARKER, normalize_markers
from clausetrail.chat import ChatService
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
    write_jsonl,
    write_text,
)
from eval.metrics import is_hit
from eval.run_retrieval import apply_calibration

SNIPPET_CHARS = 300


def _rate(n: int, total: int) -> float:
    return round(n / total, 4) if total else 0.0


async def evaluate(services: Services) -> dict:
    rows = read_jsonl(DATASETS / "citation_sample.jsonl")
    ensure_indexed(services, rows)
    apply_calibration(services)
    chat = ChatService(services)
    # 人工抽检结果按 qid 保留：重跑时回答没变，之前填的 human_ok 照样有效；回答变了才清空，等人重新看
    review_path = REPORTS / "citation_review.jsonl"
    previous = {x["qid"]: x for x in read_jsonl(review_path)} if review_path.exists() else {}
    # 引用编号是否有效要看模型的原始输出：发给前端的文本已经删掉了越界编号，再检查必然全部有效
    raw_parts: list[str] = []
    original_stream = services.llm.stream

    async def tapped_stream(*args, **kwargs):
        # 外层被 chat 用 aclosing 提前关掉时，把内层的原始流一起关掉
        async with contextlib.aclosing(original_stream(*args, **kwargs)) as inner:
            async for delta in inner:
                raw_parts.append(delta)
                yield delta

    async def review_one(r: dict) -> dict:
        raw_parts.clear()
        session_id, citations, parts, done, error = None, [], [], {}, None
        async for name, data in chat.stream(r["query"]):
            if name == "meta":
                session_id = data["session_id"]
            elif name == "retrieval":
                citations = data["citations"]
            elif name == "delta":
                parts.append(data["text"])
            elif name == "error":
                error = data["code"]
            elif name == "done":
                done = data
        await chat.drain()
        if session_id:
            services.db.delete_session(session_id)
        answer = "".join(parts)
        # 命中缓存时没有调用模型，原始输出就是回放的回答
        raw = normalize_markers("".join(raw_parts) or answer)
        prev = previous.get(r["qid"])
        human_ok = prev.get("human_ok") if prev and prev.get("answer") == answer else None
        used = [c for c in citations if c["n"] in done.get("used_citations", [])]
        return {
            "qid": r["qid"],
            "query": r["query"],
            "status": "error" if error else done.get("status"),
            "refusal_reason": done.get("refusal_reason"),
            "answer": answer,
            "markers_valid": all(1 <= int(m) <= len(citations) for m in MARKER.findall(raw)),
            "used": [
                {k: c[k] for k in ("n", "file", "section_path", "page_start")}
                | {"snippet": c["snippet"][:SNIPPET_CHARS]}
                for c in used
            ],
            "gold_supported": any(is_hit(c["doc_id"], c["snippet"], r["gold"]) for c in used),
            "gold": r["gold"],
            "human_ok": human_ok,
        }

    # 中途抛错时也要把替换掉的 stream 还原，不留在 services 上
    services.llm.stream = tapped_stream
    try:
        review = [await review_one(r) for r in rows]
    finally:
        del services.llm.stream
    answered = [x for x in review if x["status"] == "answered"]
    result = {
        "queries": len(rows),
        "documents": len(gold_docs(rows)),
        "backends": backends(services),
        "status": dict(Counter(x["status"] for x in review)),
        "refusal_reasons": dict(Counter(x["refusal_reason"] for x in review if x["refusal_reason"])),
        "answered_rate": _rate(len(answered), len(review)),
        "markers_valid_rate": _rate(sum(x["markers_valid"] for x in answered), len(answered)),
        "cited_rate": _rate(sum(bool(x["used"]) for x in answered), len(answered)),
        "gold_supported_rate": _rate(sum(x["gold_supported"] for x in answered), len(answered)),
        "human_checked": sum(x["human_ok"] is not None for x in review),
    }
    write_jsonl(REPORTS / "citation_review.jsonl", review)
    write_json(REPORTS / "citation.json", result)
    write_text(REPORTS / "citation.md", render(result))
    return result


def render(res: dict) -> str:
    lines = [
        header("引用评测", res["documents"], res["queries"], res["backends"]),
        "自动指标只是近似：被引用的片段包含 gold 原句，说明引用落在了正确的条款上，不说明回答本身无误。",
        "逐条结果在 citation_review.jsonl，人工抽检时在 human_ok 字段填 true / false。",
        "",
        f"- 作答比例：{res['answered_rate']:.2%}",
        f"- 状态分布：{res['status']}；拒答原因：{res['refusal_reasons'] or '无'}",
        f"- 作答问题中引用编号全部有效：{res['markers_valid_rate']:.2%}",
        f"- 作答问题中至少引用一条片段：{res['cited_rate']:.2%}",
        f"- 作答问题中引用片段包含 gold 原句：{res['gold_supported_rate']:.2%}",
        f"- 已人工抽检：{res['human_checked']} 条",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(run_with_services(evaluate, "embedder", "reranker", "llm"))
