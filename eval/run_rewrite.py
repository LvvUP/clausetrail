"""改写评测：python -m eval.run_rewrite"""

import sys
from collections import Counter

from clausetrail.conversation import continuity, memory
from clausetrail.conversation.rewriter import rewrite
from clausetrail.services import Services
from eval.common import DATASETS, REPORTS, backends, header, read_jsonl, run_with_services, write_json, write_text

CATEGORY_NAMES = {"continue": "承接", "switch": "话题切换", "pronoun": "指代", "ellipsis": "省略"}


def _rate(xs: list[dict], key: str) -> float:
    return round(sum(bool(x[key]) for x in xs) / len(xs), 4) if xs else 0.0


async def evaluate(services: Services) -> dict:
    cfg = services.settings
    rows = read_jsonl(DATASETS / "rewrite_test.jsonl")
    results = []
    for r in rows:
        turns = [{"question": h["question"], "standalone_query": h["question"], "answer": ""} for h in r["history"]]
        decision = await continuity.judge(
            r["query"], turns[-1]["standalone_query"], services.embedder, cfg.continuity_similarity_threshold
        )
        out = r["query"]
        if decision.rewrite:
            window = memory.window(turns, cfg.history_window_turns)
            out = await rewrite(services.llm, cfg.llm_rewrite_model, r["query"], window, "")
        predicted = "switch" if decision.topic == continuity.SWITCH else "continue"
        # 承接类：必须改写，且改写结果带上产品简称和追问的术语；话题切换类：不应改写
        if r["label"] == "switch":
            ok = out == r["query"]
        else:
            ok = out != r["query"] and all(m in out for m in r["must_include"])
        results.append(
            {
                "qid": r["qid"],
                "category": r["category"],
                "history": r["history"][-1]["question"],
                "query": r["query"],
                "label": r["label"],
                "predicted": predicted,
                "label_ok": predicted == r["label"],
                "reason": decision.reason,
                "rewritten": out,
                "ok": ok,
            }
        )
    by_category = {}
    for c in CATEGORY_NAMES:
        xs = [x for x in results if x["category"] == c]
        if xs:
            by_category[c] = {
                "n": len(xs),
                "rewrite_accuracy": _rate(xs, "ok"),
                "label_accuracy": _rate(xs, "label_ok"),
            }
    result = {
        "queries": len(rows),
        "documents": len({r["doc_id"] for r in rows}),
        "backends": backends(services),
        "rewrite_accuracy": _rate(results, "ok"),
        "label_accuracy": _rate(results, "label_ok"),
        "by_category": by_category,
        "triggers": dict(Counter(x["reason"] for x in results)),
        "failures": [x for x in results if not x["ok"]][:20],
    }
    write_json(REPORTS / "rewrite.json", result)
    write_text(REPORTS / "rewrite.md", render(result))
    return result


def render(res: dict) -> str:
    lines = [
        header("多轮改写评测", res["documents"], res["queries"], res["backends"]),
        "评分规则：承接、指代、省略三类必须改写，且改写结果包含产品简称和追问的术语；话题切换类不应改写。"
        "话题判断准确率看 continue / switch 标签是否判对。",
        "",
        f"- 改写准确率：{res['rewrite_accuracy']:.2%}",
        f"- 话题判断准确率：{res['label_accuracy']:.2%}",
        "",
        "| 类别 | n | 改写准确率 | 话题判断准确率 |",
        "| --- | --- | --- | --- |",
    ]
    for c, m in res["by_category"].items():
        lines.append(f"| {CATEGORY_NAMES[c]} | {m['n']} | {m['rewrite_accuracy']:.2%} | {m['label_accuracy']:.2%} |")
    lines += ["", "触发原因：" + "，".join(f"{k} {v}" for k, v in sorted(res["triggers"].items())), ""]
    if res["backends"]["llm"].startswith("fake") or res["backends"]["embedder"].startswith("fake"):
        lines += [
            "**读数说明（离线示例模型）**：示例模型的改写只是把上一问和追问拼接，所以这里的改写准确率实际只反映「该不该改写」判得对不对。"
            "「承接」类追问既没有指代词也没有省略句式，只能靠与上一问的向量相似度判断；"
            "示例向量模型下相似度都低于 CONTINUITY_SIMILARITY_THRESHOLD，所以这一类全部判成了话题切换。",
            "",
        ]
    if res["failures"]:
        lines += [
            "## 失败样例（最多 10 条）",
            "",
            "| 类别 | 上一问 | 追问 | 改写结果 | 原因 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for x in res["failures"][:10]:
            cells = [CATEGORY_NAMES[x["category"]], x["history"], x["query"], x["rewritten"], x["reason"]]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(run_with_services(evaluate, "embedder", "llm"))
