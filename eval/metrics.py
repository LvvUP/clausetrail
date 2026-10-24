# 评测指标（纯函数，不依赖模型和索引）
import re
from itertools import pairwise

WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return WS.sub("", text)


def is_hit(doc_id: str, text: str, gold: list[dict]) -> bool:
    """召回片段与 gold 同一文档，且片段文本（去空白后）包含 gold 句子。"""
    body = norm(text)
    return any(g["doc_id"] == doc_id and norm(g["span"]) in body for g in gold)


def reciprocal_rank(hits: list[bool], k: int = 10) -> float:
    for i, hit in enumerate(hits[:k], start=1):
        if hit:
            return 1.0 / i
    return 0.0


def recall_at(hits: list[bool], k: int) -> float:
    return float(any(hits[:k]))


def ranking_metrics(runs: list[list[bool]]) -> dict:
    n = len(runs)
    if not n:
        return {"n": 0, "mrr@10": 0.0, "recall@5": 0.0, "recall@10": 0.0}
    return {
        "n": n,
        "mrr@10": round(sum(reciprocal_rank(h) for h in runs) / n, 4),
        "recall@5": round(sum(recall_at(h, 5) for h in runs) / n, 4),
        "recall@10": round(sum(recall_at(h, 10) for h in runs) / n, 4),
    }


def truncation_rate(units: list[str], chunks: list[str]) -> dict:
    """一个语义单元（句子、列表、章节）没有完整落在任何一个片段里，就算被截断。"""
    bodies = [norm(c) for c in chunks]
    keys = [k for k in (norm(u) for u in units) if k]
    truncated = sum(1 for k in keys if not any(k in b for b in bodies))
    return {"units": len(keys), "truncated": truncated, "rate": round(truncated / len(keys), 4) if keys else 0.0}


def best_threshold(answerable: list[float], unanswerable: list[float]) -> dict | None:
    """选使「该答的答、该拒的拒」平衡准确率最高的阈值；并列时取更小的阈值，少误拒。"""
    if not answerable or not unanswerable:
        return None
    scores = sorted(set(answerable) | set(unanswerable))
    candidates = [scores[0] - 1e-6, *((a + b) / 2 for a, b in pairwise(scores)), scores[-1] + 1e-6]
    best = None
    for t in candidates:
        answered = sum(s >= t for s in answerable) / len(answerable)
        refused = sum(s < t for s in unanswerable) / len(unanswerable)
        balanced = (answered + refused) / 2
        if best is None or balanced > best["balanced_accuracy"] + 1e-12:
            best = {
                "threshold": round(t, 4),
                "balanced_accuracy": round(balanced, 4),
                "answerable_answered": round(answered, 4),
                "unanswerable_refused": round(refused, 4),
            }
    return best


def shingles(text: str, n: int = 3) -> set[str]:
    t = norm(text)
    return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0
