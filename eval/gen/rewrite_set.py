# 改写评测集：承接 / 话题切换 / 指代 / 省略 四类，全部来自测试集文档
import random

from eval.gen.candidates import REWRITE_CATEGORIES, terms_in
from eval.gen.corpus import Doc

PER_CATEGORY = 50
MAX_ATTEMPTS = 2000
FIRST = ["{p}的{t}是怎么规定的？", "{p}关于{t}有什么规定？", "请问{p}的{t}是什么？"]
FOLLOW = {
    "ellipsis": ["那{t}呢？", "{t}呢？", "那{t}怎么算？"],
    "pronoun": ["它的{t}是怎么规定的？", "这款产品的{t}有什么规定？", "该产品的{t}是什么？"],
    "continue": ["{t}有哪些例外情况？", "{t}具体怎么执行？", "关于{t}还有别的限制吗？"],
    "switch": ["{p}的{t}是怎么规定的？", "{p}关于{t}有哪些规定？"],
}


def _terms(doc: Doc, vocab: list[str]) -> list[str]:
    text = "".join(c.text for c in doc.chunks)
    return [t for t in vocab if t in text]


def _standalone(category: str, short: str, history: str, t1: str, t2: str, query: str) -> str:
    """期望的独立问句只用于人工查看；自动评分看 must_include。"""
    if category == "ellipsis":
        return history.replace(t1, t2)
    if category == "pronoun":
        for p in ("它的", "这款产品的", "该产品的"):
            query = query.replace(p, f"{short}的")
        return query
    return f"{short}的{query}"


def build_rewrite_set(docs: list[Doc], rng: random.Random) -> list[dict]:
    vocab = terms_in(REWRITE_CATEGORIES)
    pool = [(d, ts) for d in docs if len(ts := _terms(d, vocab)) >= 3]
    if not pool:
        return []
    rows, seen = [], set()
    for category, templates in FOLLOW.items():
        made = 0
        for _ in range(MAX_ATTEMPTS):
            if made >= PER_CATEGORY:
                break
            doc, ts = rng.choice(pool)
            t1, t2 = rng.sample(ts, 2)
            history = rng.choice(FIRST).format(p=doc.short, t=t1)
            if category == "switch":
                # 切到另一份产品；只有一份测试文档时换成同一产品的另一个话题
                others = [(d, o) for d, o in pool if d.doc_id != doc.doc_id] or pool
                other, ots = rng.choice(others)
                t2 = rng.choice([t for t in ots if t != t1])
                query = rng.choice(templates).format(p=other.short, t=t2)
                label, expected, must = "switch", query, []
            else:
                term = t1 if category == "continue" else t2
                query = rng.choice(templates).format(t=term)
                label, must = "continue", [doc.short, term]
                expected = _standalone(category, doc.short, history, t1, t2, query)
            key = (category, history, query)
            if key in seen:
                continue
            seen.add(key)
            made += 1
            rows.append(
                {
                    "category": category,
                    "doc_id": doc.doc_id,
                    "history": [{"question": history}],
                    "query": query,
                    "label": label,
                    "expected_standalone": expected,
                    "must_include": must,
                    "split": "test",
                }
            )
    for i, r in enumerate(rows, start=1):
        r["qid"] = f"rw-{i:04d}"
    return [{"qid": r.pop("qid"), **r} for r in rows]
