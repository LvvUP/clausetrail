"""合成评测集：python -m eval.gen.synthesize [--mode template|llm]

解析样例条款 → 按文档划分 dev / test → 挑 gold 句子 → 套模板（或用 Qwen 出题并自检）→ 泄漏检查 → 写 eval/datasets/。
"""

import argparse
import asyncio
import random
import sys
from datetime import date

from clausetrail.config import get_settings
from clausetrail.services import build_services
from eval.common import DATASETS, write_json, write_jsonl
from eval.gen import parsing_plan
from eval.gen.candidates import (
    EXCLUSION_KEYS,
    TERM_CATEGORIES,
    Candidate,
    aliases_by_term,
    detail_candidates,
    own_text,
    section_candidates,
    surface_map,
    term_candidates,
)
from eval.gen.check_leakage import check
from eval.gen.corpus import Doc, assign_splits, load_corpus
from eval.gen.llm_gen import generate
from eval.gen.rewrite_set import build_rewrite_set
from eval.metrics import norm

SEED = 20261019
TARGETS = {"test": {"general": 300, "term": 200}, "dev": {"general": 40, "term": 40}}
CITATION_SAMPLE = 200
TERM_TEMPLATES = ["{p}里的「{t}」是指什么？", "{p}所说的{t}是什么意思？", "{p}对{t}是怎么定义的？"]
SECTION_TEMPLATES = ["{p}的「{t}」是怎么规定的？", "{p}关于{t}有哪些规定？"]
DETAIL_TEMPLATES = [
    "{p}「{s}」里关于{k}是怎么约定的？",
    "{p}的{s}中，{k}有什么要求？",
    "{p}的「{s}」一节对{k}有什么规定？",
]
EXCLUSION_TEMPLATE = "{p}里，涉及{k}的情况能赔吗？"
# 开发集里的无关问题，用来标定拒答阈值；有几条是样例条款以外的险种
UNANSWERABLE = [
    "红烧肉怎么做才不腻？", "明天北京会下雨吗？", "Python 里怎么把列表倒序？", "汽车保养多久做一次？",
    "车险出险后怎么报案？", "宠物医疗险能报销疫苗费吗？", "航班延误险延误多久才赔？", "家财险能赔水管爆裂吗？",
    "信用卡逾期一天会影响征信吗？", "房贷提前还款要交违约金吗？", "租房押金房东不退怎么办？", "手机碎屏险值得买吗？",
    "马拉松赛前一周怎么训练？", "公司年会节目有什么推荐？", "护照过期了怎么补办？", "跑步膝盖疼该怎么缓解？",
    "今年的 GDP 增速是多少？", "如何给猫剪指甲？", "高考志愿怎么填报比较稳？", "雇主责任险的保费怎么算？",
]  # fmt: skip


def _item(
    cands: list[Candidate], doc: Doc, query: str, qtype: str, subtype: str, terms: list[str], source: str, verified: str
) -> dict:
    gold = [
        {
            "doc_id": c.doc_id,
            "file": doc.file_name,
            "section_path": " > ".join(c.chunk.section_path),
            "page": c.chunk.page_start,
            "span": c.span,
        }
        for c in cands
    ]
    return {
        "query": query,
        "type": qtype,
        "subtype": subtype,
        "terms": terms,
        "gold": gold,
        "source": source,
        "verified": verified,
    }


# 每道题对应一个独立的信息需求：同一句 gold 原文只出一道题，不靠换问法凑数


def term_items(cands: list[Candidate], docs: dict[str, Doc], rng: random.Random) -> list[dict]:
    """每个（文档, 标准术语）一道题。原文写的是别名时用标准术语问；原文写标准术语且有别名时，一半改用别名问，
    这样同义词的两个方向都能测到。"""
    alias_map, out = aliases_by_term(), []
    for c in cands:
        d = docs[c.doc_id]
        asked = c.key
        if c.surface == c.key and alias_map.get(c.key) and rng.random() < 0.5:
            asked = rng.choice(alias_map[c.key])
        q = rng.choice(TERM_TEMPLATES).format(p=d.short, t=asked)
        terms = list(dict.fromkeys([c.key, asked, c.surface or c.key]))
        out.append(_item([c], d, q, "term", "definition", terms, "synthetic-template", "auto"))
    return out


def section_items(groups: list[list[Candidate]], docs: dict[str, Doc], rng: random.Random) -> list[dict]:
    """每个章节一道题，章节首段的前两句都算 gold。"""
    out = []
    for group in groups:
        d = docs[group[0].doc_id]
        q = rng.choice(SECTION_TEMPLATES).format(p=d.short, t=group[0].key)
        out.append(_item(group, d, q, "general", "section", [], "synthetic-template", "auto"))
    return out


def detail_items(cands: list[Candidate], docs: dict[str, Doc], rng: random.Random) -> list[dict]:
    """章节里的具体规定；责任免除章节里的用「能不能赔」来问。"""
    out = []
    for c in cands:
        d = docs[c.doc_id]
        if any(k in seg for seg in c.chunk.section_path for k in EXCLUSION_KEYS):
            q, subtype = EXCLUSION_TEMPLATE.format(p=d.short, k=c.key), "exclusion"
        else:
            q, subtype = rng.choice(DETAIL_TEMPLATES).format(p=d.short, s=c.title, k=c.key), "detail"
        out.append(_item([c], d, q, "general", subtype, [c.key], "synthetic-template", "auto"))
    return out


async def llm_items(groups: list[list[Candidate]], docs: dict[str, Doc]) -> list[dict]:
    settings = get_settings()
    services = build_services(settings)
    try:
        services.require("llm")
        out = []
        for group in groups:
            c, d = group[0], docs[group[0].doc_id]
            seen: set[str] = set()
            for it in await generate(services.llm, settings.llm_answer_model, d.product, own_text(c.chunk)):
                if it["span"] in seen:
                    continue
                seen.add(it["span"])
                cand = Candidate(c.doc_id, c.chunk, it["span"], "section", c.key)
                out.append(_item([cand], d, it["question"], "general", "llm", [], "synthetic-llm", "llm-judge"))
        return out
    finally:
        # 模型客户端是异步的，事件循环结束前关掉
        await services.aclose()
        services.close()


def pick(items: list[dict], n: int, rng: random.Random) -> list[dict]:
    """随机取 n 道；问法完全相同的题（不同章节恰好同名）合并 gold，任一处命中都算对。"""
    pool = items[:]
    rng.shuffle(pool)
    out: dict[str, dict] = {}
    for item in pool:
        if item["query"] in out:
            out[item["query"]]["gold"].extend(item["gold"])
        elif len(out) < n:
            out[item["query"]] = item
    return list(out.values())


def mix(a: list[dict], b: list[dict], n: int, rng: random.Random) -> list[dict]:
    """两类题按候选数比例分配名额，一类不够时由另一类补。"""
    if len(a) + len(b) <= n:
        return pick(a, len(a), rng) + pick(b, len(b), rng)
    na = min(len(a), round(n * len(a) / (len(a) + len(b))))
    nb = min(len(b), n - na)
    return pick(a, min(len(a), n - nb), rng) + pick(b, nb, rng)


def build(mode: str) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    docs = list(load_corpus())
    by_id = {d.doc_id: d for d in docs}
    splits = assign_splits([d.doc_id for d in docs])
    surfaces = surface_map(TERM_CATEGORIES)
    # 同一家公司的条款共用大段标准文本；测试集的 gold 句子不能在开发集文档里出现
    dev_text = norm("".join(c.text for d in docs if splits[d.doc_id] == "dev" for c in d.chunks))
    out: dict[str, list[dict]] = {}
    for split in ("dev", "test"):
        members = [d for d in docs if splits[d.doc_id] == split]

        def fresh(c: Candidate, split: str = split) -> bool:
            return split == "dev" or norm(c.span) not in dev_text

        term_cands = [c for d in members for c in term_candidates(d, surfaces) if fresh(c)]
        # 三类候选共用一个「已用 gold」集合（按去空白后的句子），保证一句原文只出一道题
        used = {norm(c.span) for c in term_cands}
        groups = [kept for d in members for g in section_candidates(d, used) if (kept := [c for c in g if fresh(c)])]
        details = [c for d in members for c in detail_candidates(d, surfaces, used) if fresh(c)]
        target = TARGETS[split]
        term_rows = pick(term_items(term_cands, by_id, rng), target["term"], rng)
        # 术语题不够时由普通题补足总数，组成比例写进 manifest
        n_general = target["general"] + target["term"] - len(term_rows)
        if mode == "llm":
            general = pick(asyncio.run(llm_items(groups, by_id)), n_general, rng)
        else:
            general = mix(section_items(groups, by_id, rng), detail_items(details, by_id, rng), n_general, rng)
        rows = general + term_rows
        if split == "dev":
            rows += [
                {
                    "query": q,
                    "type": "unanswerable",
                    "subtype": "off_topic",
                    "terms": [],
                    "gold": [],
                    "source": "handwritten",
                    "verified": "n/a",
                }
                for q in UNANSWERABLE
            ]
        prefix = "dev" if split == "dev" else "rt"
        out[split] = [{"qid": f"{prefix}-{i:04d}", **r, "split": split} for i, r in enumerate(rows, start=1)]
    test_docs = [d for d in docs if splits[d.doc_id] == "test"]
    out["rewrite"] = build_rewrite_set(test_docs, rng)
    sample = rng.sample(out["test"], min(CITATION_SAMPLE, len(out["test"])))
    out["citation"] = [
        {"qid": f"cs-{i:04d}", "source_qid": r["qid"], "query": r["query"], "type": r["type"], "subtype": r["subtype"]}
        | {"gold": r["gold"], "split": "test"}
        for i, r in enumerate(sample, start=1)
    ]
    out["_splits"] = [{"file": d.file_name, "doc_id": d.doc_id, "split": splits[d.doc_id]} for d in docs]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="合成评测集")
    parser.add_argument("--mode", choices=["template", "llm"], default="template")
    args = parser.parse_args()
    data = build(args.mode)
    tests = {"retrieval_test": data["test"], "rewrite_test": data["rewrite"], "citation_sample": data["citation"]}
    problems = check(data["dev"], tests)
    if problems:
        print("\n".join(problems[:20]), f"\n泄漏检查未通过（{len(problems)} 个问题），没有写入文件")
        return 1
    files = {"dev": data["dev"], **tests}
    for name, rows in files.items():
        write_jsonl(DATASETS / f"{name}.jsonl", rows)
    counts = {name: len(rows) for name, rows in files.items()}
    by_type = {name: {} for name in ("dev", "retrieval_test")}
    by_subtype = {name: {} for name in by_type}
    for name in by_type:
        for r in files[name]:
            by_type[name][r["type"]] = by_type[name].get(r["type"], 0) + 1
            by_subtype[name][r["subtype"]] = by_subtype[name].get(r["subtype"], 0) + 1
    manifest = {
        "generated_on": date.today().isoformat(),
        "mode": args.mode,
        "seed": SEED,
        "counts": counts,
        "by_type": by_type,
        "by_subtype": by_subtype,
        "documents": data["_splits"],
        "parsing_sample_plan_rows": parsing_plan.write(),
        "note": "每道题对应一句不同的 gold 原文（同一句原文不重复出题），gold 都能在原文中逐字找到；"
        "问题由模板生成，未经人工校验（verified=auto）",
    }
    write_json(DATASETS / "manifest.json", manifest)
    print(counts, by_type, by_subtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
