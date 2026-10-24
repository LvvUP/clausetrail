"""切分评测：固定长度基线 vs 结构感知切分的截断率。python -m eval.run_chunking

一个语义单元（句子、「引导句 + 条目」列表、短章节）没有完整落在任何一个片段里，就算被截断。
"""

import sys

from clausetrail.chunking.merge_rules import blocks_to_paragraphs
from clausetrail.chunking.splitter import fixed_length_chunks, is_toc_like
from clausetrail.chunking.text_utils import count_tokens, is_list_item, is_list_lead_in, split_sentences
from clausetrail.config import get_settings
from clausetrail.structure.tree import walk
from clausetrail.types import SectionNode
from eval.common import REPORTS, header, write_json, write_text
from eval.gen.corpus import load_corpus
from eval.metrics import truncation_rate

KINDS = {
    "sentence": "句子",
    "list": "列表（引导句 + 条目）",
    "exclusion_list": "其中：责任免除列表",
    "section": "短章节",
}
MIN_SENTENCE_CHARS = 8
MIN_LIST_ITEMS = 2
EXCLUSION_KEYS = ("责任免除", "免责")
FIXED_SIZE, FIXED_OVERLAP = 480, 50


def units_for(root: SectionNode, max_tokens: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {k: [] for k in KINDS}
    for node, path in walk(root):
        paras = [p for p in blocks_to_paragraphs(node.blocks) if not is_toc_like(p.text)]
        for p in paras:
            pieces = p.text.splitlines() if p.is_table else split_sentences(p.text)
            out["sentence"] += [s for s in pieces if len(s) >= MIN_SENTENCE_CHARS]
        i = 0
        while i < len(paras):
            if not is_list_lead_in(paras[i].text):
                i += 1
                continue
            j = i + 1
            while j < len(paras) and is_list_item(paras[j].text):
                j += 1
            text = "\n".join(p.text for p in paras[i:j])
            # 超过片段上限的列表本来就放不进一个片段，不计入
            if j - i - 1 >= MIN_LIST_ITEMS and count_tokens(text) <= max_tokens:
                out["list"].append(text)
                if any(k in seg for seg in path for k in EXCLUSION_KEYS):
                    out["exclusion_list"].append(text)
            i = j
        if paras and not node.children:
            text = "\n".join(p.text for p in paras)
            if count_tokens(text) <= max_tokens:
                out["section"].append(text)
    return out


def evaluate() -> dict:
    settings = get_settings()
    docs = load_corpus()
    agg = {m: {k: {"units": 0, "truncated": 0} for k in KINDS} for m in ("fixed", "structure")}
    per_doc = []
    for d in docs:
        units = units_for(d.root, settings.chunk_max_tokens)
        chunks = {
            "fixed": [c.text for c in fixed_length_chunks(d.root, d.doc_id, d.file_name, FIXED_SIZE, FIXED_OVERLAP)],
            "structure": [c.text for c in d.chunks],
        }
        row = {"file": d.file_name, "chunks": {m: len(v) for m, v in chunks.items()}}
        for m, texts in chunks.items():
            for k in KINDS:
                t = truncation_rate(units[k], texts)
                row[f"{m}.{k}"] = t
                agg[m][k]["units"] += t["units"]
                agg[m][k]["truncated"] += t["truncated"]
        per_doc.append(row)
    for m in agg.values():
        for a in m.values():
            a["rate"] = round(a["truncated"] / a["units"], 4) if a["units"] else 0.0
    result = {
        "documents": len(docs),
        "max_tokens": settings.chunk_max_tokens,
        "fixed": {"size_chars": FIXED_SIZE, "overlap_chars": FIXED_OVERLAP},
        "layout_analysis": settings.layout_analysis,
        "overall": agg,
        "per_doc": per_doc,
    }
    write_json(REPORTS / "chunking.json", result)
    write_text(REPORTS / "chunking.md", render(result))
    return result


def render(res: dict) -> str:
    fixed, structure = res["overall"]["fixed"], res["overall"]["structure"]
    lines = [
        header("切分评测", res["documents"], None, {"layout_analysis": str(res["layout_analysis"]).lower()}),
        f"- 固定长度基线：每 {res['fixed']['size_chars']} 字一段，重叠 {res['fixed']['overlap_chars']} 字",
        f"- 结构感知：按章节树递归切分，上限 {res['max_tokens']} token，列表引导句与条目、但书与前句合并，句子边界重叠",
        "- 入库时还会按向量相似度合并相邻段落；这里关闭了这一步，没有向量模型也能复现",
        "",
        "| 语义单元 | 数量 | 固定长度截断率 | 结构感知截断率 |",
        "| --- | --- | --- | --- |",
    ]
    for k, name in KINDS.items():
        lines.append(f"| {name} | {fixed[k]['units']} | {fixed[k]['rate']:.2%} | {structure[k]['rate']:.2%} |")
    lines += ["", "| 文档 | 固定长度片段数 | 结构感知片段数 |", "| --- | --- | --- |"]
    for row in res["per_doc"]:
        lines.append(f"| {row['file']} | {row['chunks']['fixed']} | {row['chunks']['structure']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(evaluate()["overall"])
    sys.exit(0)
