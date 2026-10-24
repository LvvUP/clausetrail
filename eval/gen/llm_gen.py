# 可选：用百炼 Qwen 按片段出题，再让模型判断「只凭这句原文能否回答」
import json
import re

from clausetrail.models.llm import LLM
from eval.metrics import norm

GEN_SYSTEM = """你在为保险条款问答系统构造评测题。根据给出的条款片段，写 {n} 个真实用户会问的问题。
要求：问题里带上产品名「{product}」；每个问题附上片段中能回答它的一句原文，必须逐字摘抄。
只输出 JSON 数组：[{{"question": "...", "span": "..."}}]"""
JUDGE_SYSTEM = "只根据给出的这句条款原文，能否回答这个问题？只回答「能」或「不能」。"
JSON_ARRAY = re.compile(r"\[.*\]", re.S)


def parse_generated(output: str, chunk_text: str) -> list[dict]:
    """只保留 span 能在片段里逐字找到的题目。"""
    m = JSON_ARRAY.search(output)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    body = norm(chunk_text)
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        question, span = str(it.get("question", "")).strip(), str(it.get("span", "")).strip()
        if question and span and norm(span) in body:
            out.append({"question": question, "span": span})
    return out


async def generate(llm: LLM, model: str, product: str, chunk_text: str, n: int = 2) -> list[dict]:
    messages = [
        {"role": "system", "content": GEN_SYSTEM.format(n=n, product=product)},
        {"role": "user", "content": chunk_text},
    ]
    kept = []
    for it in parse_generated(await llm.complete(messages, model=model, temperature=0.7), chunk_text):
        verdict = await llm.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"问题：{it['question']}\n原文：{it['span']}"},
            ],
            model=model,
        )
        if verdict.strip().startswith("能"):
            kept.append(it)
    return kept
