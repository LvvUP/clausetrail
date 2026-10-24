from clausetrail.types import ScoredChunk

SYSTEM_PROMPT = """你是一名熟悉保险条款的助手，只根据下面提供的条款证据回答用户的问题。

规则：
1. 只能使用证据中的内容，不要引入证据以外的常识或推测。
2. 每一句陈述的末尾都要用方括号标注它依据的证据编号，例如「等待期为 90 日[1]」；一句话依据多条证据时写成 [1][3]。
3. 证据不足以回答时，直接说明「现有条款中没有找到可以回答该问题的内容」，不要编造。
4. 涉及金额、天数、比例、年龄等数字时，照抄证据原文，不要换算。
5. 用简洁的中文分点或分段作答，不要复述问题，不要写开场白和结束语。"""


# 编号从 1 开始，和 chat.py 里 citation_payload 的编号、前端的 [n] 一一对应；任何一端从 0 开始，引用都会错位一位。
# 每条证据带上文件、章节和页码，几份条款里同名的章节（如「等待期」）模型也能分清出自哪一份
def format_evidence(evidence: list[ScoredChunk]) -> str:
    blocks = []
    for i, sc in enumerate(evidence, start=1):
        c = sc.chunk
        pages = f"第{c.page_start}页" if c.page_start == c.page_end else f"第{c.page_start}-{c.page_end}页"
        head = " · ".join(x for x in (c.file_name, c.section_label, pages) if x)
        blocks.append(f"[{i}] {head}\n{c.text.strip()}")
    return "\n\n".join(blocks)


def build_messages(question: str, evidence: list[ScoredChunk]) -> list[dict[str, str]]:
    user = f"问题：{question}\n\n条款证据：\n{format_evidence(evidence)}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
