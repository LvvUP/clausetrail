from clausetrail.models.llm import LLM

REWRITE_SYSTEM = """你负责把多轮对话中的追问改写成独立问句。
要求：
1. 补全追问中省略的产品名、险种、条款名或主语，让问句脱离上下文也能看懂。
2. 只补全对话里已经出现过的信息，不要添加新的条件、数字或推测。
3. 已经完整的问句原样返回。
4. 只输出改写后的一句话，不要解释，不要加引号。"""

# 历史回答只取开头：改写只需要知道上一轮问的是哪个产品、哪个术语，整段回答会占满轻量模型的上下文，
# 还容易让模型把回答里的内容抄进问句
ANSWER_PREVIEW_CHARS = 200
MAX_QUERY_CHARS = 200


def build_rewrite_messages(question: str, window: list[dict], summary: str) -> list[dict[str, str]]:
    lines = [f"更早的对话摘要：{summary}"] if summary else []
    for t in window:
        # 历史问题用当时改写好的独立问句：原话里的「它」「那个」隔几轮就找不到指代对象了
        lines.append(f"用户：{t.get('standalone_query') or t['question']}")
        if t.get("answer"):
            lines.append(f"助手：{t['answer'][:ANSWER_PREVIEW_CHARS]}")
    lines.append(f"当前问题：{question}")
    return [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


# 轻量模型偶尔会附带解释、前缀或引号，只取第一行并去掉这些；输出为空时退回原问题，改写失败也不会把问题弄丢
def clean(output: str, fallback: str) -> str:
    text = output.strip()
    if not text:
        return fallback
    first = text.splitlines()[0].strip()
    for prefix in ("改写后：", "独立问句：", "问题："):
        if first.startswith(prefix):
            first = first[len(prefix) :].strip()
    first = first.strip("「」“”\"'")
    return first[:MAX_QUERY_CHARS] or fallback


async def rewrite(llm: LLM, model: str, question: str, window: list[dict], summary: str) -> str:
    out = await llm.complete(build_rewrite_messages(question, window, summary), model=model, temperature=0.0)
    return clean(out, question)
