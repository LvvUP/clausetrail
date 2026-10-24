from clausetrail.models.llm import LLM
from clausetrail.storage.db import Database

SUMMARY_SYSTEM = """请把下面的保险咨询对话压缩成一段不超过 {n} 字的摘要。
保留用户关心的产品名、险种、条款名称和已经确认的事实（如等待期天数、免赔额），不要评价，不要添加对话中没有的内容。"""

ANSWER_PREVIEW_CHARS = 300


USABLE_STATUS = ("answered", "refused")


# 改写窗口和摘要用同一口径：只算答完或拒答的轮次，出错 / 中止的轮次两边都不计
def window(turns: list[dict], n: int) -> list[dict]:
    usable = [t for t in turns if t.get("status", "answered") in USABLE_STATUS]
    return usable[-n:] if n > 0 else []


async def maybe_summarize(
    db: Database, llm: LLM, model: str, session_id: str, window_turns: int, max_chars: int
) -> str | None:
    session = db.get_session(session_id)
    if session is None:
        return None
    turns = [t for t in db.turns(session_id) if t["status"] in USABLE_STATUS]
    older = turns[:-window_turns] if len(turns) > window_turns else []
    done = session["summarized_turns"]
    if len(older) <= done:
        return None
    lines = [f"已有摘要：{session['summary']}"] if session["summary"] else []
    for t in older[done:]:
        lines.append(f"用户：{t['standalone_query']}")
        lines.append(f"助手：{t['answer'][:ANSWER_PREVIEW_CHARS]}")
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM.format(n=max_chars)},
        {"role": "user", "content": "\n".join(lines)},
    ]
    summary = (await llm.complete(messages, model=model)).strip()[:max_chars]
    db.set_summary(session_id, summary, len(older))
    return summary
