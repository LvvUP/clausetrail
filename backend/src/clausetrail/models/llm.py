import re
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Protocol

import httpx
from openai import AsyncOpenAI

from clausetrail.config import Settings, get_settings

Message = dict[str, str]
LLM_TIMEOUT_SECONDS = 60.0


class LLM(Protocol):
    name: str

    async def complete(self, messages: list[Message], model: str, temperature: float = 0.0) -> str: ...

    def stream(self, messages: list[Message], model: str, temperature: float = 0.2) -> AsyncIterator[str]: ...


class BailianLLM:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        if not settings.aliyun_bailian_api_key:
            raise ValueError("ALIYUN_BAILIAN_API_KEY 未配置")
        self.name = "aliyun_bailian"
        # SDK 默认超时约 10 分钟；生成卡住时 SSE 连接会长时间没有任何事件
        self._client = AsyncOpenAI(
            api_key=settings.aliyun_bailian_api_key,
            base_url=settings.aliyun_bailian_base_url,
            http_client=http_client,
            timeout=httpx.Timeout(LLM_TIMEOUT_SECONDS, connect=5.0),
            max_retries=2,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(self, messages: list[Message], model: str, temperature: float = 0.0) -> str:
        resp = await self._client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return resp.choices[0].message.content or ""

    async def stream(self, messages: list[Message], model: str, temperature: float = 0.2) -> AsyncIterator[str]:
        resp = await self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, stream=True
        )
        # 调用方提前关掉生成器（客户端断开、判定拒答）时，async with 负责关掉底层 HTTP 响应，不等垃圾回收
        async with resp:
            async for event in resp:
                # 跳过没有 choices 的事件（如用量统计）和只带 role、没有正文的 delta
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content


EVIDENCE_BLOCK = re.compile(r"^\[(\d+)\][^\n]*\n(.+?)(?=\n\[\d+\]|\Z)", re.DOTALL | re.MULTILINE)
SENTENCE = re.compile(r"[^。；！？]+[。；！？]")


class FakeLLM:
    name = "fake:extractive"

    # 靠系统提示里的关键词区分调用方：rewriter 的提示含「独立问句」，memory 的摘要提示含「摘要」；
    # 改这两处提示词时要同步改这里
    async def complete(self, messages: list[Message], model: str, temperature: float = 0.0) -> str:
        user = messages[-1]["content"]
        if "独立问句" in messages[0]["content"]:
            history, _, current = user.rpartition("当前问题：")
            current = current.strip().splitlines()[0]
            previous = [ln[3:] for ln in history.splitlines() if ln.startswith("用户：")]
            return f"{previous[-1]}，{current}" if previous else current
        if "摘要" in messages[0]["content"]:
            return user[-300:]
        return "".join([chunk async for chunk in self.stream(messages, model)])

    # 每条证据只取第一句并带上编号，最多两条，没有密钥时引用校验和拒答流程也能走通
    async def stream(self, messages: list[Message], model: str, temperature: float = 0.2) -> AsyncIterator[str]:
        user = messages[-1]["content"]
        parts = []
        for m in EVIDENCE_BLOCK.finditer(user):
            sentences = SENTENCE.findall(m.group(2))
            if sentences:
                parts.append(f"{sentences[0].strip()}[{m.group(1)}]")
            if len(parts) == 2:
                break
        text = "根据条款：" + "".join(parts) if parts else "现有条款中没有找到可以回答该问题的内容。"
        for i in range(0, len(text), 8):
            yield text[i : i + 8]


def build_llm(settings: Settings) -> LLM:
    match settings.llm_backend:
        case "aliyun_bailian":
            return BailianLLM(settings)
        case "fake":
            return FakeLLM()
    raise ValueError(f"未知的 llm_backend: {settings.llm_backend}")


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    return build_llm(get_settings())
