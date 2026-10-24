import contextlib
import json

from fastapi import APIRouter, Request
from sse_starlette import EventSourceResponse

from clausetrail.api.schemas import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
    chat_service = request.app.state.chat

    # 客户端断开就停止迭代。aclosing 会关闭 ChatService.stream，由它把这一轮记为 aborted 并落库
    # （meta 之前就断开的不落库，见 ChatService._finish_early），同时关掉还在输出的 LLM 流
    async def events():
        async with contextlib.aclosing(chat_service.stream(body.message, body.session_id)) as stream:
            async for name, data in stream:
                if await request.is_disconnected():
                    break
                yield {"event": name, "data": json.dumps(data, ensure_ascii=False)}

    # 每 15 秒发一行 SSE 注释保活，检索或精排慢的时候连接不会被代理当成空闲断开
    return EventSourceResponse(events(), ping=15)
