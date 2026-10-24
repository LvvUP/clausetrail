import type { z } from 'zod'
import { ApiError, readDetail } from './api.ts'
import { deltaEvent, doneEvent, errorEvent, metaEvent, refusalEvent, retrievalEvent } from './schemas.ts'
import { type RawEvent, SseParser } from './sse-parser.ts'

const SCHEMAS = {
  meta: metaEvent,
  retrieval: retrievalEvent,
  delta: deltaEvent,
  refusal: refusalEvent,
  error: errorEvent,
  done: doneEvent,
}

export type ChatEvent = {
  [K in keyof typeof SCHEMAS]: { event: K; data: z.infer<(typeof SCHEMAS)[K]> }
}[keyof typeof SCHEMAS]

export function parseChatEvent(raw: RawEvent): ChatEvent | null {
  if (!Object.hasOwn(SCHEMAS, raw.event)) return null
  const name = raw.event as keyof typeof SCHEMAS
  let json: unknown
  try {
    json = JSON.parse(raw.data)
  } catch {
    throw new ApiError('schema', 200, `事件 ${name} 不是合法 JSON`)
  }
  const parsed = SCHEMAS[name].safeParse(json)
  if (!parsed.success) throw new ApiError('schema', 200, `事件 ${name}：${parsed.error.message}`)
  return { event: name, data: parsed.data } as ChatEvent
}

/** POST /api/chat，逐个产出校验过的 SSE 事件；signal 中止时抛出 AbortError。 */
export async function* streamChat(message: string, sessionId: string | null, signal: AbortSignal) {
  let res: Response
  try {
    res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(sessionId ? { message, session_id: sessionId } : { message }),
      signal,
    })
  } catch (err) {
    if (signal.aborted) throw err
    throw new ApiError('network', 0, err instanceof Error ? err.message : String(err))
  }
  if (!res.ok || !res.body) throw new ApiError('http', res.status, await readDetail(res))

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  const parser = new SseParser()
  try {
    while (true) {
      const { value, done } = await reader.read()
      const raws = done ? parser.flush() : parser.feed(value)
      for (const raw of raws) {
        const event = parseChatEvent(raw)
        if (event) yield event
      }
      if (done) return
    }
  } finally {
    await reader.cancel().catch(() => undefined)
  }
}
