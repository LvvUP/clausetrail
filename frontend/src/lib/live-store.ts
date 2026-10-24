// 正在进行的这一问放在组件之外：首问创建会话后路由从 / 跳到 /c/$id，组件会重新挂载，流不能断
import { useSyncExternalStore } from 'react'
import { ApiError } from './api.ts'
import { streamChat } from './chat-stream.ts'
import { usedCitations } from './citations.ts'
import type { Citation, RefusalReason, Topic } from './schemas.ts'

export type LivePhase = 'retrieving' | 'streaming' | 'answered' | 'refused' | 'error' | 'aborted'

export interface LiveTurn {
  key: number
  origin: string | null
  sessionId: string | null
  messageId: string | null
  question: string
  askedAt: string
  phase: LivePhase
  rewritten: string | null
  topic: Topic
  cacheHit: boolean
  citations: Citation[]
  retrieved: boolean
  text: string
  used: number[]
  refusal: { reason: RefusalReason; topScore: number } | null
  error: { code: string; detail: string; status?: number } | null
  timings: Record<string, number>
  /** 事件流是否还开着：refusal / error 之后、done 之前流还没结束，这时不能发下一问 */
  open: boolean
}

let live: LiveTurn | null = null
let controller: AbortController | null = null
let seq = 0
const listeners = new Set<() => void>()

function emit() {
  listeners.forEach((l) => l())
}

function patch(key: number, p: Partial<LiveTurn>) {
  if (!live || live.key !== key) return
  live = { ...live, ...p }
  emit()
}

function toError(err: unknown): { code: string; detail: string; status?: number } {
  if (err instanceof ApiError) return { code: err.kind, detail: err.detail, status: err.status }
  return { code: 'network', detail: err instanceof Error ? err.message : String(err) }
}

export const chatStore = {
  get: () => live,
  inProgress: () => live?.phase === 'retrieving' || live?.phase === 'streaming',
  /** 这一问属于哪个会话：拿到 session_id 前算在发起时的页面上。 */
  owner: () => (live ? (live.sessionId ?? live.origin) : null),
  subscribe(l: () => void) {
    listeners.add(l)
    return () => {
      listeners.delete(l)
    }
  },
  stop() {
    controller?.abort()
  },
  /** 离开当前会话：中止生成并清掉临时状态，返回是否真的中止了一次生成。 */
  leave(): boolean {
    const stopped = chatStore.inProgress()
    controller?.abort()
    controller = null
    live = null
    emit()
    return stopped
  },
  /** 返回值 started=false 表示还没建立会话就失败了，调用方应把问题放回输入框。 */
  async ask(question: string, origin: string | null, onSession: (id: string) => void) {
    controller?.abort()
    const ctl = new AbortController()
    controller = ctl
    const key = ++seq
    live = {
      key,
      origin,
      sessionId: origin,
      messageId: null,
      question,
      askedAt: new Date().toISOString(),
      phase: 'retrieving',
      rewritten: null,
      topic: 'new',
      cacheHit: false,
      citations: [],
      retrieved: false,
      text: '',
      used: [],
      refusal: null,
      error: null,
      timings: {},
      open: true,
    }
    emit()
    let started = false
    // 这一问实际属于哪个会话，返回给调用方：live 可能已经被 leave() 清掉或换成了下一问，不能事后再读
    let sid = origin
    try {
      for await (const ev of streamChat(question, origin, ctl.signal)) {
        if (live?.key !== key) return { started: true, sessionId: sid }
        switch (ev.event) {
          case 'meta':
            started = true
            sid = ev.data.session_id
            patch(key, {
              sessionId: ev.data.session_id,
              messageId: ev.data.message_id,
              rewritten: ev.data.rewritten_query,
              topic: ev.data.topic,
              cacheHit: ev.data.cache_hit,
            })
            if (ev.data.session_id !== origin) onSession(ev.data.session_id)
            break
          case 'retrieval':
            patch(key, { citations: ev.data.citations, retrieved: true, phase: 'streaming' })
            break
          case 'delta': {
            const text = live.text + ev.data.text
            patch(key, { text, used: usedCitations(text, live.citations.length) })
            break
          }
          case 'refusal':
            // 生成后才判定无出处时，后端会撤回已经流出的文字（库里也不存），这里同步清掉
            patch(key, {
              phase: 'refused',
              text: '',
              used: [],
              refusal: { reason: ev.data.reason, topScore: ev.data.top_score },
            })
            break
          case 'error':
            patch(key, { phase: 'error', error: { code: ev.data.code, detail: ev.data.message } })
            break
          case 'done':
            patch(key, { phase: ev.data.status, used: ev.data.used_citations, timings: ev.data.timings_ms })
            break
        }
      }
      if (chatStore.inProgress() && live?.key === key) {
        patch(key, { phase: 'error', error: { code: 'stream_cut', detail: '事件流在 done 之前结束' } })
      }
    } catch (err) {
      // 结果已经定下（refusal / error / done 之后流还没关）时不改：后端按已定的状态落库，这里改成中止或出错就和库里对不上
      if (live?.key === key && chatStore.inProgress()) {
        if (ctl.signal.aborted) patch(key, { phase: 'aborted' })
        else patch(key, { phase: 'error', error: toError(err) })
      }
    } finally {
      if (controller === ctl) controller = null
      patch(key, { open: false })
    }
    return { started, sessionId: sid }
  },
}

export function useLiveTurn(): LiveTurn | null {
  return useSyncExternalStore(chatStore.subscribe, chatStore.get)
}
