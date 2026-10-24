// 把已保存的回合和正在进行的这一问统一成同一种视图
import type { LivePhase, LiveTurn } from './live-store.ts'
import type { Citation, RefusalReason, Topic, Turn } from './schemas.ts'

export interface TurnView {
  id: string
  question: string
  askedAt: string
  rewritten: string | null
  topic: Topic | string
  phase: LivePhase
  text: string
  citations: Citation[]
  retrieved: boolean
  used: number[]
  cacheHit: boolean
  refusal: { reason: RefusalReason; topScore: number } | null
  error: { code: string; detail: string; status?: number } | null
  totalMs?: number
  live: boolean
}

const PHASES: Record<string, LivePhase> = { answered: 'answered', refused: 'refused', aborted: 'aborted' }
const REASONS = new Set(['no_hit', 'low_score', 'unsupported'])

// 已保存的回合按对象缓存视图：Query 数据没变时返回同一个对象，TurnBlock 的 memo 才能生效
const persisted = new WeakMap<Turn, TurnView>()

export function fromPersisted(t: Turn): TurnView {
  let view = persisted.get(t)
  if (!view) {
    view = buildPersisted(t)
    persisted.set(t, view)
  }
  return view
}

function buildPersisted(t: Turn): TurnView {
  const phase = PHASES[t.status] ?? 'error'
  const top = Math.max(0, ...t.citations.map((c) => c.rerank_score))
  return {
    id: t.turn_id,
    question: t.question,
    askedAt: t.created_at,
    rewritten: t.standalone_query !== t.question ? t.standalone_query : null,
    topic: t.topic,
    phase,
    text: t.answer,
    citations: t.citations,
    retrieved: t.citations.length > 0,
    used: t.used_citations,
    cacheHit: t.cache_hit,
    refusal:
      phase === 'refused' && t.refusal_reason && REASONS.has(t.refusal_reason)
        ? { reason: t.refusal_reason as RefusalReason, topScore: top }
        : null,
    error: phase === 'error' ? { code: 'internal', detail: '' } : null,
    totalMs: t.timings.total,
    live: false,
  }
}

export function fromLive(l: LiveTurn): TurnView {
  return {
    id: l.messageId ?? `live-${l.key}`,
    question: l.question,
    askedAt: l.askedAt,
    rewritten: l.rewritten && l.rewritten !== l.question ? l.rewritten : null,
    topic: l.topic,
    phase: l.phase,
    text: l.text,
    citations: l.citations,
    retrieved: l.retrieved,
    used: l.used,
    cacheHit: l.cacheHit,
    refusal: l.refusal,
    error: l.error,
    totalMs: l.timings.total,
    live: true,
  }
}

/**
 * 生成中以实时版本为准（后端可能已经存好、而前端还没读完流）；
 * 结束后已保存的回合优先，出错或中止的保留实时版本，因为它带着更完整的错误信息和已生成的文字。
 */
export function mergeTurns(saved: Turn[], live: LiveTurn | null): TurnView[] {
  const views = saved.map(fromPersisted)
  if (!live) return views
  const lv = fromLive(live)
  const i = views.findIndex((v) => v.id === lv.id)
  if (i < 0) return [...views, lv]
  if (lv.phase !== 'answered' && lv.phase !== 'refused') views[i] = lv
  return views
}
