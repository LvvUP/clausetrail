// 后端接口与 SSE 事件的结构校验；字段名与 backend/src/clausetrail/api、chat.py 一致
import { z } from 'zod'

const timings = z.record(z.string(), z.number())

export const citationSchema = z.object({
  n: z.number().int().positive(),
  chunk_id: z.string(),
  doc_id: z.string(),
  file: z.string(),
  section_path: z.array(z.string()),
  page_start: z.number().int(),
  page_end: z.number().int(),
  snippet: z.string(),
  parser: z.string(),
  rerank_score: z.number(),
  rrf_score: z.number(),
  dense_rank: z.number().int().nullable(),
  bm25_rank: z.number().int().nullable(),
})
export type Citation = z.infer<typeof citationSchema>

export const refusalReasonSchema = z.enum(['no_hit', 'low_score', 'unsupported'])
export type RefusalReason = z.infer<typeof refusalReasonSchema>
export const topicSchema = z.enum(['new', 'continue', 'switch'])
export type Topic = z.infer<typeof topicSchema>
export const turnStatusSchema = z.enum(['answered', 'refused', 'error', 'aborted'])

export const metaEvent = z.object({
  session_id: z.string(),
  message_id: z.string(),
  question: z.string(),
  rewritten_query: z.string().nullable(),
  topic: topicSchema,
  topic_reason: z.string(),
  similarity: z.number().nullable(),
  cache_hit: z.boolean(),
  kb_version: z.number().int(),
})
export const retrievalEvent = z.object({
  citations: z.array(citationSchema),
  cached: z.boolean(),
  candidates: z.number().int(),
  dense_hits: z.number().int().nullable(),
  bm25_hits: z.number().int().nullable(),
  bm25_backend: z.string(),
  top_score: z.number(),
  timings_ms: timings,
})
export const deltaEvent = z.object({ text: z.string() })
export const refusalEvent = z.object({ reason: refusalReasonSchema, message: z.string(), top_score: z.number() })
export const errorEvent = z.object({ code: z.string(), message: z.string() })
export const doneEvent = z.object({
  session_id: z.string(),
  message_id: z.string(),
  status: turnStatusSchema,
  refusal_reason: refusalReasonSchema.nullable(),
  used_citations: z.array(z.number().int()),
  cache_hit: z.boolean(),
  timings_ms: timings,
})

export const sessionSchema = z.object({
  session_id: z.string(),
  title: z.string(),
  summary: z.string(),
  summarized_turns: z.number().int(),
  created_at: z.string(),
  updated_at: z.string(),
})
export type Session = z.infer<typeof sessionSchema>
export const sessionListSchema = z.object({ sessions: z.array(sessionSchema) })

export const turnSchema = z.object({
  turn_id: z.string(),
  session_id: z.string(),
  question: z.string(),
  standalone_query: z.string(),
  topic: z.string(),
  answer: z.string(),
  status: z.string(),
  refusal_reason: z.string().nullable(),
  citations: z.array(citationSchema),
  used_citations: z.array(z.number().int()),
  cache_hit: z.boolean(),
  timings: timings,
  created_at: z.string(),
})
export type Turn = z.infer<typeof turnSchema>
export const sessionDetailSchema = z.object({ session: sessionSchema, turns: z.array(turnSchema) })
export type SessionDetail = z.infer<typeof sessionDetailSchema>

export const documentSchema = z.object({
  doc_id: z.string(),
  file_name: z.string(),
  pages: z.number().int(),
  chunks: z.number().int(),
  page_types: z.record(z.string(), z.number()),
  parsers: z.record(z.string(), z.number()),
  status: z.string(),
  kb_version: z.number().int(),
  ingested_at: z.string(),
})
export type ClauseDocument = z.infer<typeof documentSchema>
export const documentListSchema = z.object({ kb_version: z.number().int(), documents: z.array(documentSchema) })

const componentSchema = z.object({
  status: z.enum(['ok', 'degraded', 'unavailable']),
  backend: z.string(),
  detail: z.string().optional(),
  index: z.string().optional(),
  dim: z.number().int().optional(),
})
export type HealthComponent = z.infer<typeof componentSchema>
export const healthSchema = z.object({
  status: z.enum(['ok', 'unavailable']),
  version: z.string(),
  kb_version: z.number().int(),
  documents: z.number().int(),
  components: z.record(z.string(), componentSchema),
})
export type Health = z.infer<typeof healthSchema>
