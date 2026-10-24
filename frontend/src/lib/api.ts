import type { ZodType } from 'zod'
import { documentListSchema, healthSchema, sessionDetailSchema, sessionListSchema } from './schemas.ts'

export type ApiErrorKind = 'network' | 'http' | 'schema'

export class ApiError extends Error {
  kind: ApiErrorKind
  status: number
  detail: string

  constructor(kind: ApiErrorKind, status: number, detail: string) {
    super(detail || kind)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
    this.detail = detail
  }
}

// FastAPI 的错误是带 detail 的 JSON；TrustedHost 这类中间件返回纯文本（如 Invalid host header），也带出来，方便看出被拒的原因
export async function readDetail(res: Response): Promise<string> {
  let raw = ''
  try {
    raw = await res.text()
  } catch {
    return `HTTP ${res.status}`
  }
  try {
    const detail = (JSON.parse(raw) as { detail?: unknown } | null)?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((d) => (d as { msg?: string }).msg ?? '').join('；')
  } catch {
    // 不是 JSON：短的纯文本原样带出；代理返回的 HTML 错误页不带
    const text = raw.trim()
    if (text && text.length <= 200 && !text.startsWith('<')) return text
  }
  return `HTTP ${res.status}`
}

async function request<T>(path: string, schema: ZodType<T>, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`, init)
  } catch (err) {
    throw new ApiError('network', 0, err instanceof Error ? err.message : String(err))
  }
  if (!res.ok) throw new ApiError('http', res.status, await readDetail(res))
  let body: unknown
  try {
    body = await res.json()
  } catch {
    // 200 但不是 JSON（比如代理返回了 HTML），统一成 ApiError，界面和重试逻辑才认得
    throw new ApiError('schema', res.status, '响应不是 JSON')
  }
  const parsed = schema.safeParse(body)
  if (!parsed.success) throw new ApiError('schema', res.status, parsed.error.message)
  return parsed.data
}

export const api = {
  sessions: () => request('/sessions', sessionListSchema),
  // 路径参数来自 URL，先编码，避免 ../ 之类的字符改变请求路径
  session: (id: string) => request(`/sessions/${encodeURIComponent(id)}`, sessionDetailSchema),
  async deleteSession(id: string): Promise<void> {
    let res: Response
    try {
      res = await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' })
    } catch (err) {
      throw new ApiError('network', 0, err instanceof Error ? err.message : String(err))
    }
    if (!res.ok && res.status !== 404) throw new ApiError('http', res.status, await readDetail(res))
  },
  documents: () => request('/documents', documentListSchema),
  health: () => request('/health', healthSchema),
}

export function fileUrl(docId: string, page?: number): string {
  return `/api/documents/${encodeURIComponent(docId)}/file${page ? `#page=${page}` : ''}`
}
