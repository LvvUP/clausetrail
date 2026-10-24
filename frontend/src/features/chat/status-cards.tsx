import { CircleAlert, RotateCw, SearchX, TriangleAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { errorCopy, REFUSAL_COPY } from '@/lib/copy'
import { formatScore } from '@/lib/format'
import type { RefusalReason } from '@/lib/schemas'

/** 拒答卡片：琥珀色，按原因说明为什么不作答。拒答由后端决定，这时不会调用生成模型。 */
export function RefusalCard({
  refusal,
  retrieved,
  onShowRetrieved,
}: {
  refusal: { reason: RefusalReason; topScore: number } | null
  retrieved: number
  onShowRetrieved?: () => void
}) {
  const copy = refusal ? REFUSAL_COPY[refusal.reason] : { title: '这一问没有作答', hint: '条款依据不足，按规则不作答。' }
  const Icon = refusal?.reason === 'no_hit' ? SearchX : TriangleAlert
  return (
    <div role="status" className="rounded-card border border-caution/50 bg-caution/[0.07] px-4 py-3">
      <p className="flex items-center gap-2 text-sm font-semibold">
        <Icon className="size-4 text-caution" aria-hidden="true" />
        {copy.title}
      </p>
      <p className="mt-1 text-sm text-ink-muted">{copy.hint}</p>
      {refusal && refusal.reason !== 'no_hit' && (
        <p className="mt-2 font-mono text-xs text-ink-muted">最高相关度 {formatScore(refusal.topScore)}</p>
      )}
      {onShowRetrieved && retrieved > 0 && (
        <Button variant="ghost" size="sm" className="mt-2 -ml-2" onClick={onShowRetrieved}>
          看看检索到的 {retrieved} 段
        </Button>
      )}
    </div>
  )
}

/** 出错卡片：独立于正文，不显示引用，报错内容不会混进回答。 */
export function ErrorCard({
  error,
  onRetry,
  interrupted = false,
}: {
  error: { code: string; detail: string; status?: number } | null
  onRetry: () => void
  interrupted?: boolean
}) {
  const copy = errorCopy(error?.code ?? 'internal', error?.status)
  // 会话已经不在时，原地重问必然再失败，改成去新的咨询里重问
  const gone = error?.code === 'session_not_found'
  return (
    <div role="alert" className="rounded-card border border-danger/50 bg-danger/[0.06] px-4 py-3">
      <p className="flex items-center gap-2 text-sm font-semibold">
        <CircleAlert className="size-4 text-danger" aria-hidden="true" />
        {copy.title}
      </p>
      <p className="mt-1 text-sm text-ink-muted">{copy.hint}</p>
      {interrupted && <p className="mt-1 text-sm text-ink-muted">上面是中断前已经输出的部分。</p>}
      {error?.detail && (
        <details className="mt-2 text-xs text-ink-muted">
          <summary className="cursor-pointer select-none">技术细节</summary>
          <code className="mt-1 block font-mono break-all whitespace-pre-wrap">
            {error.code}：{error.detail}
          </code>
        </details>
      )}
      <Button variant="outline" size="sm" className="mt-3" onClick={onRetry}>
        <RotateCw />
        {gone ? '在新的咨询里重问' : '重新提问'}
      </Button>
    </div>
  )
}
