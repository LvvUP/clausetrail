import { Check, Copy, LoaderCircle, Quote, RotateCw, Zap } from 'lucide-react'
import { memo, useCallback, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { IconButton } from '@/components/ui/tooltip'
import { evidenceSummary } from '@/lib/citations'
import { formatSeconds, formatTime } from '@/lib/format'
import type { TurnView } from '@/lib/turns'
import { AnswerMarkdown } from './answer-markdown'
import { ErrorCard, RefusalCard } from './status-cards'

interface Props {
  view: TurnView
  first: boolean
  activeN: number | null
  onCite: (turnId: string, n: number) => void
  onShowEvidence: (turnId: string) => void
  onRetry: (view: TurnView) => void
}

interface AnswerProps {
  view: TurnView
  activeN: number | null
  onCite: (n: number) => void
  onShowEvidence: () => void
  onRetry: () => void
}

// memo：回调由 Workbench 保持同一个引用，已完成的回合在打字、流式输出时不重渲染
export const TurnBlock = memo(function TurnBlock({ view, first, activeN, onCite, onShowEvidence, onRetry }: Props) {
  const cite = useCallback((n: number) => onCite(view.id, n), [onCite, view.id])
  const show = useCallback(() => onShowEvidence(view.id), [onShowEvidence, view.id])
  const retry = useCallback(() => onRetry(view), [onRetry, view])
  return (
    <article className="flex flex-col gap-4">
      {/* 不用 role="separator"：它的子内容在无障碍树里是展示性的，读屏听不到「新话题」 */}
      {view.topic === 'switch' && !first && (
        <p className="flex items-center gap-3 text-xs text-ink-muted">
          <span aria-hidden="true" className="h-px flex-1 bg-rule" />
          新话题
          <span aria-hidden="true" className="h-px flex-1 bg-rule" />
        </p>
      )}
      <div className="rounded-card border border-rule bg-surface px-4 py-3 shadow-hair">
        <div className="flex items-start gap-3">
          <p className="min-w-0 flex-1 text-base whitespace-pre-wrap">{view.question}</p>
          {/* 时间取后端记录的 created_at，不用渲染时刻，历史回合才显示得对 */}
          <time dateTime={view.askedAt} className="shrink-0 pt-0.5 font-mono text-xs text-ink-muted">
            {formatTime(view.askedAt)}
          </time>
        </div>
        {view.rewritten && (
          <p className="mt-2 border-t border-dashed border-rule pt-2 text-sm text-ink-muted">
            已理解为：<span className="text-ink">{view.rewritten}</span>
          </p>
        )}
      </div>
      <div className="flex gap-3">
        <span
          aria-hidden="true"
          className="mt-1 flex size-6 shrink-0 items-center justify-center rounded-control border border-seal/50 font-serif text-xs text-seal"
        >
          答
        </span>
        <div className="min-w-0 flex-1">
          <Answer view={view} activeN={activeN} onCite={cite} onShowEvidence={show} onRetry={retry} />
        </div>
      </div>
    </article>
  )
})

function Answer({ view, activeN, onCite, onShowEvidence, onRetry }: AnswerProps) {
  const count = view.citations.length
  switch (view.phase) {
    case 'retrieving':
      return (
        <div className="flex flex-col gap-2" role="status">
          <p className="flex items-center gap-2 text-sm text-ink-muted">
            <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
            正在检索相关条款
          </p>
          <div className="h-3 w-4/5 animate-pulse rounded-control bg-ink/[0.06]" />
          <div className="h-3 w-3/5 animate-pulse rounded-control bg-ink/[0.06]" />
        </div>
      )
    case 'streaming':
      return (
        <div className="flex flex-col gap-3">
          <p className="flex items-center gap-2 text-xs text-ink-muted" role="status">
            <LoaderCircle className="size-3.5 animate-spin" aria-hidden="true" />
            已拿到 {count} 段候选原文，正在组织回答
          </p>
          {/* aria-busy 只标在还在生成的正文上：包住上面的进度行时，有的读屏会连进度一起不播报 */}
          <div aria-busy="true">
            <AnswerMarkdown text={view.text} count={count} streaming activeN={activeN} onCite={onCite} />
          </div>
        </div>
      )
    case 'answered':
      return (
        <div className="flex flex-col gap-3">
          <AnswerMarkdown text={view.text} count={count} activeN={activeN} onCite={onCite} />
          <EvidenceStrip view={view} onOpen={onShowEvidence} />
          <AnswerMeta view={view} />
        </div>
      )
    case 'refused':
      return (
        <RefusalCard refusal={view.refusal} retrieved={count} onShowRetrieved={count ? onShowEvidence : undefined} />
      )
    case 'error':
      // 中途出错时保留已经输出的文字，和中止的回合一致
      return (
        <div className="flex flex-col gap-3">
          {view.text && <AnswerMarkdown text={view.text} count={count} activeN={activeN} onCite={onCite} />}
          <ErrorCard error={view.error} onRetry={onRetry} interrupted={Boolean(view.text)} />
        </div>
      )
    case 'aborted':
      return (
        <div className="flex flex-col gap-3">
          {view.text && <AnswerMarkdown text={view.text} count={count} activeN={activeN} onCite={onCite} />}
          <p className="text-sm text-ink-muted">
            {view.text ? '生成被中途停止，上面是停止前已输出的部分。' : '这一问在生成前就被停止了。'}
          </p>
          <div>
            <Button variant="outline" size="sm" onClick={onRetry}>
              <RotateCw />
              重新提问
            </Button>
          </div>
        </div>
      )
  }
}

function EvidenceStrip({ view, onOpen }: { view: TurnView; onOpen: () => void }) {
  const s = evidenceSummary(view.citations, view.used)
  if (!s.total) return null
  return (
    <button
      type="button"
      onClick={onOpen}
      className="flex w-fit flex-wrap items-center gap-x-2 rounded-control border border-evidence/40 bg-evidence/[0.06] px-3 py-1 text-left text-xs text-evidence hover:bg-evidence/[0.12]"
    >
      <Quote className="size-3.5" aria-hidden="true" />
      依据 {s.cited} 段条款原文，出自 {s.files} 份条款
      <span className="text-ink-muted">· 共检索到 {s.total} 段</span>
    </button>
  )
}

function AnswerMeta({ view }: { view: TurnView }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(view.text)
      setCopied(true)
      toast('回答已复制')
      setTimeout(() => setCopied(false), 1500)
    } catch {
      toast.error('浏览器没有允许写入剪贴板')
    }
  }
  const secs = formatSeconds(view.totalMs)
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-ink-muted">
      <IconButton label={copied ? '已复制' : '复制回答'} className="-ml-2 size-7" onClick={() => void copy()}>
        {copied ? <Check /> : <Copy />}
      </IconButton>
      {view.cacheHit && (
        <span className="flex items-center gap-1">
          <Zap className="size-3.5" aria-hidden="true" />
          复用了之前相同问题的回答
        </span>
      )}
      {secs && <span className="font-mono">用时 {secs}</span>}
    </div>
  )
}
