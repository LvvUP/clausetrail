import { ExternalLink, X } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { IconButton } from '@/components/ui/tooltip'
import { fileUrl } from '@/lib/api'
import { claimsFor, highlightRanges, splitByRanges } from '@/lib/citations'
import { formatScore } from '@/lib/format'
import type { Citation } from '@/lib/schemas'
import type { TurnView } from '@/lib/turns'
import { cn } from '@/lib/utils'

export function EvidencePanel({
  view,
  focusN,
  focusKey,
  onClose,
  showHeader,
}: {
  view: TurnView
  focusN: number | null
  /** 每次点击都会变：再点同一个 [n] 也重新定位 */
  focusKey: number
  onClose: () => void
  showHeader: boolean
}) {
  // 点正文里的 [n] 时定位到第 n 段；系统设置了「减少动画」时不做平滑滚动
  useEffect(() => {
    if (focusN === null) return
    const el = document.getElementById(`evidence-${view.id}-${focusN}`)
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    el?.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' })
    el?.focus({ preventScroll: true })
  }, [focusN, focusKey, view.id])

  const n = view.citations.length
  const note =
    view.phase === 'refused'
      ? `检索到 ${n} 段，但都没有用于作答。`
      : view.phase === 'answered'
        ? `检索到 ${n} 段，回答引用了其中 ${view.used.length} 段。`
        : view.phase === 'aborted' || view.phase === 'error'
          ? `检索到 ${n} 段；回答没有完成，标出的是已输出部分引用到的段落。`
          : `已检索到 ${n} 段，被引用的段落会随回答逐步标出。`
  return (
    <div className="flex h-full flex-col">
      {showHeader && (
        <div className="flex h-12 shrink-0 items-center justify-between border-b border-rule px-4">
          <h2 className="text-sm font-semibold">回答依据</h2>
          <IconButton label="收起回答依据" side="left" onClick={onClose}>
            <X />
          </IconButton>
        </div>
      )}
      <p className="px-4 pt-3 text-xs text-ink-muted">{note}</p>
      <ol className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-4">
        {view.citations.map((c) => {
          const used = view.used.includes(c.n)
          return (
            <EvidenceItem
              key={c.n}
              id={`evidence-${view.id}-${c.n}`}
              c={c}
              used={used}
              focused={focusN === c.n}
              text={used ? view.text : ''}
            />
          )
        })}
      </ol>
    </div>
  )
}

function EvidenceItem({
  id,
  c,
  used,
  focused,
  text,
}: {
  id: string
  c: Citation
  used: boolean
  focused: boolean
  text: string
}) {
  // 最长公共子串是 O(n·m)，只在回答文字或片段变化时重算，点别的编号不重算
  const parts = useMemo(
    () => splitByRanges(c.snippet, highlightRanges(c.snippet, text ? claimsFor(text, c.n) : [])),
    [c.snippet, c.n, text],
  )
  const pages = c.page_end > c.page_start ? `第 ${c.page_start}–${c.page_end} 页` : `第 ${c.page_start} 页`
  return (
    <li
      id={id}
      tabIndex={-1}
      className={cn('rounded-card border bg-surface p-3', focused ? 'border-seal ring-1 ring-seal/40' : 'border-rule')}
    >
      <div className="flex items-start gap-2">
        <span
          className={cn(
            'flex h-5 min-w-5 shrink-0 items-center justify-center rounded-control px-1 font-mono text-xs',
            used ? 'bg-seal text-seal-ink' : 'border border-rule text-ink-muted',
          )}
        >
          {c.n}
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium" title={c.file}>
            {c.file}
          </p>
          <p className="text-xs text-ink-muted">
            {c.section_path.join(' › ') || '（无章节）'} · {pages}
          </p>
        </div>
        <span
          className={cn('shrink-0 rounded-control px-1.5 text-xs', used ? 'bg-evidence/[0.1] text-evidence' : 'text-ink-muted')}
        >
          {used ? '已引用' : '未引用'}
        </span>
      </div>
      <blockquote className="clause-text mt-2 max-h-72 overflow-y-auto border-l-2 border-rule pl-3">
        {parts.map((p, i) => (p.marked ? <mark key={i}>{p.text}</mark> : <span key={i}>{p.text}</span>))}
      </blockquote>
      <div className="mt-2 flex items-center justify-between gap-2 text-xs text-ink-muted">
        <span className="font-mono">相关度 {formatScore(c.rerank_score)}</span>
        <a
          href={fileUrl(c.doc_id, c.page_start)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 text-seal hover:underline"
        >
          打开原文第 {c.page_start} 页
          <ExternalLink className="size-3.5" aria-hidden="true" />
        </a>
      </div>
    </li>
  )
}
