import { cn } from '@/lib/utils'

/** 标识：一页条款 + 一条从正文引出、落到出处圆点上的追踪线。 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={cn('size-7', className)}>
      <path
        d="M7 3.5h12.5L25 9v19.5H7z"
        fill="var(--surface)"
        stroke="var(--ink)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path d="M19.5 3.5V9H25" fill="none" stroke="var(--ink)" strokeWidth="2" strokeLinejoin="round" />
      <path d="M11 13h7M11 17.5h4.5" stroke="var(--ink)" strokeWidth="2" strokeLinecap="round" />
      <path d="M11 22h6.5a4 4 0 0 0 4-4v-1.5" fill="none" stroke="var(--seal)" strokeWidth="2" strokeLinecap="round" />
      <circle cx="21.5" cy="14" r="2" fill="var(--seal)" />
    </svg>
  )
}

export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <span className="flex items-center gap-2">
      <LogoMark />
      {!compact && (
        <span className="flex items-baseline gap-1.5">
          <span className="text-base font-semibold tracking-wide">款知</span>
          <span className="font-mono text-xs text-ink-muted">ClauseTrail</span>
        </span>
      )}
    </span>
  )
}
