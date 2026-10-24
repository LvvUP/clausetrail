import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { Tip } from '@/components/ui/tooltip'
import { healthQuery } from '@/lib/queries'
import { cn } from '@/lib/utils'

export function HealthDot({ compact = false }: { compact?: boolean }) {
  const { data, isError } = useQuery(healthQuery)
  const degraded = data && Object.values(data.components).some((c) => c.status === 'degraded')
  const [tone, text] = isError
    ? ['bg-danger', '连不上问答服务']
    : !data
      ? ['bg-rule', '正在检查服务']
      : data.status !== 'ok'
        ? ['bg-danger', '模型未就绪']
        : degraded
          ? ['bg-caution', '服务可用，部分组件降级']
          : ['bg-evidence', '服务正常']
  const link = (
    <Link
      to="/library"
      hash="components"
      className="flex min-w-0 items-center gap-2 rounded-control px-2 py-1 text-xs text-ink-muted hover:bg-ink/[0.06] hover:text-ink"
      aria-label={compact ? `${text}，查看检索组件` : undefined}
    >
      <span className={cn('size-2 shrink-0 rounded-full', tone)} aria-hidden="true" />
      {!compact && <span className="truncate">{text}</span>}
    </Link>
  )
  return compact ? (
    <Tip label={text} side="right">
      {link}
    </Tip>
  ) : (
    link
  )
}
