import { Link, useRouterState } from '@tanstack/react-router'
import { ArrowLeft, SearchX } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useDocTitle } from '@/hooks/use-media'

export function NotFound({
  title = '这个地址没有对应的页面',
  hint = '链接可能写错了，或者页面已经调整过位置。',
}: {
  title?: string
  hint?: string
}) {
  const path = useRouterState({ select: (s) => s.location.pathname })
  useDocTitle(title)
  return (
    <div className="mx-auto flex max-w-md flex-col items-center px-4 py-24 text-center">
      <SearchX className="size-10 text-ink-muted" aria-hidden="true" />
      <h1 className="mt-4 text-xl font-semibold">{title}</h1>
      <p className="mt-2 text-sm text-ink-muted">{hint}</p>
      <code className="mt-3 max-w-full truncate rounded-control bg-ink/[0.05] px-2 py-0.5 font-mono text-xs text-ink-muted">
        {path}
      </code>
      <Button asChild variant="secondary" className="mt-6">
        <Link to="/">
          <ArrowLeft />
          回到提问页
        </Link>
      </Button>
    </div>
  )
}
