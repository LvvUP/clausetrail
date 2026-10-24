import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useParams } from '@tanstack/react-router'
import { Library, PanelLeftClose, PanelLeftOpen, Plus, RotateCw, Trash2 } from 'lucide-react'
import { type MouseEvent, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { IconButton, Tip } from '@/components/ui/tooltip'
import { api } from '@/lib/api'
import { formatShort } from '@/lib/format'
import { chatStore } from '@/lib/live-store'
import { sessionsQuery } from '@/lib/queries'
import type { Session } from '@/lib/schemas'
import { cn } from '@/lib/utils'
import { HealthDot } from './health-dot'
import { Wordmark } from './logo'
import { ThemeToggle } from './theme-toggle'

/** 普通左键点击；按住修饰键或用中键是在新标签页打开，当前页的生成不应被中止。 */
export function plainClick(e: MouseEvent): boolean {
  return e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey
}

/** 离开正在生成的会话前先中止它并提示，生成中点「新建咨询」也能立刻切换。 */
export function leaveLive(target: string | null) {
  if (chatStore.get() === null || chatStore.owner() === target) return
  if (chatStore.leave()) toast('已停止上一条回答的生成')
}

export function Sidebar({
  collapsed = false,
  onToggle,
  hideBrand = false,
  onNavigate,
}: {
  collapsed?: boolean
  onToggle?: () => void
  hideBrand?: boolean
  /** 点了会话、新建、条款库之后调用；小屏抽屉靠它收起（路径没变时按路径收起的 effect 不会触发） */
  onNavigate?: () => void
}) {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const current = useParams({ strict: false }).sessionId ?? null
  const sessions = useQuery(sessionsQuery)
  const [pending, setPending] = useState<Session | null>(null)
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteSession(id),
    onSuccess: (_, id) => {
      qc.removeQueries({ queryKey: ['session', id] })
      void qc.invalidateQueries({ queryKey: ['sessions'] })
      if (chatStore.owner() === id) chatStore.leave()
      if (current === id) void navigate({ to: '/' })
      toast('已删除这条咨询记录')
    },
    onError: () => toast.error('这条记录没能删掉，过一会儿再试一次'),
  })
  const startNew = () => {
    leaveLive(null)
    onNavigate?.()
    void navigate({ to: '/' })
  }
  const follow = (e: MouseEvent, target: string | null) => {
    if (!plainClick(e)) return
    leaveLive(target)
    onNavigate?.()
  }

  if (collapsed) {
    return (
      <nav aria-label="主导航" className="flex h-full flex-col items-center gap-2 bg-surface py-3">
        <IconButton label="展开侧栏" side="right" onClick={onToggle}>
          <PanelLeftOpen />
        </IconButton>
        <IconButton label="新建咨询" side="right" variant="default" onClick={startNew}>
          <Plus />
        </IconButton>
        <Tip label="条款库" side="right">
          <Button asChild size="icon" variant="ghost" aria-label="条款库">
            <Link to="/library">
              <Library />
            </Link>
          </Button>
        </Tip>
        <div className="mt-auto flex flex-col items-center gap-2">
          <HealthDot compact />
          <ThemeToggle side="right" />
        </div>
      </nav>
    )
  }

  const list = sessions.data?.sessions ?? []
  return (
    <nav aria-label="主导航" className="flex h-full flex-col bg-surface">
      {!hideBrand && (
        <div className="flex h-14 shrink-0 items-center justify-between px-3">
          <Link to="/" onClick={(e) => follow(e, null)} className="rounded-control" aria-label="款知 ClauseTrail，回到提问页">
            <Wordmark />
          </Link>
          {onToggle && (
            <IconButton label="收起侧栏" side="right" onClick={onToggle}>
              <PanelLeftClose />
            </IconButton>
          )}
        </div>
      )}
      <div className={cn('px-3', hideBrand && 'pt-3')}>
        <Button className="w-full justify-start" onClick={startNew}>
          <Plus />
          新建咨询
        </Button>
      </div>
      <h2 className="mt-5 px-4 text-xs font-medium text-ink-muted">最近的咨询</h2>
      <div className="mt-1 min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {sessions.isPending ? (
          <div className="flex flex-col gap-2 px-2 pt-1" aria-hidden="true">
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-9 animate-pulse rounded-control bg-ink/[0.05]" />
            ))}
          </div>
        ) : sessions.isError ? (
          <div className="px-2 pt-1 text-xs text-ink-muted">
            <p>咨询记录没有加载出来。</p>
            <Button variant="ghost" size="sm" className="mt-1 -ml-2" onClick={() => void sessions.refetch()}>
              <RotateCw />
              重新加载
            </Button>
          </div>
        ) : list.length === 0 ? (
          <p className="px-2 pt-1 text-xs text-ink-muted">还没有咨询记录。提出第一个问题后会出现在这里。</p>
        ) : (
          <>
          <ul className="flex flex-col gap-0.5">
            {list.map((s) => {
              const active = s.session_id === current
              return (
                <li key={s.session_id} className="group relative">
                  <Link
                    to="/c/$sessionId"
                    params={{ sessionId: s.session_id }}
                    onClick={(e) => follow(e, s.session_id)}
                    aria-current={active ? 'page' : undefined}
                    className={cn(
                      'flex flex-col rounded-control border-l-2 border-transparent py-1.5 pr-9 pl-3 hover:bg-ink/[0.05]',
                      active && 'border-seal bg-seal/[0.07]',
                    )}
                  >
                    <span className="truncate text-sm">{s.title}</span>
                    <span className="font-mono text-xs text-ink-muted">{formatShort(s.updated_at)}</span>
                  </Link>
                  <IconButton
                    label={`删除「${s.title}」`}
                    onClick={() => setPending(s)}
                    className="absolute top-1/2 right-1 size-7 -translate-y-1/2 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 max-lg:opacity-100"
                  >
                    <Trash2 />
                  </IconButton>
                </li>
              )
            })}
          </ul>
          {/* 后端只返回最近 100 条，到了上限要说明，免得以为更早的记录没了 */}
          {list.length >= 100 && <p className="px-2 pt-2 text-xs text-ink-muted">这里只列出最近 100 条咨询。</p>}
          </>
        )}
      </div>
      <div className="flex flex-col gap-1 border-t border-rule p-2">
        <Link
          to="/library"
          onClick={(e) => plainClick(e) && onNavigate?.()}
          className="flex items-center gap-2 rounded-control px-2 py-1.5 text-sm text-ink-muted hover:bg-ink/[0.06] hover:text-ink [&.active]:text-ink"
        >
          <Library className="size-4" aria-hidden="true" />
          条款库
        </Link>
        <div className="flex items-center justify-between gap-2">
          <HealthDot />
          <ThemeToggle />
        </div>
      </div>
      <ConfirmDialog
        open={pending !== null}
        onOpenChange={(open) => !open && setPending(null)}
        title="删除这条咨询记录？"
        description={`「${pending?.title ?? ''}」里的问答会一并删除，无法恢复。`}
        confirmLabel="确认删除"
        onConfirm={() => pending && remove.mutate(pending.session_id)}
      />
    </nav>
  )
}
