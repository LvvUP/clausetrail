import { Link, useRouterState } from '@tanstack/react-router'
import { Menu } from 'lucide-react'
import { type ReactNode, useEffect, useState } from 'react'
import { Toaster } from 'sonner'
import { Sheet } from '@/components/ui/sheet'
import { IconButton } from '@/components/ui/tooltip'
import { DESKTOP, useMedia, useStoredFlag } from '@/hooks/use-media'
import { useTheme } from '@/lib/theme'
import { cn } from '@/lib/utils'
import { Wordmark } from './logo'
import { leaveLive, plainClick, Sidebar } from './sidebar'
import { ThemeToggle } from './theme-toggle'

export function AppShell({ children }: { children: ReactNode }) {
  const desktop = useMedia(DESKTOP)
  const [collapsed, setCollapsed] = useStoredFlag('ct-sidebar-collapsed', false)
  const [menuOpen, setMenuOpen] = useState(false)
  const path = useRouterState({ select: (s) => s.location.pathname })
  const { resolved } = useTheme()

  // 小屏抽屉在跳转后自动收起
  useEffect(() => {
    setMenuOpen(false)
  }, [path])

  return (
    <div className="flex h-full">
      {desktop ? (
        <aside className={cn('shrink-0 border-r border-rule', collapsed ? 'w-14' : 'w-[264px]')}>
          <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
        </aside>
      ) : (
        <Sheet open={menuOpen} onOpenChange={setMenuOpen} side="left" title="咨询记录">
          <Sidebar hideBrand onNavigate={() => setMenuOpen(false)} />
        </Sheet>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        {!desktop && (
          <header className="flex h-12 shrink-0 items-center gap-2 border-b border-rule bg-surface px-2">
            <IconButton label="打开咨询记录" side="bottom" onClick={() => setMenuOpen(true)}>
              <Menu />
            </IconButton>
            <Link to="/" onClick={(e) => plainClick(e) && leaveLive(null)} aria-label="款知 ClauseTrail，回到提问页" className="rounded-control">
              <Wordmark />
            </Link>
            <div className="ml-auto">
              <ThemeToggle side="bottom" />
            </div>
          </header>
        )}
        <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
      </div>
      <Toaster theme={resolved} position="top-center" toastOptions={{ style: { fontFamily: 'var(--font-body)' } }} />
    </div>
  )
}
