import { useQuery } from '@tanstack/react-query'
import { RotateCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useDocTitle } from '@/hooks/use-media'
import { COMPONENT_LABELS, COMPONENT_STATUS } from '@/lib/copy'
import { documentsQuery, healthQuery } from '@/lib/queries'
import type { HealthComponent } from '@/lib/schemas'
import { cn } from '@/lib/utils'
import { DocTable } from './doc-table'

const TONE = { ok: 'bg-evidence', degraded: 'bg-caution', unavailable: 'bg-danger' }

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-card border border-rule bg-surface px-4 py-3">
      <p className="text-xs text-ink-muted">{label}</p>
      <p className="mt-1 font-mono text-xl">{value}</p>
    </div>
  )
}

function ComponentCard({ name, c }: { name: string; c: HealthComponent }) {
  const extra = [c.index && `索引 ${c.index}`, c.dim && `${c.dim} 维`].filter(Boolean).join(' · ')
  return (
    <li className="rounded-card border border-rule bg-surface p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-medium">{COMPONENT_LABELS[name] ?? name}</p>
        <span className="inline-flex items-center gap-1.5 rounded-control border border-rule px-1.5 text-xs">
          <span className={cn('size-1.5 rounded-full', TONE[c.status])} aria-hidden="true" />
          {COMPONENT_STATUS[c.status]}
        </span>
      </div>
      <p className="mt-1 font-mono text-xs break-all text-ink-muted">{c.backend}</p>
      {extra && <p className="font-mono text-xs text-ink-muted">{extra}</p>}
      {c.detail && <p className="mt-1 text-xs text-ink-muted">{c.detail}</p>}
    </li>
  )
}

export function LibraryPage() {
  useDocTitle('条款库')
  const docs = useQuery(documentsQuery)
  const health = useQuery(healthQuery)
  const list = docs.data?.documents ?? []
  const sum = (k: 'pages' | 'chunks') => list.reduce((n, d) => n + d[k], 0)
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex max-w-[1080px] flex-col gap-8 px-4 py-8">
        <header>
          <h1 className="text-display leading-tight font-semibold">条款库</h1>
          <p className="mt-2 max-w-2xl text-sm text-ink-muted">
            问答只会引用这里列出的条款。新条款通过命令行入库：
            <code className="mx-1 rounded-control bg-ink/[0.05] px-1.5 font-mono text-xs">clausetrail ingest 路径</code>
            或 <code className="rounded-control bg-ink/[0.05] px-1.5 font-mono text-xs">make ingest</code>。
          </p>
        </header>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="索引版本" value={docs.data ? `v${docs.data.kb_version}` : '–'} />
          <Stat label="条款文件" value={docs.data ? list.length : '–'} />
          <Stat label="总页数" value={docs.data ? sum('pages') : '–'} />
          <Stat label="检索片段" value={docs.data ? sum('chunks') : '–'} />
        </div>
        <section aria-labelledby="docs-title">
          <h2 id="docs-title" className="mb-3 text-base font-semibold">
            已入库的条款
          </h2>
          {docs.isPending ? (
            <div className="h-40 animate-pulse rounded-card bg-ink/[0.05]" aria-hidden="true" />
          ) : docs.isError ? (
            <div className="text-sm text-ink-muted" role="alert">
              <p>条款清单没有加载出来，确认 API 已经启动。</p>
              <Button variant="outline" size="sm" className="mt-2" onClick={() => void docs.refetch()}>
                <RotateCw />
                重新加载
              </Button>
            </div>
          ) : list.length === 0 ? (
            <p className="rounded-card border border-dashed border-rule px-4 py-8 text-center text-sm text-ink-muted">
              还没有入库任何条款。在项目根目录运行 <code className="font-mono">make ingest</code>，样例条款就会出现在这里。
            </p>
          ) : (
            <DocTable docs={list} />
          )}
        </section>
        <section id="components" aria-labelledby="components-title" className="scroll-mt-6">
          <h2 id="components-title" className="text-base font-semibold">
            检索组件
          </h2>
          <p className="mt-1 mb-3 text-sm text-ink-muted">
            当前实际使用的后端。标为「已降级」的组件仍能作答，只是走了备用实现，例如连不上 Elasticsearch 时改用 Milvus 内置 BM25。
          </p>
          {health.data ? (
            <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(health.data.components).map(([name, c]) => (
                <ComponentCard key={name} name={name} c={c} />
              ))}
            </ul>
          ) : health.isError ? (
            <p className="text-sm text-ink-muted">组件信息没有加载出来。</p>
          ) : (
            <div className="h-24 animate-pulse rounded-card bg-ink/[0.05]" aria-hidden="true" />
          )}
        </section>
      </div>
    </div>
  )
}
