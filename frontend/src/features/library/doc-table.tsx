import { ExternalLink } from 'lucide-react'
import { fileUrl } from '@/lib/api'
import { DOC_STATUS, PAGE_TYPE_LABELS, PARSER_LABELS } from '@/lib/copy'
import { formatTime } from '@/lib/format'
import type { ClauseDocument } from '@/lib/schemas'

const byCount = (r: Record<string, number>) => Object.entries(r).sort((a, b) => b[1] - a[1])

function StatusPill({ status }: { status: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-control border border-rule px-1.5 text-xs">
      <span className="size-1.5 rounded-full bg-evidence" aria-hidden="true" />
      {DOC_STATUS[status] ?? status}
    </span>
  )
}

function PageTypes({ d }: { d: ClauseDocument }) {
  return (
    <span className="flex flex-wrap gap-1">
      {byCount(d.page_types).map(([k, n]) => (
        <span key={k} className="rounded-control bg-ink/[0.05] px-1.5 text-xs">
          {PAGE_TYPE_LABELS[k] ?? k} <span className="font-mono">{n}</span>
        </span>
      ))}
    </span>
  )
}

function Parsers({ d }: { d: ClauseDocument }) {
  return <span className="text-xs text-ink-muted">{byCount(d.parsers).map(([k]) => PARSER_LABELS[k] ?? k).join(' · ')}</span>
}

function OpenLink({ d }: { d: ClauseDocument }) {
  return (
    <a
      href={fileUrl(d.doc_id)}
      target="_blank"
      rel="noreferrer"
      aria-label={`打开原文：${d.file_name}`}
      className="inline-flex items-center gap-1 text-xs text-seal hover:underline"
    >
      打开原文
      <ExternalLink className="size-3.5" aria-hidden="true" />
    </a>
  )
}

export function DocTable({ docs }: { docs: ClauseDocument[] }) {
  return (
    <>
      <div className="hidden overflow-x-auto rounded-card border border-rule bg-surface md:block">
        <table className="w-full text-sm">
          <thead className="border-b border-rule text-left text-xs text-ink-muted">
            <tr>
              <th className="px-3 py-2 font-medium">条款文件</th>
              <th className="px-3 py-2 text-right font-medium">页数</th>
              <th className="px-3 py-2 text-right font-medium">检索片段</th>
              <th className="px-3 py-2 font-medium">页面类型</th>
              <th className="px-3 py-2 font-medium">解析器</th>
              <th className="px-3 py-2 font-medium">入库于</th>
              <th className="px-3 py-2">
                <span className="sr-only">原文</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.doc_id} className="border-b border-rule last:border-0 align-top">
                <td className="max-w-[280px] px-3 py-2.5">
                  <p className="truncate font-medium" title={d.file_name}>
                    {d.file_name}
                  </p>
                  <div className="mt-1 flex items-center gap-2">
                    <StatusPill status={d.status} />
                    <span className="font-mono text-xs text-ink-muted">v{d.kb_version}</span>
                  </div>
                </td>
                <td className="px-3 py-2.5 text-right font-mono">{d.pages}</td>
                <td className="px-3 py-2.5 text-right font-mono">{d.chunks}</td>
                <td className="px-3 py-2.5">
                  <PageTypes d={d} />
                </td>
                <td className="px-3 py-2.5">
                  <Parsers d={d} />
                </td>
                <td className="px-3 py-2.5 font-mono text-xs whitespace-nowrap text-ink-muted">
                  {formatTime(d.ingested_at)}
                </td>
                <td className="px-3 py-2.5">
                  <OpenLink d={d} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="flex flex-col gap-3 md:hidden">
        {docs.map((d) => (
          <li key={d.doc_id} className="rounded-card border border-rule bg-surface p-3">
            <p className="text-sm font-medium break-all">{d.file_name}</p>
            <p className="mt-1 font-mono text-xs text-ink-muted">
              {d.pages} 页 · {d.chunks} 个检索片段 · {formatTime(d.ingested_at)}
            </p>
            <div className="mt-2 flex flex-col gap-1.5">
              <PageTypes d={d} />
              <Parsers d={d} />
            </div>
            <div className="mt-2 flex items-center justify-between">
              <StatusPill status={d.status} />
              <OpenLink d={d} />
            </div>
          </li>
        ))}
      </ul>
    </>
  )
}
