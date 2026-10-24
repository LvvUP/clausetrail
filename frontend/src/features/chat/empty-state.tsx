import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { LogoMark } from '@/components/shell/logo'
import { documentsQuery, healthQuery } from '@/lib/queries'

// 示例问题都对应样例条款里真实存在的内容
const EXAMPLES = [
  { topic: '等待期', q: '康宁悦享的等待期是多少天？' },
  { topic: '责任免除', q: '恒裕一生在哪些情况下不承担给付责任？' },
  { topic: '理赔条件', q: '安医保e享申请理赔需要提供哪些材料？' },
  { topic: '保障范围', q: '康医保易享的住院医疗费用保险金包括哪些费用？' },
]

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  const docs = useQuery(documentsQuery)
  const health = useQuery(healthQuery)
  const n = docs.data?.documents.length
  return (
    <div className="flex flex-col items-start gap-6 py-6">
      <LogoMark className="size-12" />
      <div>
        <h2 className="text-display leading-tight font-semibold">问一问条款里是怎么写的</h2>
        <p className="mt-2 max-w-xl text-base text-ink-muted">
          每句回答都会标出依据的条款原文，点编号就能对照。条款里没有写的，会直接说明不作答。
        </p>
      </div>
      {n === 0 ? (
        <p className="rounded-card border border-caution/50 bg-caution/[0.07] px-3 py-2 text-sm">
          条款库还是空的。先在命令行运行 <code className="font-mono">make ingest</code> 入库样例条款。
        </p>
      ) : n ? (
        <p className="text-sm text-ink-muted">
          当前可检索 {n} 份条款，
          <Link to="/library" className="text-seal underline-offset-2 hover:underline">
            查看条款库
          </Link>
        </p>
      ) : null}
      <ul className="grid w-full gap-3 sm:grid-cols-2" aria-label="示例问题">
        {EXAMPLES.map((e) => (
          <li key={e.q}>
            <button
              type="button"
              onClick={() => onPick(e.q)}
              className="flex h-full w-full flex-col items-start gap-1 rounded-card border border-rule bg-surface px-4 py-3 text-left shadow-hair hover:border-seal/50"
            >
              <span className="font-mono text-xs text-seal">{e.topic}</span>
              <span className="text-sm">{e.q}</span>
            </button>
          </li>
        ))}
      </ul>
      {health.data && health.data.status !== 'ok' && (
        <p className="text-sm text-danger" role="status">
          模型服务还没配置好，现在提问会失败。可以到条款库页面看看各检索组件是否可用。
        </p>
      )}
    </div>
  )
}
