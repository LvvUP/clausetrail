import { type ComponentProps, createContext, useContext, useMemo } from 'react'
import Markdown, { type Components, type ExtraProps } from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import { citationNumber, linkCitations } from '@/lib/citations'
import { cn } from '@/lib/utils'

export function CiteButton({ n, active, onClick }: { n: number; active?: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`查看第 ${n} 段依据`}
      className={cn(
        // after 伪元素把可点区域扩到 28px，满足触控目标尺寸，视觉大小不变
        "relative mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-control border border-seal/40 bg-seal/[0.08] px-1 align-[2px] font-mono text-xs leading-none text-seal after:absolute after:-inset-1 after:content-[''] hover:bg-seal/[0.16]",
        active && 'border-seal bg-seal text-seal-ink hover:bg-seal',
      )}
    >
      {n}
    </button>
  )
}

const CiteContext = createContext<{ activeN: number | null; onCite: (n: number) => void }>({
  activeN: null,
  onCite: () => undefined,
})

type HastNode = { type: string; tagName?: string; value?: string; children?: HastNode[] }

// 图片不渲染，只包着图片的链接（[![x](…)](…)）就成了能聚焦、没有名称的空链接，整个不渲染
function hasVisible(node: HastNode): boolean {
  if (node.type === 'text') return Boolean(node.value?.trim())
  if (node.type === 'element' && node.tagName === 'img') return false
  return (node.children ?? []).some(hasVisible)
}

function MarkdownLink({ href, children, node }: ComponentProps<'a'> & ExtraProps) {
  const { activeN, onCite } = useContext(CiteContext)
  const n = citationNumber(href)
  if (n !== null) return <CiteButton n={n} active={activeN === n} onClick={() => onCite(n)} />
  if (node && !hasVisible(node)) return null
  return (
    <a href={href} target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  )
}

// components 放在模块级：每次渲染新建的话组件类型都会变，引用按钮会被整个卸载重装，流式输出时点不中、点完焦点丢失。
// 图片一律不渲染：条款文本里如果混进了诱导指令，模型输出的外链图片会让浏览器把内容带到别的域名
const COMPONENTS: Components = { a: MarkdownLink, img: () => null }

/** 模型输出经 rehype-sanitize 净化后再渲染，不直接插入 HTML，防止 XSS。 */
export function AnswerMarkdown({
  text,
  count,
  streaming = false,
  activeN,
  onCite,
}: {
  text: string
  count: number
  streaming?: boolean
  activeN: number | null
  onCite: (n: number) => void
}) {
  // 正文只在文字变化时重新解析；点了别的编号只通过 context 更新按钮的高亮
  const body = useMemo(
    () => (
      <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]} components={COMPONENTS}>
        {linkCitations(text, count)}
      </Markdown>
    ),
    [text, count],
  )
  const cite = useMemo(() => ({ activeN, onCite }), [activeN, onCite])
  return (
    <div className={cn('answer-prose', streaming && 'caret')}>
      <CiteContext.Provider value={cite}>{body}</CiteContext.Provider>
    </div>
  )
}
