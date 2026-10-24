import { ArrowUp, Square } from 'lucide-react'
import { type FormEvent, type KeyboardEvent, useLayoutEffect, useRef } from 'react'
import { IconButton } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

const LIMIT = 500

export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  busy,
  autoFocus = false,
}: {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onStop: () => void
  busy: boolean
  autoFocus?: boolean
}) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  const over = value.length > LIMIT
  const empty = !value.trim()
  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    if (!busy && !empty && !over) onSubmit()
  }
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // 输入法组字时的回车不算发送；Safari 选词回车时 isComposing 已经是 false，要再看 keyCode 229
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing && e.keyCode !== 229) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="shrink-0 border-t border-rule bg-bg px-4 pt-3 pb-4">
      <form onSubmit={submit} className="mx-auto max-w-[760px]">
        <div className="flex items-end gap-2 rounded-card border border-rule bg-surface p-2 shadow-hair focus-within:border-seal/70 focus-within:ring-1 focus-within:ring-seal/30">
          <label htmlFor="question" className="sr-only">
            要咨询的条款问题
          </label>
          <textarea
            id="question"
            ref={ref}
            rows={1}
            autoFocus={autoFocus}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="就条款内容提问，例如：恒裕一生的犹豫期有多长"
            className="max-h-[200px] min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-base leading-6 outline-none placeholder:text-ink-muted"
          />
          {busy ? (
            <IconButton label="停止生成" variant="secondary" onClick={onStop}>
              <Square className="fill-current" />
            </IconButton>
          ) : (
            <IconButton label="发出问题" type="submit" variant="default" disabled={empty || over}>
              <ArrowUp />
            </IconButton>
          )}
        </div>
        <div className="mt-1.5 flex justify-between gap-2 px-1 text-xs text-ink-muted">
          <span className="hidden sm:inline">Enter 发出，Shift + Enter 换行。回答只依据已入库的条款。</span>
          {value.length > 400 && (
            <span className={cn('ml-auto font-mono', over && 'text-danger')}>
              {value.length} / {LIMIT}
            </span>
          )}
        </div>
      </form>
    </div>
  )
}
