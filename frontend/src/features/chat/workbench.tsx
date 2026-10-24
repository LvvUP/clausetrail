import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useRouter } from '@tanstack/react-router'
import { PanelRightClose, PanelRightOpen, RotateCw } from 'lucide-react'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { NotFound } from '@/components/shell/not-found'
import { Button } from '@/components/ui/button'
import { Sheet } from '@/components/ui/sheet'
import { IconButton } from '@/components/ui/tooltip'
import { DESKTOP, useDocTitle, useMedia } from '@/hooks/use-media'
import { ApiError } from '@/lib/api'
import { chatStore, useLiveTurn } from '@/lib/live-store'
import { sessionQuery } from '@/lib/queries'
import { mergeTurns, type TurnView } from '@/lib/turns'
import { Composer } from './composer'
import { EmptyState } from './empty-state'
import { EvidencePanel } from './evidence-panel'
import { TurnBlock } from './turn-block'

// tick 每次点击都加一：再点同一个 [n] 也会重新定位
type Focus = { turnId: string; n: number | null; tick: number } | null

// 首问拿到 session_id 后从 / 跳到 /c/$id，Workbench 会重新挂载；这时输入框里已经写的字和焦点带过去，不丢。
// 记下要带去哪个会话（null 是新的咨询），只有挂载到那里的实例才接手：跳转被别的导航抢掉时不会串到别处
let carried: { sessionId: string | null; draft: string; focused: boolean } | null = null
// 当前挂载着的那个实例的输入框内容：onSession 是发问那个实例的闭包，它可能已经卸载（去过条款库又回来），要读现在这个
let activeDraft: { current: string } | null = null

export function Workbench({ sessionId }: { sessionId: string | null }) {
  const navigate = useNavigate()
  const router = useRouter()
  const qc = useQueryClient()
  const desktop = useMedia(DESKTOP)
  const live = useLiveTurn()
  const [restored] = useState(() => (carried?.sessionId === sessionId ? carried : null))
  const [draft, setDraft] = useState(restored?.draft ?? '')
  const draftRef = useRef(draft)
  draftRef.current = draft
  const [focus, setFocus] = useState<Focus>(null)
  const tick = useRef(0)
  const scroller = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  useEffect(() => {
    carried = null
    activeDraft = draftRef
    return () => {
      if (activeDraft === draftRef) activeDraft = null
    }
  }, [])

  // 切到别的会话时中止正在生成的回答，免得上一个会话的输出继续写进来
  useLayoutEffect(() => {
    if (chatStore.get() && chatStore.owner() !== sessionId && chatStore.leave()) toast('已停止上一条回答的生成')
  }, [sessionId])

  // 状态按 sessionId 隔离，切换会话后不会残留上一个会话的证据
  const detail = useQuery({ ...sessionQuery(sessionId ?? '-'), enabled: sessionId !== null })
  const ask = useMutation({
    mutationFn: (q: string) =>
      chatStore.ask(q, sessionId, (id) => {
        // 新会话一建好就刷新侧栏，不用等整轮结束
        void qc.invalidateQueries({ queryKey: ['sessions'] })
        // 用户已经切到条款库等别的页面时不把人拉回来
        if (router.state.location.pathname !== '/') return
        carried = {
          sessionId: id,
          draft: activeDraft?.current ?? '',
          focused: document.activeElement?.id === 'question',
        }
        void navigate({ to: '/c/$sessionId', params: { sessionId: id } })
      }),
    // 会话 id 取 ask 的返回值：live 可能已经被 leave() 清空，或者换成了下一问
    onSettled: (data) => {
      void qc.invalidateQueries({ queryKey: ['sessions'] })
      const id = data?.sessionId ?? sessionId
      if (id) void qc.invalidateQueries({ queryKey: ['session', id] })
    },
  })

  const mine = live !== null && (live.sessionId === sessionId || live.origin === sessionId)
  const mineLive = mine ? live : null
  const views = useMemo(() => mergeTurns(detail.data?.turns ?? [], mineLive), [detail.data, mineLive])
  // 流没结束就算忙：refusal / error 之后、done 之前发下一问，会把这一轮已经定下的状态冲掉
  const busy = mine && live.open
  // 读屏软件只在一轮结束时播报一次，不逐字朗读流式输出。
  // 拒答卡片（role=status）和出错卡片（role=alert）自己会播报，这里只播报正常作答，免得同一件事听两遍
  const announce = mine && live.phase === 'answered' ? '回答已经生成。' : ''
  const missing =
    detail.error instanceof ApiError && detail.error.status >= 400 && detail.error.status < 500 && !mine
  const title = missing ? '找不到这条咨询记录' : (detail.data?.session.title ?? (mine ? live.question : ''))
  useDocTitle(title)

  const lastKey = `${views.length}:${mine ? live.text.length : 0}:${mine ? live.phase : ''}`
  useEffect(() => {
    const el = scroller.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [lastKey])
  useEffect(() => setFocus(null), [sessionId])

  const submit = (text = draft) => {
    const q = text.trim()
    if (!q || busy) return
    // 发出即清空输入框，不等整段回答结束；会话还没建立就失败时把问题放回去。
    // 点示例问题、重新提问发的不是草稿，草稿不动；放回时输入框里已经有新内容的话也不覆盖
    const fromDraft = text === draft
    if (fromDraft) setDraft('')
    setFocus(null)
    stick.current = true
    ask.mutate(q, { onSuccess: ({ started }) => !started && fromDraft && !draftRef.current && setDraft(q) })
  }
  const retry = (v: TurnView) => {
    if (v.error?.code !== 'session_not_found') return submit(v.question)
    // 会话已经不在了：原地重问必然再失败，去新的咨询里，把问题放进输入框。
    // 这一问和缓存都等跳走之后再清，不然跳走之前这个页面会闪一下「找不到这条咨询记录」：
    // 先 leave() 的话这一轮立刻不算这个页面的了；先清缓存的话会重新请求一次、拿到 404。
    // 这一问由新页面挂载时的 useLayoutEffect 停掉（它不属于新页面）
    void qc.invalidateQueries({ queryKey: ['sessions'] })
    carried = { sessionId: null, draft: v.question, focused: true }
    void navigate({ to: '/' }).then(() => qc.removeQueries({ queryKey: ['session', sessionId] }))
  }
  // 给 TurnBlock 的回调保持同一个引用，配合 memo：打字和流式输出时已完成的回合不重渲染
  const latest = useRef({ retry })
  latest.current = { retry }
  const onCite = useCallback((turnId: string, n: number) => setFocus({ turnId, n, tick: ++tick.current }), [])
  const onShowEvidence = useCallback((turnId: string) => setFocus({ turnId, n: null, tick: ++tick.current }), [])
  const onRetry = useCallback((v: TurnView) => latest.current.retry(v), [])

  if (missing) return <NotFound title="找不到这条咨询记录" hint="它可能已经被删除了。可以从左侧新建一次咨询。" />

  const focusView = focus ? views.find((v) => v.id === focus.turnId) : undefined
  const lastCited = [...views].reverse().find((v) => v.citations.length > 0)
  const panelOpen = Boolean(focusView)
  const togglePanel = () =>
    panelOpen ? setFocus(null) : lastCited && onShowEvidence(lastCited.id)
  const panel = focusView && (
    <EvidencePanel
      view={focusView}
      focusN={focus?.n ?? null}
      focusKey={focus?.tick ?? 0}
      onClose={() => setFocus(null)}
      showHeader={desktop}
    />
  )

  return (
    <div className="flex h-full">
      <section className="flex min-w-0 flex-1 flex-col" aria-label="问答">
        <div className="flex h-12 shrink-0 items-center gap-2 border-b border-rule px-4">
          <h1 className="min-w-0 flex-1 truncate text-sm font-medium">{title || '新的咨询'}</h1>
          {lastCited && (
            <IconButton label={panelOpen ? '收起回答依据' : '查看回答依据'} side="bottom" onClick={togglePanel}>
              {panelOpen ? <PanelRightClose /> : <PanelRightOpen />}
            </IconButton>
          )}
        </div>
        <div
          ref={scroller}
          onScroll={(e) => {
            const el = e.currentTarget
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
          }}
          className="min-h-0 flex-1 overflow-y-auto"
        >
          <div className="mx-auto flex max-w-[760px] flex-col gap-8 px-4 py-6">
            {sessionId && detail.isPending && !mine ? (
              <div className="flex flex-col gap-3" aria-hidden="true">
                <div className="h-16 animate-pulse rounded-card bg-ink/[0.05]" />
                <div className="h-24 animate-pulse rounded-card bg-ink/[0.05]" />
              </div>
            ) : detail.isError && !missing && !mine ? (
              <div className="text-sm text-ink-muted" role="alert">
                <p>这条咨询记录没有加载出来。</p>
                <Button variant="outline" size="sm" className="mt-2" onClick={() => void detail.refetch()}>
                  <RotateCw />
                  重新加载
                </Button>
              </div>
            ) : views.length === 0 ? (
              <EmptyState onPick={(q) => submit(q)} />
            ) : (
              views.map((v, i) => (
                <TurnBlock
                  key={v.id}
                  view={v}
                  first={i === 0}
                  activeN={focus?.turnId === v.id ? focus.n : null}
                  onCite={onCite}
                  onShowEvidence={onShowEvidence}
                  onRetry={onRetry}
                />
              ))
            )}
          </div>
        </div>
        <p className="sr-only" role="status" aria-live="polite">
          {announce}
        </p>
        <Composer
          value={draft}
          onChange={setDraft}
          onSubmit={() => submit()}
          onStop={chatStore.stop}
          busy={busy}
          autoFocus={restored?.focused ?? false}
        />
      </section>
      {desktop ? (
        panel && <aside className="w-[380px] shrink-0 border-l border-rule bg-bg">{panel}</aside>
      ) : (
        <Sheet open={panelOpen} onOpenChange={(o) => !o && setFocus(null)} side="bottom" title="回答依据">
          {panel}
        </Sheet>
      )}
    </div>
  )
}
