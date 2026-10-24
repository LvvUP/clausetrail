const sameYear = new Intl.DateTimeFormat('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })
const otherYear = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric',
  month: 'numeric',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})
const clock = new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit' })
const day = new Intl.DateTimeFormat('zh-CN', { month: 'numeric', day: 'numeric' })

/** 后端时间是带时区的 ISO 字符串，按本地时区显示。 */
export function formatTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return (d.getFullYear() === new Date().getFullYear() ? sameYear : otherYear).format(d)
}

export function formatShort(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toDateString() === new Date().toDateString() ? clock.format(d) : day.format(d)
}

export function formatSeconds(ms: number | undefined): string | null {
  return typeof ms === 'number' ? `${(ms / 1000).toFixed(1)} 秒` : null
}

export function formatScore(score: number): string {
  return score.toFixed(2)
}
