import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'

export function useMedia(query: string): boolean {
  const subscribe = useCallback(
    (cb: () => void) => {
      const m = window.matchMedia(query)
      m.addEventListener('change', cb)
      return () => m.removeEventListener('change', cb)
    },
    [query],
  )
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches)
}

export const DESKTOP = '(min-width: 1024px)'

export function useStoredFlag(key: string, fallback: boolean): [boolean, (v: boolean) => void] {
  const [value, setValue] = useState(() => {
    try {
      const v = localStorage.getItem(key)
      return v === null ? fallback : v === '1'
    } catch {
      return fallback
    }
  })
  const set = useCallback(
    (v: boolean) => {
      setValue(v)
      try {
        localStorage.setItem(key, v ? '1' : '0')
      } catch {
        // 写不进去只影响下次打开
      }
    },
    [key],
  )
  return [value, set]
}

export function useDocTitle(title: string) {
  useEffect(() => {
    document.title = title ? `${title} · 款知 ClauseTrail` : '款知 ClauseTrail'
  }, [title])
}
