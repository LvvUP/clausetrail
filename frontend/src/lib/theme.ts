import { useSyncExternalStore } from 'react'

export type ThemePref = 'system' | 'light' | 'dark'
const KEY = 'ct-theme'
const media = window.matchMedia('(prefers-color-scheme: dark)')
const listeners = new Set<() => void>()

function read(): ThemePref {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

let pref = read()

function resolve(p: ThemePref): 'light' | 'dark' {
  return p === 'system' ? (media.matches ? 'dark' : 'light') : p
}

function apply() {
  document.documentElement.dataset.theme = resolve(pref)
  listeners.forEach((l) => l())
}

media.addEventListener('change', () => pref === 'system' && apply())

export function setTheme(next: ThemePref) {
  pref = next
  try {
    localStorage.setItem(KEY, next)
  } catch {
    // 隐私模式下写不进去，只影响下次打开
  }
  apply()
}

function subscribe(l: () => void) {
  listeners.add(l)
  return () => {
    listeners.delete(l)
  }
}

export function useTheme() {
  const current = useSyncExternalStore(subscribe, () => pref)
  const resolved = useSyncExternalStore(subscribe, () => resolve(pref))
  return { pref: current, resolved, setTheme }
}
