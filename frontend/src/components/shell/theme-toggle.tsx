import { Monitor, Moon, Sun } from 'lucide-react'
import { IconButton } from '@/components/ui/tooltip'
import { type ThemePref, useTheme } from '@/lib/theme'

const NEXT: Record<ThemePref, ThemePref> = { system: 'light', light: 'dark', dark: 'system' }
const LABEL: Record<ThemePref, string> = {
  system: '配色跟随系统，点击改为浅色',
  light: '当前浅色，点击改为深色',
  dark: '当前深色，点击改为跟随系统',
}

export function ThemeToggle({ side = 'top' }: { side?: 'top' | 'right' | 'bottom' }) {
  const { pref, setTheme } = useTheme()
  const Icon = pref === 'system' ? Monitor : pref === 'light' ? Sun : Moon
  return (
    <IconButton label={LABEL[pref]} side={side} onClick={() => setTheme(NEXT[pref])}>
      <Icon />
    </IconButton>
  )
}
