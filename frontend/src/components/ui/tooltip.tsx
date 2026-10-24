import { Tooltip } from 'radix-ui'
import type { ComponentProps, ReactElement, ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Button } from './button'

export function TooltipProvider({ children }: { children: ReactNode }) {
  return <Tooltip.Provider delayDuration={300}>{children}</Tooltip.Provider>
}

export function Tip({
  label,
  side = 'top',
  children,
}: {
  label: string
  side?: 'top' | 'right' | 'bottom' | 'left'
  children: ReactElement
}) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          side={side}
          sideOffset={6}
          className="anim-fade z-50 rounded-control bg-ink px-2 py-1 text-xs text-bg shadow-hair"
        >
          {label}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}

/** 纯图标按钮：aria-label 与 tooltip 用同一段文字。 */
export function IconButton({
  label,
  side,
  className,
  variant = 'ghost',
  ...props
}: ComponentProps<typeof Button> & { label: string; side?: 'top' | 'right' | 'bottom' | 'left' }) {
  return (
    <Tip label={label} side={side}>
      <Button size="icon" variant={variant} aria-label={label} className={cn(className)} {...props} />
    </Tip>
  )
}
