import { X } from 'lucide-react'
import { Dialog } from 'radix-ui'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Button } from './button'
import { Tip } from './tooltip'

const SIDES = {
  left: 'anim-left inset-y-0 left-0 h-full w-[300px] max-w-[85vw] border-r',
  bottom: 'anim-bottom inset-x-0 bottom-0 max-h-[80vh] rounded-t-card border-t',
}

/** 小屏用的抽屉：左侧放咨询记录，底部放回答依据。 */
export function Sheet({
  open,
  onOpenChange,
  side,
  title,
  children,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  side: keyof typeof SIDES
  title: string
  children: ReactNode
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="anim-fade fixed inset-0 z-40 bg-black/40" />
        <Dialog.Content
          className={cn('fixed z-50 flex flex-col border-rule bg-surface shadow-hair outline-none', SIDES[side])}
        >
          <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-rule px-4">
            <Dialog.Title className="text-sm font-semibold">{title}</Dialog.Title>
            <Tip label="收起">
              <Dialog.Close asChild>
                <Button size="icon" variant="ghost" aria-label="收起">
                  <X />
                </Button>
              </Dialog.Close>
            </Tip>
          </div>
          <Dialog.Description className="sr-only">{title}</Dialog.Description>
          <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
