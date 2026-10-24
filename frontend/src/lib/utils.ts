import { type ClassValue, clsx } from 'clsx'
import { extendTailwindMerge } from 'tailwind-merge'

// 让 tailwind-merge 认识自定义的字号、圆角和阴影 token，避免误删
const twMerge = extendTailwindMerge({
  extend: {
    theme: { text: ['display'], radius: ['card', 'control'], shadow: ['hair'] },
  },
})

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
