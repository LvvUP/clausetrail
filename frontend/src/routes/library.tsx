import { createFileRoute } from '@tanstack/react-router'
import { LibraryPage } from '@/features/library/library-page'

export const Route = createFileRoute('/library')({
  component: LibraryPage,
})
