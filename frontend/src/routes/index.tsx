import { createFileRoute } from '@tanstack/react-router'
import { Workbench } from '@/features/chat/workbench'

export const Route = createFileRoute('/')({
  component: () => <Workbench sessionId={null} />,
})
