import { createFileRoute } from '@tanstack/react-router'
import { Workbench } from '@/features/chat/workbench'

export const Route = createFileRoute('/c/$sessionId')({
  component: SessionPage,
})

function SessionPage() {
  const { sessionId } = Route.useParams()
  // key 让每个会话拿到独立的组件状态
  return <Workbench key={sessionId} sessionId={sessionId} />
}
