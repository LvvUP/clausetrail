import { queryOptions } from '@tanstack/react-query'
import { api } from './api.ts'

export const sessionsQuery = queryOptions({ queryKey: ['sessions'], queryFn: api.sessions })
export const sessionQuery = (id: string) => queryOptions({ queryKey: ['session', id], queryFn: () => api.session(id) })
export const documentsQuery = queryOptions({ queryKey: ['documents'], queryFn: api.documents })
export const healthQuery = queryOptions({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000 })
