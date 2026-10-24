import assert from 'node:assert/strict'
import { test } from 'node:test'
import { SseParser } from './sse-parser.ts'

test('跨分块的 \\r\\n 与 ping 注释', () => {
  const p = new SseParser()
  const out = [
    ...p.feed('event: meta\r\ndata: {"a":1}\r'),
    ...p.feed('\n\r\n: ping\r\n\r\nevent: del'),
    ...p.feed('ta\r\ndata: {"text":"x"}\r\n\r\n'),
  ]
  assert.deepEqual(out, [
    { event: 'meta', data: '{"a":1}' },
    { event: 'delta', data: '{"text":"x"}' },
  ])
})

test('多行 data 合并，结尾残留在 flush 时输出', () => {
  const p = new SseParser()
  assert.deepEqual(p.feed('data: a\ndata: b\n\nevent: done\ndata: {}'), [{ event: 'message', data: 'a\nb' }])
  assert.deepEqual(p.flush(), [{ event: 'done', data: '{}' }])
})
