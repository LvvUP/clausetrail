import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readDetail } from './api.ts'

test('错误详情：JSON 取 detail，短纯文本原样带出，HTML 错误页只给状态码', async () => {
  assert.equal(await readDetail(new Response(JSON.stringify({ detail: '会话不存在' }), { status: 404 })), '会话不存在')
  assert.equal(
    await readDetail(new Response(JSON.stringify({ detail: [{ msg: '太长' }, { msg: '为空' }] }), { status: 422 })),
    '太长；为空',
  )
  assert.equal(await readDetail(new Response('Invalid host header', { status: 400 })), 'Invalid host header')
  assert.equal(await readDetail(new Response('<html><body>502</body></html>', { status: 502 })), 'HTTP 502')
  assert.equal(await readDetail(new Response('null', { status: 500 })), 'HTTP 500')
})
