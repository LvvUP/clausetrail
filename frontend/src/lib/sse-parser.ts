// 按 SSE 规范把文本流切成事件；兼容 \r\n、跨分块的换行和 ping 注释行
export interface RawEvent {
  event: string
  data: string
}

export class SseParser {
  private buffer = ''

  feed(chunk: string): RawEvent[] {
    this.buffer += chunk
    // 末尾的 \r 可能和下一块开头的 \n 是同一个换行，先留着
    let held = ''
    if (this.buffer.endsWith('\r')) {
      held = '\r'
      this.buffer = this.buffer.slice(0, -1)
    }
    const text = this.buffer.replace(/\r\n?/g, '\n')
    const blocks = text.split('\n\n')
    this.buffer = blocks.pop() + held
    return blocks.map(parseBlock).filter((e): e is RawEvent => e !== null)
  }

  flush(): RawEvent[] {
    const rest = this.buffer.replace(/\r\n?/g, '\n').trim()
    this.buffer = ''
    const event = rest ? parseBlock(rest) : null
    return event ? [event] : []
  }
}

function parseBlock(block: string): RawEvent | null {
  let event = 'message'
  const data: string[] = []
  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) continue
    const i = line.indexOf(':')
    const field = i < 0 ? line : line.slice(0, i)
    let value = i < 0 ? '' : line.slice(i + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') event = value
    else if (field === 'data') data.push(value)
  }
  return data.length ? { event, data: data.join('\n') } : null
}
