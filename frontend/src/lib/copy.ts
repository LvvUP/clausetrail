// 界面上和后端状态码对应的文案集中在这里；条款原文、文件名、回答正文按后端结果原样显示
import type { RefusalReason } from './schemas.ts'

export const REFUSAL_COPY: Record<RefusalReason, { title: string; hint: string }> = {
  no_hit: {
    title: '条款里查不到相关内容',
    hint: '条款库里没有和这个问题对得上的内容。可以带上产品名和具体事项再问一次，例如「恒裕一生的犹豫期有多长」。',
  },
  low_score: {
    title: '找到的内容不足以回答',
    hint: '检索到了一些片段，但和问题的相关度太低，按规则不作答。可以把问题问得更具体一些。',
  },
  unsupported: {
    title: '回答无法对应到原文',
    hint: '生成的内容找不到可以核对的条款出处，已经撤下，避免给出没有依据的说法。',
  },
}

export interface ErrorCopy {
  title: string
  hint: string
}

const ERROR_COPY: Record<string, ErrorCopy> = {
  model_unavailable: {
    title: '模型服务还没配置好',
    hint: '向量、精排或生成模型当前不可用。检查 .env 里的阿里云密钥；只想离线体验时，可以把三个 *_BACKEND 设为 fake。',
  },
  internal: { title: '这次没能完成回答', hint: '服务端处理时出了问题，可以重新提问；如果反复出现，请查看 API 日志。' },
  network: { title: '连不上问答服务', hint: '确认 API 已经启动（make api），然后重新提问。' },
  http: { title: '服务没有接受这个请求', hint: '可以重新提问；反复出现时请查看 API 日志。' },
  session_not_found: {
    title: '这条咨询记录已经不在了',
    hint: '它可能在别的页面里被删除了。可以在新的咨询里重问这个问题。',
  },
  schema: { title: '服务返回的数据和界面对不上', hint: '前后端版本可能不一致，刷新页面后重试。' },
  empty_question: { title: '问题是空的', hint: '写下想了解的条款内容后再发出。' },
  stream_cut: { title: '回答在中途断开了', hint: '连接意外结束，可以重新提问。' },
}

// HTTP 错误按状态码给提示：只有 422 是问题本身的问题，5xx 是服务端的问题。
// 400 只来自 Host 校验（参数错误 FastAPI 返回 422），重问也一样会被拒，要改配置
export function errorCopy(code: string, status?: number): ErrorCopy {
  if (code === 'http' && status) {
    if (status === 422) return { title: '服务没有接受这个问题', hint: '问题最长 500 字，或者格式不对。改短后再发出。' }
    if (status === 400)
      return {
        title: '请求被服务端拒绝了',
        hint: '访问用的域名或 IP 可能不在后端的 ALLOWED_HOSTS 里。用 localhost 打开页面，或者把这个地址加进 .env 的 ALLOWED_HOSTS 后重启 API。',
      }
    if (status >= 500) return { title: '问答服务出错了', hint: `服务端返回了 ${status}。确认 API 正常运行后重新提问。` }
    return { title: '服务没有接受这个请求', hint: `服务端返回了 ${status}，可以重新提问。` }
  }
  return ERROR_COPY[code] ?? { title: '这次没能完成回答', hint: '可以重新提问。' }
}

export const PARSER_LABELS: Record<string, string> = {
  deepdoc: 'DeepDoc 版面分析',
  deepdoc_tsr_fallback: 'DeepDoc 表格结构还原',
  deepdoc_ocr: 'DeepDoc OCR',
  mineru: 'MinerU',
  paddleocr: 'PaddleOCR',
  docx: 'Word 解析',
}

export const PAGE_TYPE_LABELS: Record<string, string> = {
  native: '原生文本',
  borderless_table: '无边框表格',
  scanned: '扫描件',
}

export const COMPONENT_LABELS: Record<string, string> = {
  embedder: '向量模型',
  reranker: '精排模型',
  llm: '生成模型',
  vector_store: '向量库',
  bm25: '关键词检索',
  cache: '查询缓存',
}

export const COMPONENT_STATUS: Record<string, string> = { ok: '正常', degraded: '已降级', unavailable: '不可用' }
export const DOC_STATUS: Record<string, string> = { ready: '可检索' }
