// 引用编号与后端一致：从 1 开始，[n] 对应 citations 里 n 字段相同的那一条；按数组下标取会错位一位
import type { Citation } from './schemas.ts'

const MARKER = /\[(\d{1,2})\]/g
// rehype-sanitize 可能给锚点加 user-content- 前缀
const HREF = /^#(?:user-content-)?cite-(\d{1,2})$/
// 一句话加上紧跟在句末标点后面的编号，例如「……犹豫期。[1]」
const SENTENCE = /([^。；！？\n]*[。；！？\n]?)((?:\s*\[\d{1,2}\])*)/g
const MIN_MATCH = 8

// 代码块和行内代码，里面的 [n] 不是引用：
// - 围栏代码块：行首 ``` 或 ~~~（可带语言名），到第一个同种符号、不短于开头的围栏行为止（空代码块、CRLF 换行也认）；
//   流式输出时还没闭合的算到末尾
// - 行内代码：一串反引号到同样长度的一串为止（``a[1]`` 这种双反引号也认），不跨空行
// 缩进四格的代码块在模型回答里几乎不出现，不处理
// （围栏的语言名里不能有反引号，所以行首的 ```a``` 按行内代码算，不会把后面的内容都吞掉）
const CODE =
  /^[ \t]{0,3}(([`~])\2{2,})[^\n`]*(?:\n(?:[\s\S]*?\n)??[ \t]{0,3}\1\2*[ \t]*\r?(?=\n|$(?![\s\S]))|(?:\n[\s\S]*)?$(?![\s\S]))|(?<!`)(`+)(?!`)(?:(?!\n[ \t\r]*\n)[\s\S])*?(?<!`)\3(?!`)/gm

function linkPlain(part: string, count: number): string {
  return part.replace(MARKER, (m, d: string) => {
    const n = Number(d)
    return n >= 1 && n <= count ? `[${n}](#cite-${n})` : m
  })
}

/** 把有效的 [n] 换成 markdown 链接，交给渲染层画成可点击的编号；无效编号、代码里的 [n] 原样保留。 */
export function linkCitations(text: string, count: number): string {
  let out = ''
  let at = 0
  for (const m of text.matchAll(CODE)) {
    out += linkPlain(text.slice(at, m.index), count) + m[0]
    at = m.index + m[0].length
  }
  return out + linkPlain(text.slice(at), count)
}

/** 回答里出现过的有效编号，按首次出现的顺序；和后端 used_citations 同一规则，生成中也能标出已引用的段落。 */
export function usedCitations(text: string, count: number): number[] {
  const seen: number[] = []
  for (const [, d] of text.matchAll(MARKER)) {
    const n = Number(d)
    if (n >= 1 && n <= count && !seen.includes(n)) seen.push(n)
  }
  return seen
}

export function citationNumber(href: string | undefined): number | null {
  const m = href?.match(HREF)
  return m ? Number(m[1]) : null
}

export function citationFor(citations: Citation[], n: number): Citation | undefined {
  return citations.find((c) => c.n === n)
}

/** 回答里标了 [n] 的句子（去掉编号），用来在原文里找对应的那一段高亮。 */
export function claimsFor(answer: string, n: number): string[] {
  const out: string[] = []
  for (const [, sentence, markers] of answer.matchAll(SENTENCE)) {
    if (!(sentence + markers).includes(`[${n}]`)) continue
    const claim = sentence
      .replace(MARKER, '')
      .replace(/^根据条款[:：]/, '')
      .trim()
    if (claim) out.push(claim)
  }
  return out
}

function longestCommon(a: string, b: string): [number, number] {
  // 返回 b 中与 a 的最长公共子串 [start, length]；片段很短，O(n·m) 足够
  let best: [number, number] = [0, 0]
  let prev = new Array<number>(b.length + 1).fill(0)
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array<number>(b.length + 1).fill(0)
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        cur[j] = prev[j - 1] + 1
        if (cur[j] > best[1]) best = [j - cur[j], cur[j]]
      }
    }
    prev = cur
  }
  return best
}

/** 原文中与回答句子重合的区间（至少 8 个字），合并后按位置排序。 */
export function highlightRanges(snippet: string, claims: string[]): [number, number][] {
  const ranges: [number, number][] = []
  for (const claim of claims) {
    const [start, len] = longestCommon(claim, snippet)
    if (len >= MIN_MATCH) ranges.push([start, start + len])
  }
  ranges.sort((x, y) => x[0] - y[0])
  const merged: [number, number][] = []
  for (const r of ranges) {
    const last = merged.at(-1)
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1])
    else merged.push([...r])
  }
  return merged
}

export function splitByRanges(text: string, ranges: [number, number][]): { text: string; marked: boolean }[] {
  const parts: { text: string; marked: boolean }[] = []
  let at = 0
  for (const [s, e] of ranges) {
    if (s > at) parts.push({ text: text.slice(at, s), marked: false })
    parts.push({ text: text.slice(s, e), marked: true })
    at = e
  }
  if (at < text.length) parts.push({ text: text.slice(at), marked: false })
  return parts
}

export function evidenceSummary(citations: Citation[], used: number[]) {
  const cited = citations.filter((c) => used.includes(c.n))
  return { total: citations.length, cited: cited.length, files: new Set(cited.map((c) => c.file)).size }
}
