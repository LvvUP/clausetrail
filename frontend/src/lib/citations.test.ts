import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  citationFor,
  citationNumber,
  claimsFor,
  evidenceSummary,
  highlightRanges,
  linkCitations,
  splitByRanges,
} from './citations.ts'
import type { Citation } from './schemas.ts'

const cite = (n: number, file = 'a.pdf'): Citation => ({
  n,
  chunk_id: `c${n}`,
  doc_id: 'd',
  file,
  section_path: [],
  page_start: 1,
  page_end: 1,
  snippet: '',
  parser: 'deepdoc',
  rerank_score: 0.5,
  rrf_score: 0.01,
  dense_rank: null,
  bm25_rank: null,
})

test('[n] 从 1 开始，对应 n 字段相同的那一条，不按数组下标错位', () => {
  const list = [cite(1), cite(2), cite(3)]
  assert.equal(citationFor(list, 1)?.chunk_id, 'c1')
  assert.equal(citationFor(list, 3)?.chunk_id, 'c3')
  assert.equal(citationFor(list, 4), undefined)
})

test('只把有效编号变成链接', () => {
  assert.equal(linkCitations('甲[1]乙[3]丙[0]', 2), '甲[1](#cite-1)乙[3]丙[0]')
  assert.equal(citationNumber('#cite-2'), 2)
  assert.equal(citationNumber('#user-content-cite-12'), 12)
  assert.equal(citationNumber('https://example.com'), null)
})

test('取出标了编号的句子，并在原文里找到重合的部分', () => {
  const answer = '根据条款：犹豫期为十五日。[1]退保时退还现金价值。[2]'
  assert.deepEqual(claimsFor(answer, 1), ['犹豫期为十五日。'])
  const snippet = '第二十五条 犹豫期 自您签收本合同次日起，有15日的犹豫期为十五日。在此期间请您认真审视。'
  const ranges = highlightRanges(snippet, claimsFor(answer, 1))
  assert.equal(ranges.length, 1)
  assert.equal(snippet.slice(...ranges[0]), '犹豫期为十五日。')
  const parts = splitByRanges(snippet, ranges)
  assert.equal(parts.map((p) => p.text).join(''), snippet)
  assert.equal(parts.filter((p) => p.marked).length, 1)
})

test('证据条的数字', () => {
  const s = evidenceSummary([cite(1, 'a.pdf'), cite(2, 'a.pdf'), cite(3, 'b.pdf')], [1, 3])
  assert.deepEqual(s, { total: 3, cited: 2, files: 2 })
})
