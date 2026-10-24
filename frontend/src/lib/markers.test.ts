import assert from 'node:assert/strict'
import { test } from 'node:test'
import { linkCitations, usedCitations } from './citations.ts'

test('代码里的 [n] 不变成引用链接', () => {
  assert.equal(linkCitations('见[1]，`arr[1]`', 2), '见[1](#cite-1)，`arr[1]`')
  assert.equal(linkCitations('```\nx[2]\n```\n[2]', 2), '```\nx[2]\n```\n[2](#cite-2)')
  // 流式输出时代码块还没闭合
  assert.equal(linkCitations('```\nx[1]', 2), '```\nx[1]')
})

test('~~~ 围栏、带语言名的围栏、双反引号行内代码', () => {
  assert.equal(linkCitations('~~~\nx[1]\n~~~\n见[1]', 2), '~~~\nx[1]\n~~~\n见[1](#cite-1)')
  assert.equal(linkCitations('```python\na[1]\n```\n[2]', 2), '```python\na[1]\n```\n[2](#cite-2)')
  assert.equal(linkCitations('用 ``a[1]`` 取值[1]', 2), '用 ``a[1]`` 取值[1](#cite-1)')
  // ~~~ 不能用 ``` 闭合
  assert.equal(linkCitations('~~~\nx[1]\n```\ny[2]', 2), '~~~\nx[1]\n```\ny[2]')
})

test('空代码块、更长的结尾围栏、CRLF 换行都按闭合处理，不吞掉后面的引用', () => {
  assert.equal(linkCitations('```\n```\n见[1]', 2), '```\n```\n见[1](#cite-1)')
  assert.equal(linkCitations('```\nx[1]\n````\n见[1]', 2), '```\nx[1]\n````\n见[1](#cite-1)')
  assert.equal(linkCitations('```\r\nx[1]\r\n```\r\n见[1]', 2), '```\r\nx[1]\r\n```\r\n见[1](#cite-1)')
  // 第一个结尾围栏就闭合；后面单独的 ``` 是新的、还没闭合的代码块
  assert.equal(linkCitations('```\n```\n见[1]\n```\nx[2]', 2), '```\n```\n见[1](#cite-1)\n```\nx[2]')
  // 比开头短的不算结尾
  assert.equal(linkCitations('````\nx[1]\n```\ny[1]', 2), '````\nx[1]\n```\ny[1]')
})

test('不成对的反引号不吞掉后面的引用', () => {
  // 行首的 ```a``` 是行内代码，不是围栏
  assert.equal(linkCitations('```a```\n见[1]', 2), '```a```\n见[1](#cite-1)')
  // 只有开头没有结尾的反引号、隔着空行的反引号都不算代码
  assert.equal(linkCitations('`a 见[1]', 2), '`a 见[1](#cite-1)')
  assert.equal(linkCitations('`a\n\n见[1]`', 2), '`a\n\n见[1](#cite-1)`')
  // 长度不同的反引号串不配对
  assert.equal(linkCitations('``a` 见[1]', 2), '``a` 见[1](#cite-1)')
})

test('生成中按出现顺序取已引用的编号', () => {
  assert.deepEqual(usedCitations('甲[2]乙[1]丙[2]丁[9]', 3), [2, 1])
})
