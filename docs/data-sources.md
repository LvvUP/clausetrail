# 数据来源

## 公开条款的规模

国内保险公司按监管要求，在官网「公开信息披露」栏目公开在售和停售产品的条款 PDF。下表中四家寿险公司的数字是 2026-10-22 用 `clausetrail corpus list` 逐条遍历官网目录得到的；同一产品修订过条款时，各版本是单独备案、单独公开的文件，分别计数。机器可读版本在 [`data/corpus_manifest.csv`](../data/corpus_manifest.csv)。

| 来源 | 产品 | 条款文件 | 说明 |
| --- | --- | --- | --- |
| [平安人寿 在售 / 停售保险产品](https://life.pingan.com/gongkaixinxipilu/baoxianchanpinmulujitiaokuan.jsp) | 1,342 | 1,841 | 目录接口列出历史版本，每个版本一份条款 |
| [中国人寿 产品基本信息](https://www.e-chinalife.com/xxpl/gsgk/cpjbxx/) | 1,578 | 1,578 | 在售 512 + 停售 1,066，每个产品一份条款 |
| [新华保险 产品基本信息](https://www.newchinalife.com/node/372) | 830 | 1,346 | 产品详情页列出现行和已停用的条款版本 |
| [太保寿险 经营的保险产品目录及条款](https://life.cpic.com.cn/xrsbx/gkxxpl/jbxx/gsgk/jydbxcpmljtk/) | 1,136 | 1,137 | 在办 / 停办 × 寿险、年金、意外、健康 8 个目录 |
| [中国保险行业协会 财产险产品信息库](https://www.iachina.cn/art/2017/6/29/art_226_45696.html) | 约 14,000 | — | 页面声明；查询入口需要注册，脚本不抓取 |
| [InsQABench](https://arxiv.org/abs/2501.10943) | 约 25,000 | — | 论文声明；开放数据只含问答和条款片段，且很可能与上面几家重叠，不参与加总 |

四家寿险公司合计 4,886 个产品、5,902 份条款文件；加上行业协会财产险库约 14,000 个产品。

遍历时只请求目录页和产品详情页，没有下载条款 PDF。仓库不包含语料本体，`make ingest` 默认只处理下面的样例；需要全量语料时按 README 的「下载全量语料」一节自行下载。

## 仓库内的样例

`data/samples/` 里有 5 份文件，来源、版权和用途见 [`data/samples/README.md`](../data/samples/README.md)：

- 4 份原生 PDF 条款（中国人寿 3 份、平安人寿 1 份），覆盖重疾、医疗、寿险，含病种表、给付比例表和三栏排版。
- 1 份合成的英文扫描件（strata-insurance-corpus，CC-BY-4.0），只用来演示扫描件分流和 OCR 路径。公开渠道没有找到真实的扫描版条款。

## 查过但没有使用的开放数据

| 数据集 | 许可 | 不用的原因 |
| --- | --- | --- |
| [Insur-QA（InsQABench）](https://huggingface.co/datasets/FrankRin/Insur-QA) | Apache-2.0 | 只有问答对和条款片段，没有条款原文；约 1.2 GB，没有下载 |
| [strata-insurance-corpus](https://huggingface.co/datasets/NikolaiSachok/strata-insurance-corpus) | CC-BY-4.0 | 全合成的欧洲财产险文档，只取了 1 份扫描件做样例 |
| [HICRIC](https://github.com/TPAFS/hicric) | CC BY-SA 4.0 | 美国医疗保险监管与案例文档，不是保单条款 |
| [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) | Apache-2.0 | 美国上市公司年报，没有行业字段，筛不出保险公司 |
| [insuranceqa-corpus-zh](https://github.com/fssqawj/insuranceqa-corpus-zh) | 仅限研究 | 问答语料，只作术语参考 |

## 模型权重

DeepDoc 的版面、OCR、表格结构模型来自 HuggingFace [`InfiniFlow/deepdoc`](https://huggingface.co/InfiniFlow/deepdoc)，首次运行时下载到 `~/.cache/clausetrail/deepdoc`，不进仓库。本地 bge 权重（可选）来自 `BAAI/bge-m3` 和 `BAAI/bge-reranker-large`，用 `make models LOCAL=1` 下载。

## 术语词典

`backend/resources/terms/` 下的保险术语（571 条）、同义词（55 组）和 OCR 混淆字表由 `backend/scripts/build_terms.py` 生成：一部分是人工整理，一部分从样例条款的病种表里抽取。
