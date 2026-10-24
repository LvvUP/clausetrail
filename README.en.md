# ClauseTrail

Question answering over Chinese insurance policy clauses. Every answer points back to the section and page it came from, and the system says so plainly when the clauses don't support an answer.

[![License](https://img.shields.io/badge/license-Apache--2.0-2F6B5A)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-1F2328)
![Node](https://img.shields.io/badge/node-24-1F2328)

[中文](README.md)

![Answer with evidence: clicking a citation opens the clause text on the right](docs/images/answer-with-evidence.png)

Screenshots are for illustration only; the UI behaves the same with the Alibaba Cloud models.

## Tech stack

| Stage | Components |
| --- | --- |
| Service | Python 3.12, FastAPI (async), SSE streaming |
| Parsing | DeepDoc (layout, table structure, OCR), MinerU, PaddleOCR, OpenCV, pdfplumber |
| Structure | XGBoost heading-level classifier, jieba with an insurance term dictionary |
| Retrieval | Milvus (HNSW vectors), Elasticsearch (BM25 + synonyms), RRF fusion |
| Models | bge-m3 (embeddings), bge-reranker-large (reranking), Qwen (rewriting / answering) |
| Cache & storage | Redis (query cache), SQLite (documents, sessions, knowledge-base version) |
| Frontend | React 19, TanStack Router / Query, shadcn/ui, Tailwind CSS 4, Vite |

Models run on Alibaba Cloud by default: bge-m3 deployed on PAI-EAS, the BGE reranking service hosted by OpenSearch (`ops-bge-reranker-larger`), and Qwen on Model Studio (Bailian). bge-m3 and bge-reranker-large can also run locally through FlagEmbedding.

## Features

- **Routed parsing**: DeepDoc layout analysis for native PDFs, MinerU for borderless tables (DeepDoc TSR if MinerU is not installed), and OpenCV denoise/deskew followed by PaddleOCR for scans (DeepDoc OCR as fallback).
- **Section structure**: an XGBoost heading-level classifier, with a section tree whose depth follows the order in which numbering styles first appear; OCR text is corrected with a term dictionary and a confusable-character table.
- **Structure-aware chunking**: recursive splitting along the section tree that keeps exclusion lists and provisos intact; every chunk carries its section path, pages and layout boxes.
- **Hybrid retrieval**: bge-m3 vectors (Milvus HNSW) and BM25 (Elasticsearch, jieba with a 571-term insurance dictionary and synonym expansion) are recalled in parallel and fused with RRF; the top 20 are reranked by bge-reranker and the top 5 go into the prompt as evidence.
- **Grounded answers**: Qwen streams the answer with a citation marker after each sentence. It refuses when retrieval scores are too low or the answer cites nothing valid.
- **Multi-turn**: topic-continuity detection, rewriting of follow-ups into standalone questions, and a sliding window plus summary memory.
- **Query cache**: the Redis key combines the normalized question with the knowledge-base version, so cached answers go stale automatically when documents change.
- **Evaluation**: synthetic, auto-verified datasets with document-level dev/test isolation, covering MRR@10, Recall@k, truncation rate, rewrite accuracy and citation accuracy.

## Quick start

Requires Python 3.12, Node.js 24, [uv](https://docs.astral.sh/uv/) and GNU Make.

```bash
git clone https://github.com/LvvUP/clausetrail.git && cd clausetrail
cp .env.example .env   # add Alibaba Cloud keys, or set the three *_BACKEND values to fake
make setup
make models            # DeepDoc models, ~100 MB
make demo              # ingest samples → run eval → API on :8000, web on :5173
```

The repo ships with no API keys. So that you can see the whole pipeline without signing up for anything, and so that no secrets end up in a public repo, the screenshots and tests use the offline illustrative implementation (all three `*_BACKEND` values set to `fake`). It calls no external service and runs parsing, retrieval, refusal and streaming end to end, but its answers only demonstrate the flow. To see what retrieval and answering actually do, you are encouraged to put your own Alibaba Cloud API keys in `.env` and try it: `.env.example` already defaults to the Alibaba Cloud backends, and `.env` is git-ignored so it will not be committed. The embedder and reranker can also run as local models (`local`), and you can swap in your own policy documents.

You can also try it from the command line after `make ingest`. The Milvus Lite data file can be opened by only one process at a time, so stop the API first if it is running. The questions are in Chinese because the clauses are:

```bash
backend/.venv/bin/clausetrail search "恒裕一生的犹豫期有多长"       # ranks from vector, BM25, RRF and reranker; no generation
backend/.venv/bin/clausetrail ask "恒裕一生解除合同要提供哪些资料"  # streamed answer, followed by cited file, section and page
backend/.venv/bin/clausetrail health                                # backend and status of each component
```

`docker compose --profile full up -d` starts Redis, Elasticsearch and Milvus standalone; then set `VECTOR_STORE_URI=http://localhost:19530`. See [docs/deployment.md](docs/deployment.md).

## Layout

| Path | Contents |
| --- | --- |
| `backend/src/clausetrail/ingest/` | Per-page routing: native PDF, borderless tables, scans |
| `backend/src/clausetrail/structure/` | OCR correction, heading levels, section tree |
| `backend/src/clausetrail/chunking/` | Structure-aware chunking and merge rules |
| `backend/src/clausetrail/index/` | Milvus vectors + BM25 (Elasticsearch / Milvus) |
| `backend/src/clausetrail/retrieval/` | Parallel recall, RRF, reranking |
| `backend/src/clausetrail/answer/` | Prompt, citation checks, refusal |
| `backend/src/clausetrail/conversation/` | Topic continuity, rewriting, memory |
| `backend/src/clausetrail/models/` | Embedder / reranker / LLM: Alibaba Cloud API, local models, offline illustrative implementation |
| `backend/src/clausetrail/corpus/` | Public clause catalog listing and download (`clausetrail corpus`) |
| `backend/resources/` | Term dictionary, synonyms, confusable characters, heading model |
| `backend/scripts/` | Heading model training, term table generation, model download |
| `backend/third_party/deepdoc/` | DeepDoc vision inference code (Apache-2.0, see NOTICE) |
| `frontend/` | React + TanStack Router / Query + shadcn/ui + Tailwind, built with Vite |
| `eval/` | Dataset synthesis, metrics, reports |
| `data/samples/` | 4 public policy samples + 1 synthetic scan |

## Configuration

Everything is set in `.env`; see [.env.example](.env.example) for the full list.

| Group | Variables |
| --- | --- |
| Model backends | `EMBEDDER_BACKEND`, `RERANKER_BACKEND`, `LLM_BACKEND` (`aliyun_*` / `local` / `fake`) |
| Alibaba Cloud | `ALIYUN_PAI_EAS_*`, `ALIYUN_OPENSEARCH_*`, `ALIYUN_BAILIAN_*`, `LLM_REWRITE_MODEL`, `LLM_ANSWER_MODEL` |
| Retrieval | `VECTOR_STORE_URI`, `BM25_BACKEND`, `ES_URL`, `RECALL_TOP_K`, `RRF_K`, `RERANK_TOP_K`, `CONTEXT_TOP_K` |
| Refusal & cache | `REFUSAL_THRESHOLD` (calibrated on the dev set), `REDIS_URL` |

## Evaluation

```bash
make eval    # calibrate → retrieval → rewrite → citation → chunking; reports go to eval/reports/
```

How many clause files you download and how you split the evaluation sets is up to you. By default the generator splits by document, 20% dev and 80% test, with no document shared between them, and only the dev set is used for tuning. That ratio is just a suggestion; change it in `eval/gen/corpus.py`. Reports are generated locally and not committed.

## Documentation

All docs are in Chinese:

- [docs/architecture.md](docs/architecture.md): module design
- [docs/deployment.md](docs/deployment.md): setup, dependent services, offline mode, troubleshooting
- [docs/evaluation.md](docs/evaluation.md): dataset generation, metrics, calibration
- [docs/data-sources.md](docs/data-sources.md): public clause sources and counts

## Full corpus (optional)

The bundled samples are enough to run everything. To reproduce at scale, `clausetrail corpus` fetches clause PDFs from the public disclosure pages of four Chinese life insurers (Ping An Life, China Life, New China Life, CPIC Life). A full walk of their catalogs on 2026-10-22 found 4,886 products and 5,902 clause files, counting superseded clause versions separately.

```bash
make corpus                                                # list catalogs into data/corpus/manifest.csv (catalog pages only, ~20 min)
backend/.venv/bin/clausetrail corpus check                 # spot-check links; reads only file headers
backend/.venv/bin/clausetrail corpus download --limit 100  # try 100 files first; with no options it downloads 20
backend/.venv/bin/clausetrail corpus download --all        # download every clause file in the list into data/corpus/files/
backend/.venv/bin/clausetrail ingest data/corpus/files
```

- The tool is deliberately polite: at most one request per second per site, it honours robots.txt, keeps only responses that are real PDFs, resumes after interruption, pauses a site after 10 consecutive failures, and stops when free disk drops below 2 GB.
- A full download is about 5,900 files (several GB, estimated from the samples) and takes a few hours. `data/corpus/` is git-ignored.
- Catalog layouts were verified on 2026-10-22; a site redesign can break one source, and `corpus list` reports which.
- The clauses are mandatory public disclosures and remain the copyright of each insurer. Use downloads for personal research only, do not redistribute them, and follow each site's terms of use.

## Limitations

- Only 4 sample policies ship with the repo; download the full corpus with `clausetrail corpus` (see "Full corpus" above).
- The scanned sample is synthetic. MinerU and PaddleOCR adapters exist but are optional extras.
- The API has no authentication. Put it behind an authenticating reverse proxy before exposing it.

## Disclaimer

This project is for research and demonstration only and gives no insurance advice. The policy text and the insurer's interpretation prevail. The sample clauses are copyright of their insurers; see [data/samples/README.md](data/samples/README.md).

## License

[Apache-2.0](LICENSE). `backend/third_party/deepdoc` contains code derived from InfiniFlow/ragflow; see [NOTICE](NOTICE).
