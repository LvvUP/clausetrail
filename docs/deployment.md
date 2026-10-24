# 部署与运行

## 环境要求

- Python 3.12（推荐用 [uv](https://docs.astral.sh/uv/) 管理）、Node.js 24、GNU Make
- 首次运行会下载 DeepDoc 模型到 `~/.cache/clausetrail/deepdoc`（约 100 MB）
- 可选：Docker（Redis / Elasticsearch / Milvus standalone）

## 本机快速开始

```bash
cp .env.example .env      # 填写阿里云的三组密钥，或改成离线模式（见下）
make setup                # 创建 backend/.venv，安装后端和前端依赖
make models               # 下载 DeepDoc 模型（已存在时跳过）
make ingest               # 解析 data/samples 并建索引
make dev                  # API :8000 + 前端 :5173
```

`make help` 列出全部命令。后端命令行是 `backend/.venv/bin/clausetrail`，子命令有 `ingest`、`remove`、`docs`、`search`、`ask`、`health`、`serve`。

### 离线模式

没有阿里云密钥时，把三个后端设成离线示例实现（`fake`）。不调用任何外部服务，也能跑通解析 → 索引 → 检索 → 拒答 → SSE 的全流程，回答和指标仅作示例：

```bash
EMBEDDER_BACKEND=fake
RERANKER_BACKEND=fake
LLM_BACKEND=fake
```

换成阿里云模型时，向量维度会从 256 变成 1024，需要删除 `data/milvus.db*` 后重新 `make ingest`。

### 阿里云模型

| 变量 | 说明 |
| --- | --- |
| `ALIYUN_PAI_EAS_URL` / `ALIYUN_PAI_EAS_TOKEN` | PAI Model Gallery 部署的 bge-m3 服务地址和 Token |
| `ALIYUN_OPENSEARCH_HOST` / `ALIYUN_OPENSEARCH_API_KEY` | OpenSearch AI 搜索开放平台的地址和 API Key，精排服务 `ops-bge-reranker-larger` |
| `ALIYUN_BAILIAN_API_KEY` | 百炼 API Key；`LLM_REWRITE_MODEL`、`LLM_ANSWER_MODEL` 以控制台列出的模型名为准 |

缺少哪一组，`/api/health` 和 `clausetrail health` 就把对应组件标成不可用，提问时返回 `model_unavailable`。

## 依赖服务（compose.yaml）

```bash
docker compose --profile core up -d     # Redis
docker compose --profile search up -d   # Redis + Elasticsearch
docker compose --profile full up -d     # Redis + Elasticsearch + Milvus standalone（etcd、MinIO）
```

- **Redis**：`REDIS_URL=redis://localhost:6379/0`。留空时用进程内 LRU，重启后缓存清空。
- **Elasticsearch**：`BM25_BACKEND=es`（默认）。启动时连不上 ES，就自动改用 Milvus 内置 BM25，并在 `/api/health` 里标出降级原因。
- **Milvus standalone**：把 `VECTOR_STORE_URI` 改成 `http://localhost:19530`，代码不用改。这个变量不叫 `MILVUS_URI`，因为 pymilvus 导入时会读取同名变量，而且只接受 http(s) 地址。

## 可选解析组件

```bash
uv pip install --python backend/.venv/bin/python -e "backend[mineru]"   # 无边框表格用 MinerU
uv pip install --python backend/.venv/bin/python -e "backend[ocr]"      # 扫描件用 PaddleOCR
uv pip install --python backend/.venv/bin/python -e "backend[es]"       # Elasticsearch 客户端
```

没有安装时自动退回 DeepDoc TSR / DeepDoc OCR，入库日志里提示一次，片段元数据的 `parser` 字段会记下实际用的解析器。

## 数据与更新

| 内容 | 位置 |
| --- | --- |
| 向量和 BM25 索引 | `data/milvus.db*`（Milvus Lite） |
| 文档、会话、知识库版本 | `data/clausetrail.sqlite3` |

- 重复入库同一份文件时按文件哈希覆盖，不会产生重复片段。新版本写完才替换旧版本：入库中途失败时旧版本照常可用；进程被中断时，下次启动会自动清掉写了一半的数据。删旧版本这一步失败时，旧片段不会被检索到，下次启动或下次入库这份文档时清掉。
- 同一份文档同一时间只能有一个进程在入库：另一个进程正在写时，再次入库会直接报错。换成 Milvus standalone、API 和 CLI 同时运行时，API 启动不会清理 CLI 正在写的数据。
- `clausetrail remove <doc_id>` 删除一份文档。入库或删除都会让知识库版本号加一，旧的查询缓存随之失效。
- 改了术语表或同义词表（`resources/terms/`）后重启 API：使用 Elasticsearch 时，索引会按词表指纹自动重建；Milvus 里存的预分词结果要重新入库才会更新。

## 安全

- API **没有认证和权限控制**，默认只监听 `127.0.0.1:8000`，CORS 只允许本机前端地址。如果要对外开放，必须放在带认证的反向代理后面，并收紧 `CORS_ORIGINS`、`ALLOWED_HOSTS`。
- API 只接受 Host 为 `localhost`、`127.0.0.1`、`[::1]` 的请求（`ALLOWED_HOSTS`，只比主机名，不比端口）。恶意网页通过 DNS rebinding 把自己的域名解析到本机时，请求会被拒绝。用局域网 IP 或自定义域名打开前端时，要把它加进 `ALLOWED_HOSTS`，否则页面会提示请求被拒绝。
- compose 里的 Elasticsearch 关闭了安全认证，MinIO 用默认账号，所有端口只绑定 `127.0.0.1`。这套配置只适合本机开发，不要直接用于生产。
- 密钥只放在 `.env` 里（已在 `.gitignore` 中），不要提交到仓库。

## 常见问题

- **进程崩溃，日志里有 `OMP: Error #15`**：说明在同一进程里同时导入了 xgboost 和 milvus-lite。运行期的标题分类用纯 numpy 实现，不会导入 xgboost；只有 `scripts/train_heading_xgb.py` 用到它，请单独运行这个脚本。
- **日志里有 gRPC 的 `AllocTimestamp NotImplementedError` 或 `too_many_pings`**：这是 milvus-lite 的内部日志，不影响结果。
- **入库时报 `Open local milvus failed`**：Milvus Lite 的数据文件同一时间只能被一个进程打开，API 运行期间不能另开终端执行 `clausetrail ingest` / `remove`。先停掉 API 再入库；换成 Milvus standalone 就没有这个限制。
