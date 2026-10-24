# 款知 ClauseTrail 常用命令（兼容 macOS 自带的 GNU Make 3.81）
VENV    := backend/.venv
PY      := $(VENV)/bin/python
CLI     := $(VENV)/bin/clausetrail
SAMPLES := data/samples

.PHONY: help setup models ingest eval demo dev api web test lint corpus

help: ## 列出可用命令
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-8s %s\n", $$1, $$2}'

setup: ## 创建 Python 3.12 虚拟环境，安装后端和前端依赖
	@test -x $(PY) || uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e "backend[dev]"
	cd frontend && npm ci

models: ## 下载 DeepDoc 模型；LOCAL=1 时同时下载本地 bge 权重
	$(PY) backend/scripts/download_models.py $(if $(LOCAL),--local-bge)

ingest: ## 解析 data/samples 并写入索引
	$(CLI) ingest $(SAMPLES)

eval: ## 在样例上跑评测，报告写到 eval/reports/
	$(PY) -m eval.run_all

corpus: ## 列出四家官网的公开条款清单（只请求目录页，不下载文件）
	$(CLI) corpus list

demo: ingest eval ## 入库样例、跑评测，然后启动 API 和前端
	$(MAKE) dev

dev: ## 同时启动 API（:8000）和前端（:5173）
	$(MAKE) -j2 api web

api: ## 只启动 API
	$(CLI) serve --reload

web: ## 只启动前端
	cd frontend && npm run dev

test: ## 后端测试（离线示例模型，不联网）
	cd backend && .venv/bin/python -m pytest
	$(PY) -m pytest -q -p no:cacheprovider eval/tests

lint: ## ruff 检查 + 前端类型检查
	cd backend && .venv/bin/ruff check --config pyproject.toml . ../eval
	cd backend && .venv/bin/ruff format --check --config pyproject.toml . ../eval
	cd frontend && npm run lint
