# 参与贡献

欢迎提 issue 和 PR。提交前请先看下面几条。

## 开发环境

```bash
make setup     # Python 3.12 虚拟环境 + 前端依赖
make test      # 后端与评测脚本的测试（离线示例模型，不联网）
make lint      # ruff + 前端类型检查
```

测试默认使用离线示例模型（`fake` 后端）和临时目录里的 Milvus Lite，不需要任何密钥，也不会调用外部服务。

## 约定

- 后端代码在 `backend/src/clausetrail/`，按 RAG 环节分目录：`ingest` → `structure` → `chunking` → `index` → `retrieval` → `answer`。新功能放进对应的目录，不要另起一层嵌套。
- 依赖版本精确锁定（`==` / 不带 `^`）。新增依赖时说明用途；只在部分场景用到的依赖放进 `pyproject.toml` 的 extra。
- 改了解析或切分规则，请同时补单元测试（`backend/tests/unit/`），并跑一次 `python -m eval.run_chunking` 看截断率的变化。
- 改了标题特征，需要重新训练：`backend/.venv/bin/python backend/scripts/train_heading_xgb.py`。脚本会先检查 numpy 推理与 XGBoost 的输出是否一致，然后才保存模型。
- 评测集由 `python -m eval.gen.synthesize` 生成，开发集和测试集按文档隔离。不要手工改测试集来提高分数；阈值只在开发集上标定。
- 注释写中文，只解释代码本身看不出来的原因。

## 数据与版权

- 不要提交保险条款语料、模型权重或 `.env`。
- `data/samples/` 只放少量公开样例，并在 `data/samples/README.md` 里写明来源、下载日期和版权归属。
- `backend/third_party/` 下的文件保留原版权头；修改时在文件头注明。

## 许可

提交的代码按 [Apache-2.0](LICENSE) 授权。
