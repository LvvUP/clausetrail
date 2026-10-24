import argparse
import sys

from deepdoc._paths import ensure_models
from huggingface_hub import snapshot_download

from clausetrail.config import get_settings

# 固定到具体提交，上游更新权重时本地结果不会悄悄变化；升级时改这里
LOCAL_BGE = {
    "bge-m3": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    "bge-reranker-large": ("BAAI/bge-reranker-large", "55611d7bca2a7133960a6d3b71e083071bbfc312"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 DeepDoc 版面 / OCR / 表格结构模型，以及可选的本地 bge 权重")
    parser.add_argument(
        "--local-bge",
        action="store_true",
        help="同时下载 bge-m3 和 bge-reranker-large（约 3.3 GB，只有 *_BACKEND=local 时才需要）",
    )
    args = parser.parse_args()

    # 已存在时不会重复下载
    print("DeepDoc ->", ensure_models())
    if args.local_bge:
        root = get_settings().local_model_dir
        for name, (repo, revision) in LOCAL_BGE.items():
            target = root / name
            print(f"{repo}@{revision[:8]} -> {target}")
            snapshot_download(repo_id=repo, revision=revision, local_dir=target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
