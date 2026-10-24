import logging
import os

from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

MODEL_REPO = "InfiniFlow/deepdoc"
# 固定到一个提交，上游更新模型时本项目不会悄悄换权重；升级时改这里并重跑测试
MODEL_REVISION = "9c7aa2c730d7a242d7f04cf6109a6a239aefc717"
REQUIRED_FILES = ("det.onnx", "rec.onnx", "ocr.res", "layout.onnx", "tsr.onnx")


def model_dir() -> str:
    return os.environ.get(
        "DEEPDOC_MODEL_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "clausetrail", "deepdoc"),
    )


# 逐个检查需要的文件：上次下载中断、只缺 layout.onnx 时也会补齐，而不是到推理时才报找不到模型
def ensure_models() -> str:
    target = model_dir()
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(target, f))]
    if missing:
        log.warning(
            "DeepDoc 模型缺少 %s，正在从 HuggingFace %s 下载到 %s（也可以先运行 make models）",
            ", ".join(missing),
            MODEL_REPO,
            target,
        )
        snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=target)
    return target
