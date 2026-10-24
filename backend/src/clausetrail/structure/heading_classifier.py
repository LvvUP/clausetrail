import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from clausetrail.config import get_settings
from clausetrail.structure.heading_features import FEATURE_NAMES, extract_features, rule_labels
from clausetrail.structure.tree_ensemble import TreeEnsemble
from clausetrail.types import Block

if TYPE_CHECKING:
    import xgboost as xgb

log = logging.getLogger(__name__)

NUM_CLASSES = 4
MODEL_FILE = "heading_xgb.json"
META_FILE = "heading_xgb.meta.json"


def model_path() -> Path:
    return get_settings().resources_dir / "models" / MODEL_FILE


# 训练专用：xgboost 只在这里按需导入，推理走 TreeEnsemble
def train(features: np.ndarray, labels: np.ndarray, seed: int = 7) -> "xgb.Booster":
    import xgboost as xgb

    dtrain = xgb.DMatrix(features, label=labels, feature_names=FEATURE_NAMES)
    params = {
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "max_depth": 4,
        "eta": 0.15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "eval_metric": "mlogloss",
        "seed": seed,
    }
    return xgb.train(params, dtrain, num_boost_round=120)


def save(booster: "xgb.Booster", meta: dict, path: Path | None = None) -> Path:
    path = path or model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(path)
    path.with_name(META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _load.cache_clear()
    return path


@lru_cache(maxsize=1)
def _load(path: str) -> TreeEnsemble | None:
    p = Path(path)
    if not p.exists():
        log.warning("未找到标题层级模型 %s，改用规则标注（运行 scripts/train_heading_xgb.py 训练）", p)
        return None
    model = TreeEnsemble.load(p)
    if model.feature_names and model.feature_names != FEATURE_NAMES:
        log.warning("标题层级模型的特征与当前代码不一致，改用规则标注（请重新训练）")
        return None
    return model


def predict_levels(blocks: list[Block]) -> list[int]:
    if not blocks:
        return []
    model = _load(str(model_path()))
    if model is None:
        return rule_labels(blocks)
    probs = model.predict_proba(extract_features(blocks))
    return [int(i) for i in probs.argmax(axis=1)]


def assign_levels(blocks: list[Block]) -> list[Block]:
    for block, level in zip(blocks, predict_levels(blocks), strict=True):
        block.level = level
    return blocks
