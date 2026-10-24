# 用纯 numpy 执行 XGBoost 导出的 JSON 树模型。
# 运行期不加载 libxgboost：它链接的 libomp 与 milvus-lite（faiss）自带的 libomp
# 同进程初始化会直接崩溃（OMP Error #15 / SIGSEGV）。训练仍用 XGBoost，见 scripts/train_heading_xgb.py。
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_OBJECTIVES = {"multi:softprob", "multi:softmax"}


@dataclass(frozen=True)
class _Tree:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    default_left: np.ndarray
    is_leaf: np.ndarray
    value: np.ndarray
    group: int


class TreeEnsemble:
    def __init__(self, trees: list[_Tree], base_score: np.ndarray, num_class: int, feature_names: list[str]):
        self.trees = trees
        self.base_score = base_score
        self.num_class = num_class
        self.feature_names = feature_names

    @classmethod
    def from_json(cls, model: dict) -> "TreeEnsemble":
        learner = model["learner"]
        objective = learner["objective"]["name"]
        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"不支持的目标函数 {objective}")
        booster = learner["gradient_booster"]
        if booster["name"] != "gbtree":
            raise ValueError(f"不支持的 booster {booster['name']}")
        param = learner["learner_model_param"]
        num_class = int(param["num_class"])
        base = np.asarray(json.loads(param["base_score"]), dtype=np.float32).reshape(-1)
        if base.size == 1:
            base = np.repeat(base, num_class)
        trees = []
        for spec, group in zip(booster["model"]["trees"], booster["model"]["tree_info"], strict=True):
            if any(spec["split_type"]):
                raise ValueError("不支持类别型切分")
            left = np.asarray(spec["left_children"], dtype=np.int64)
            trees.append(
                _Tree(
                    left=left,
                    right=np.asarray(spec["right_children"], dtype=np.int64),
                    feature=np.asarray(spec["split_indices"], dtype=np.int64),
                    threshold=np.asarray(spec["split_conditions"], dtype=np.float32),
                    default_left=np.asarray(spec["default_left"], dtype=bool),
                    is_leaf=left == -1,
                    value=np.asarray(spec["split_conditions"], dtype=np.float32),
                    group=int(group),
                )
            )
        return cls(trees, base, num_class, list(learner.get("feature_names") or []))

    @classmethod
    def load(cls, path: Path) -> "TreeEnsemble":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict_margin(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        margin = np.tile(self.base_score, (x.shape[0], 1))
        rows = np.arange(x.shape[0])
        for tree in self.trees:
            node = np.zeros(x.shape[0], dtype=np.int64)
            active = ~tree.is_leaf[node]
            while active.any():
                idx = node[active]
                fval = x[rows[active], tree.feature[idx]]
                go_left = np.where(np.isnan(fval), tree.default_left[idx], fval < tree.threshold[idx])
                node[active] = np.where(go_left, tree.left[idx], tree.right[idx])
                active = ~tree.is_leaf[node]
            margin[:, tree.group] += tree.value[node]
        return margin

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        margin = self.predict_margin(x).astype(np.float64)
        margin -= margin.max(axis=1, keepdims=True)
        exp = np.exp(margin)
        return exp / exp.sum(axis=1, keepdims=True)
