import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from clausetrail.structure.tree_ensemble import TreeEnsemble

SRC = Path(__file__).resolve().parents[2] / "src"


def _stump(feature: int, threshold: float, left: float, right: float, default_left: bool) -> dict:
    return {
        "left_children": [1, -1, -1],
        "right_children": [2, -1, -1],
        "split_indices": [feature, 0, 0],
        "split_conditions": [threshold, left, right],
        "default_left": [int(default_left), 0, 0],
        "split_type": [0, 0, 0],
    }


def _model(objective: str = "multi:softprob") -> dict:
    return {
        "learner": {
            "objective": {"name": objective},
            "feature_names": ["a", "b"],
            "learner_model_param": {"base_score": "[5E-1,-5E-1]", "num_class": "2"},
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "trees": [_stump(0, 1.0, 0.3, -0.2, True), _stump(1, 0.0, 0.1, 0.4, False)],
                    "tree_info": [0, 1],
                },
            },
        }
    }


def test_margin_follows_split_rules_and_missing_defaults():
    ensemble = TreeEnsemble.from_json(_model())
    x = np.array([[0.5, -1.0], [1.0, 0.0], [np.nan, np.nan]], dtype=np.float32)
    expected = np.array([[0.8, -0.4], [0.3, -0.1], [0.8, -0.1]], dtype=np.float32)
    np.testing.assert_allclose(ensemble.predict_margin(x), expected, atol=1e-6)
    np.testing.assert_allclose(ensemble.predict_proba(x).sum(axis=1), 1.0, atol=1e-9)
    assert ensemble.feature_names == ["a", "b"]


def test_unsupported_objective_is_rejected():
    with pytest.raises(ValueError):
        TreeEnsemble.from_json(_model("reg:squarederror"))


def _run(code: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


# xgboost 与 milvus-lite 不能同进程加载，对拍放在子进程里
@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost 未安装")
def test_matches_xgboost_predictions():
    code = """
import json
import numpy as np
import xgboost as xgb
from clausetrail.structure.tree_ensemble import TreeEnsemble

rng = np.random.default_rng(3)
x = rng.normal(size=(600, 6)).astype(np.float32)
x[rng.random(x.shape) < 0.15] = np.nan
y = rng.integers(0, 3, size=600)
params = {"objective": "multi:softprob", "num_class": 3, "max_depth": 4, "eta": 0.3, "seed": 1}
booster = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=30)
ensemble = TreeEnsemble.from_json(json.loads(booster.save_raw("json")))
test = rng.normal(size=(400, 6)).astype(np.float32)
test[rng.random(test.shape) < 0.15] = np.nan
print(float(np.abs(booster.predict(xgb.DMatrix(test)) - ensemble.predict_proba(test)).max()))
"""
    assert float(_run(code)) < 1e-5


def test_runtime_path_does_not_load_xgboost():
    code = """
import sys
import clausetrail.ingest.pipeline
import clausetrail.structure.heading_classifier
print("xgboost" in sys.modules)
"""
    assert _run(code) == "False"
