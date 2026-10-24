import argparse
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import xgboost as xgb

from clausetrail.config import get_settings
from clausetrail.ingest.router import body_blocks, parse_document
from clausetrail.structure import heading_classifier
from clausetrail.structure.heading_features import FEATURE_NAMES, extract_features, rule_labels
from clausetrail.structure.tree_ensemble import TreeEnsemble

PARITY_TOLERANCE = 1e-5


def load_overrides(path: Path) -> dict[tuple[str, int, str], int]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[(row["file"], row["page"], row["text"])] = row["level"]
    return out


def build_dataset(files: list[Path], overrides: dict, use_layout: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for gid, path in enumerate(files):
        blocks = body_blocks(parse_document(path, use_layout=use_layout))
        labels = rule_labels(blocks)
        for i, b in enumerate(blocks):
            key = (path.name, b.page, b.text.strip())
            labels[i] = overrides.get(key, labels[i])
        xs.append(extract_features(blocks))
        ys.extend(labels)
        groups.extend([gid] * len(blocks))
    return np.vstack(xs), np.asarray(ys), np.asarray(groups)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> float:
    scores = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def leave_one_document_out(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    classes = sorted(set(y.tolist()))
    folds = []
    for g in sorted(set(groups.tolist())):
        train, test = groups != g, groups == g
        booster = heading_classifier.train(x[train], y[train])
        pred = booster.predict(xgb.DMatrix(x[test], feature_names=FEATURE_NAMES)).argmax(axis=1)
        folds.append(
            {
                "held_out_group": int(g),
                "n": int(test.sum()),
                "accuracy": float((pred == y[test]).mean()),
                "macro_f1": macro_f1(y[test], pred, classes),
            }
        )
    return {
        "folds": folds,
        "mean_accuracy": float(np.mean([f["accuracy"] for f in folds])),
        "mean_macro_f1": float(np.mean([f["macro_f1"] for f in folds])),
    }


# 运行期用 TreeEnsemble 推理，保存前确认它与 XGBoost 输出一致
def check_parity(booster: xgb.Booster, x: np.ndarray) -> float:
    ensemble = TreeEnsemble.from_json(json.loads(booster.save_raw("json")))
    expected = booster.predict(xgb.DMatrix(x, feature_names=FEATURE_NAMES))
    diff = float(np.abs(expected - ensemble.predict_proba(x)).max())
    if diff > PARITY_TOLERANCE:
        raise RuntimeError(f"TreeEnsemble 与 XGBoost 输出不一致：max diff {diff:.2e}")
    return diff


def main() -> int:
    parser = argparse.ArgumentParser(description="训练标题层级 XGBoost 分类器")
    parser.add_argument("--samples", type=Path, default=get_settings().samples_dir)
    parser.add_argument(
        "--overrides", type=Path, default=get_settings().resources_dir / "models" / "heading_overrides.jsonl"
    )
    # 训练和入库要用同一种解析方式，否则 layout_title 等特征的分布对不上
    parser.add_argument("--no-layout", action="store_true", help="不跑版面分析（默认跟随 LAYOUT_ANALYSIS）")
    args = parser.parse_args()

    random.seed(7)
    use_layout = get_settings().layout_analysis and not args.no_layout
    files = sorted(args.samples.glob("*.pdf"))
    x, y, groups = build_dataset(files, load_overrides(args.overrides), use_layout)
    cv = leave_one_document_out(x, y, groups)
    booster = heading_classifier.train(x, y)
    parity = check_parity(booster, x)
    importance = booster.get_score(importance_type="gain")
    meta = {
        "trained_on": date.today().isoformat(),
        "files": [f.name for f in files],
        "n_blocks": len(y),
        "layout_analysis": use_layout,
        "label_distribution": {str(k): v for k, v in sorted(Counter(y.tolist()).items())},
        "labels": "0=正文 1=章 2=节/条 3=款/小标题",
        "label_source": "规则预标注 + heading_overrides.jsonl 人工修正",
        "top_features": sorted(importance.items(), key=lambda kv: -kv[1])[:10],
        "runtime": "TreeEnsemble（纯 numpy）",
        "parity_max_abs_diff": parity,
    }
    path = heading_classifier.save(booster, meta)
    # 交叉验证结果只打印给训练的人看，不写进模型文件：数值取决于用哪些条款训练
    print(json.dumps({"n_blocks": meta["n_blocks"], "label_distribution": meta["label_distribution"]}, indent=1))
    print("留一文档交叉验证：", json.dumps(cv, ensure_ascii=False, indent=1))
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
