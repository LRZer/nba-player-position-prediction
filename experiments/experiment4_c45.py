from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
RESULT_DIR = ROOT / "results" / "experiment4_c45"

TRAIN_PATH = PROCESSED_DIR / "nba_discrete_train.csv"
TEST_PATH = PROCESSED_DIR / "nba_discrete_test.csv"
FEATURE_PATH = PROCESSED_DIR / "feature_columns.txt"
TARGET_COL = "Pos"


def load_feature_columns() -> list[str]:
    return [line.strip() for line in FEATURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def entropy(labels: pd.Series) -> float:
    counts = labels.value_counts()
    total = len(labels)
    return float(-sum((count / total) * math.log2(count / total) for count in counts if count > 0))


def information_gain(frame: pd.DataFrame, feature: str) -> float:
    base_entropy = entropy(frame[TARGET_COL])
    total = len(frame)
    conditional_entropy = 0.0
    for _, subset in frame.groupby(feature):
        conditional_entropy += len(subset) / total * entropy(subset[TARGET_COL])
    return base_entropy - conditional_entropy


def split_info(frame: pd.DataFrame, feature: str) -> float:
    total = len(frame)
    value_counts = frame[feature].value_counts()
    return float(-sum((count / total) * math.log2(count / total) for count in value_counts if count > 0))


def gain_ratio(frame: pd.DataFrame, feature: str) -> float:
    split = split_info(frame, feature)
    if split == 0:
        return 0.0
    return information_gain(frame, feature) / split


def majority_label(labels: pd.Series) -> str:
    return str(labels.value_counts().idxmax())


def build_c45_tree(frame: pd.DataFrame, features: list[str], feature_usage: Counter[str]) -> dict[str, Any]:
    labels = frame[TARGET_COL].astype(str)
    majority = majority_label(labels)

    if labels.nunique() == 1:
        return {"type": "leaf", "label": str(labels.iloc[0]), "samples": int(len(frame))}
    if not features:
        return {"type": "leaf", "label": majority, "samples": int(len(frame))}

    ratios = {feature: gain_ratio(frame, feature) for feature in features}
    best_feature = max(ratios, key=ratios.get)
    if ratios[best_feature] <= 0:
        return {"type": "leaf", "label": majority, "samples": int(len(frame))}

    feature_usage[best_feature] += 1
    remaining_features = [feature for feature in features if feature != best_feature]
    children = {}
    for value, subset in frame.groupby(best_feature):
        children[str(value)] = build_c45_tree(subset, remaining_features, feature_usage)

    return {
        "type": "node",
        "feature": best_feature,
        "gain_ratio": float(ratios[best_feature]),
        "information_gain": float(information_gain(frame, best_feature)),
        "split_info": float(split_info(frame, best_feature)),
        "majority_label": majority,
        "samples": int(len(frame)),
        "children": children,
    }


def predict_one(tree: dict[str, Any], row: pd.Series) -> str:
    current = tree
    while current["type"] != "leaf":
        feature = current["feature"]
        value = str(row[feature])
        child = current["children"].get(value)
        if child is None:
            return str(current["majority_label"])
        current = child
    return str(current["label"])


def tree_summary(tree: dict[str, Any]) -> dict[str, int]:
    if tree["type"] == "leaf":
        return {"node_count": 1, "leaf_count": 1, "max_depth": 0}
    node_count = 1
    leaf_count = 0
    max_child_depth = 0
    for child in tree["children"].values():
        child_summary = tree_summary(child)
        node_count += child_summary["node_count"]
        leaf_count += child_summary["leaf_count"]
        max_child_depth = max(max_child_depth, child_summary["max_depth"])
    return {"node_count": node_count, "leaf_count": leaf_count, "max_depth": max_child_depth + 1}


def write_rules(tree: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []

    def walk(node: dict[str, Any], conditions: list[str]) -> None:
        if node["type"] == "leaf":
            prefix = " AND ".join(conditions) if conditions else "TRUE"
            lines.append(f"IF {prefix} THEN Pos = {node['label']}  [samples={node['samples']}]")
            return
        feature = node["feature"]
        for value, child in sorted(node["children"].items()):
            walk(child, conditions + [f"{feature} == {value}"])

    walk(tree, [])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_confusion_matrix_plot(cm_df: pd.DataFrame) -> None:
    matrix = cm_df.to_numpy()
    labels_x = [col.replace("pred_", "") for col in cm_df.columns]
    labels_y = [idx.replace("true_", "") for idx in cm_df.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("C4.5 Confusion Matrix")
    ax.set_xlabel("Predicted Position")
    ax.set_ylabel("True Position")
    ax.set_xticks(range(len(labels_x)), labels=labels_x)
    ax.set_yticks(range(len(labels_y)), labels=labels_y)
    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > threshold else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def save_class_metrics_plot(report_dict: dict[str, Any], labels: list[str]) -> None:
    rows = []
    for label in labels:
        for metric in ["precision", "recall", "f1-score"]:
            rows.append({"Position": label, "Metric": metric, "Score": report_dict[label][metric]})
    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(RESULT_DIR / "class_metrics.csv", index=False, encoding="utf-8")
    pivot = metric_df.pivot(index="Position", columns="Metric", values="Score").loc[labels]
    ax = pivot.plot(kind="bar", figsize=(9, 5.5), width=0.78)
    plt.title("C4.5 Precision, Recall and F1 by Position")
    plt.xlabel("Position")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    ax.legend(title="Metric", loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "class_metrics.png", dpi=160)
    plt.close()


def save_distribution_plot(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> None:
    distribution = pd.DataFrame({
        "Position": labels,
        "True": [int((y_true == label).sum()) for label in labels],
        "Predicted": [int((y_pred == label).sum()) for label in labels],
    })
    distribution.to_csv(RESULT_DIR / "prediction_distribution.csv", index=False, encoding="utf-8")
    ax = distribution.set_index("Position")[["True", "Predicted"]].loc[labels].plot(kind="bar", figsize=(9, 5.5), width=0.78)
    plt.title("C4.5 True vs Predicted Position Distribution")
    plt.xlabel("Position")
    plt.ylabel("Count")
    ax.legend(title="Type", loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "prediction_distribution.png", dpi=160)
    plt.close()


def save_feature_usage_plot(feature_usage: Counter[str]) -> pd.DataFrame:
    usage_df = pd.DataFrame(
        [{"Feature": feature, "Split_Count": count} for feature, count in feature_usage.items()]
    ).sort_values("Split_Count", ascending=False)
    usage_df.to_csv(RESULT_DIR / "feature_usage.csv", index=False, encoding="utf-8")
    if not usage_df.empty:
        ax = usage_df.head(15).set_index("Feature")["Split_Count"].plot(kind="barh", figsize=(9, 5.5), color="#2f7f6f")
        ax.invert_yaxis()
        plt.title("C4.5 Feature Usage")
        plt.xlabel("Split Count")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "feature_usage.png", dpi=160)
        plt.close()
    return usage_df


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Please run scripts/preprocess_nba.py before this experiment.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH).astype(str)
    test_df = pd.read_csv(TEST_PATH).astype(str)
    feature_cols = load_feature_columns()
    labels = sorted(train_df[TARGET_COL].unique())

    feature_usage: Counter[str] = Counter()
    tree = build_c45_tree(train_df[feature_cols + [TARGET_COL]], feature_cols, feature_usage)
    y_test = test_df[TARGET_COL].reset_index(drop=True)
    y_pred = pd.Series([predict_one(tree, row) for _, row in test_df[feature_cols].iterrows()], name="Predicted_Pos")

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, average="weighted", zero_division=0)

    cm_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    cm_df.to_csv(RESULT_DIR / "confusion_matrix.csv", encoding="utf-8")
    save_confusion_matrix_plot(cm_df)

    report_text = classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    (RESULT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_class_metrics_plot(report_dict, labels)
    save_distribution_plot(y_test, y_pred, labels)
    usage_df = save_feature_usage_plot(feature_usage)

    predictions = test_df[[TARGET_COL]].copy()
    predictions["Predicted_Pos"] = y_pred.values
    predictions["Correct"] = predictions[TARGET_COL] == predictions["Predicted_Pos"]
    predictions.to_csv(RESULT_DIR / "predictions.csv", index=False, encoding="utf-8")

    (RESULT_DIR / "tree.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_rules(tree, RESULT_DIR / "tree_rules.txt")
    (RESULT_DIR / "selected_features.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

    summary = tree_summary(tree)
    parameter_selection = {"parameter_tuning": False, "note": "基础版 C4.5 未做参数调优、剪枝增强或集成。"}
    (RESULT_DIR / "parameter_selection.json").write_text(json.dumps(parameter_selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = {
        "algorithm": "C4.5 Decision Tree",
        "implementation": "custom recursive C4.5 using gain ratio",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "classes": labels,
        "criterion": "gain_ratio",
        "data": "processed/nba_discrete_train.csv / processed/nba_discrete_test.csv",
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "tree_summary": summary,
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# 实验四：C4.5 分类

## 方法

本实验使用基础版 C4.5 思路。算法使用增益率选择划分属性，递归生成决策树。输入为预处理阶段生成的三档离散特征，不加入额外特征工程，不使用 Bagging，不做调参增强。

## 实验结果

- Accuracy：{accuracy:.4f}
- Macro Precision：{precision:.4f}
- Macro Recall：{recall:.4f}
- Macro F1：{f1:.4f}
- Weighted F1：{weighted_f1:.4f}
- Tree nodes：{summary["node_count"]}
- Leaf nodes：{summary["leaf_count"]}
- Max depth：{summary["max_depth"]}

## 可视化结果

![Confusion Matrix](confusion_matrix.png)

![Class Metrics](class_metrics.png)

![Prediction Distribution](prediction_distribution.png)

![Feature Usage](feature_usage.png)

## 混淆矩阵

{cm_df.to_markdown()}

## 分类报告

```text
{report_text}
```

## 特征使用次数

{usage_df.to_markdown(index=False)}

## 说明

该版本只体现 C4.5 相对 ID3 的核心区别：用增益率代替信息增益，以减弱多取值属性偏好的影响。为了保持基础实验口径，本版本不加入连续阈值优化、剪枝调参或集成方法。
"""
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("Experiment 4 completed: basic C4.5")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Result directory: {RESULT_DIR}")


if __name__ == "__main__":
    main()
