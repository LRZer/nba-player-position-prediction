from __future__ import annotations

import json
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
from sklearn.naive_bayes import GaussianNB


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
RESULT_DIR = ROOT / "results" / "experiment1_bayes"

TRAIN_PATH = PROCESSED_DIR / "nba_scaled_train.csv"
TEST_PATH = PROCESSED_DIR / "nba_scaled_test.csv"
FEATURE_PATH = PROCESSED_DIR / "feature_columns.txt"
TARGET_COL = "Pos"


def load_feature_columns() -> list[str]:
    return [line.strip() for line in FEATURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_confusion_matrix_plot(cm_df: pd.DataFrame) -> None:
    matrix = cm_df.to_numpy()
    x_labels = [col.replace("pred_", "") for col in cm_df.columns]
    y_labels = [idx.replace("true_", "") for idx in cm_df.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Gaussian Naive Bayes Confusion Matrix")
    ax.set_xlabel("Predicted Position")
    ax.set_ylabel("True Position")
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_yticks(range(len(y_labels)), labels=y_labels)

    threshold = matrix.max() / 2
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=12,
            )
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
    plt.title("GaussianNB Precision, Recall and F1 by Position")
    plt.xlabel("Position")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    ax.legend(title="Metric", loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "class_metrics.png", dpi=160)
    plt.close()


def save_distribution_plot(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> None:
    distribution = pd.DataFrame(
        {
            "Position": labels,
            "True": [int((y_true == label).sum()) for label in labels],
            "Predicted": [int((y_pred == label).sum()) for label in labels],
        }
    )
    distribution.to_csv(RESULT_DIR / "prediction_distribution.csv", index=False, encoding="utf-8")
    ax = distribution.set_index("Position")[["True", "Predicted"]].loc[labels].plot(kind="bar", figsize=(9, 5.5), width=0.78)
    plt.title("GaussianNB True vs Predicted Position Distribution")
    plt.xlabel("Position")
    plt.ylabel("Count")
    ax.legend(title="Type", loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "prediction_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Please run scripts/preprocess_nba.py before this experiment.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    feature_cols = load_feature_columns()

    x_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(str)
    x_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(str).reset_index(drop=True)
    labels = sorted(y_train.unique())

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = pd.Series(model.predict(x_test), name="Predicted_Pos")

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

    predictions = test_df[[TARGET_COL]].copy()
    predictions["Predicted_Pos"] = y_pred.values
    predictions["Correct"] = predictions[TARGET_COL] == predictions["Predicted_Pos"]
    predictions.to_csv(RESULT_DIR / "predictions.csv", index=False, encoding="utf-8")

    metrics = {
        "algorithm": "Gaussian Naive Bayes",
        "implementation": "sklearn.naive_bayes.GaussianNB",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "classes": labels,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# 实验一：贝叶斯分类

## 方法

本实验使用原始基础版 Gaussian Naive Bayes。输入特征为预处理后标准化的 26 个原始数值统计字段，不加入额外特征工程，不做分箱优化，不使用集成方法。

## 实验结果

- Accuracy：{accuracy:.4f}
- Macro Precision：{precision:.4f}
- Macro Recall：{recall:.4f}
- Macro F1：{f1:.4f}
- Weighted F1：{weighted_f1:.4f}

## 可视化结果

![Confusion Matrix](confusion_matrix.png)

![Class Metrics](class_metrics.png)

![Prediction Distribution](prediction_distribution.png)

## 混淆矩阵

{cm_df.to_markdown()}

## 分类报告

```text
{report_text}
```

## 说明

该版本作为贝叶斯分类的原始基线，只体现 GaussianNB 本身的分类效果。由于朴素贝叶斯假设各特征条件独立，而 NBA 技术统计字段之间相关性较强，因此该模型主要用于基础算法对比。
"""
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("Experiment 1 completed: Gaussian Naive Bayes")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Result directory: {RESULT_DIR}")


if __name__ == "__main__":
    main()
