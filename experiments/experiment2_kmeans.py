from __future__ import annotations

import ctypes
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
RESULT_DIR = ROOT / "results" / "experiment2_kmeans"

DATA_PATH = PROCESSED_DIR / "nba_scaled.csv"
FEATURE_PATH = PROCESSED_DIR / "feature_columns.txt"
TARGET_COL = "Pos"
TARGET_LABEL_COL = "Pos_Label"

RANDOM_STATE = 42
MAIN_K = 5
K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE_SIZE = 5000


def patch_threadpoolctl_openblas_version() -> None:
    try:
        import threadpoolctl
    except ImportError:
        return

    openblas_module = getattr(threadpoolctl, "_OpenBLASModule", None)
    if openblas_module is None:
        return

    def safe_get_version(self):
        get_config = getattr(self._dynlib, "openblas_get_config", None)
        if get_config is None:
            return None
        get_config.restype = ctypes.c_char_p
        config = get_config()
        if not config:
            return None
        parts = config.split()
        if parts and parts[0] == b"OpenBLAS" and len(parts) > 1:
            return parts[1].decode("utf-8")
        return None

    def safe_get_threading_layer(self):
        get_parallel = getattr(self._dynlib, "openblas_get_parallel", None)
        if get_parallel is None:
            return "unknown"
        threading_layer = get_parallel()
        if threading_layer == 2:
            return "openmp"
        if threading_layer == 1:
            return "pthreads"
        return "disabled"

    def safe_get_architecture(self):
        get_corename = getattr(self._dynlib, "openblas_get_corename", None)
        if get_corename is None:
            return None
        get_corename.restype = ctypes.c_char_p
        corename = get_corename()
        return corename.decode("utf-8") if corename else None

    openblas_module.get_version = safe_get_version
    openblas_module.get_threading_layer = safe_get_threading_layer
    openblas_module.get_architecture = safe_get_architecture


def load_feature_columns() -> list[str]:
    return [line.strip() for line in FEATURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_kmeans(k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)


def save_line_plot(x_values: list[int], y_values: list[float], title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel(ylabel)
    plt.xticks(x_values)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=160)
    plt.close()


def save_heatmap(table: pd.DataFrame, title: str, filename: str, fmt: str = "d") -> None:
    matrix = table.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(matrix, cmap="YlGnBu")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("True Position")
    ax.set_ylabel("Cluster")
    ax.set_xticks(range(len(table.columns)), labels=list(table.columns))
    ax.set_yticks(range(len(table.index)), labels=list(table.index))

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = f"{value:.1%}" if fmt == ".1%" else str(int(value))
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=10,
            )
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=160)
    plt.close()


def save_pca_plot(pca_df: pd.DataFrame, color_col: str, filename: str, title: str) -> None:
    plt.figure(figsize=(8, 6))
    values = sorted(pca_df[color_col].astype(str).unique())
    cmap = plt.get_cmap("tab10")
    for idx, value in enumerate(values):
        subset = pca_df[pca_df[color_col].astype(str) == value]
        plt.scatter(subset["PC1"], subset["PC2"], s=14, alpha=0.55, label=value, color=cmap(idx % 10))
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title=color_col, markerscale=1.4)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=160)
    plt.close()


def get_majority_mapping(assignments: pd.DataFrame) -> dict[int, str]:
    mapping = {}
    for cluster, group in assignments.groupby("Cluster"):
        mapping[int(cluster)] = str(group[TARGET_COL].value_counts().idxmax())
    return mapping


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Please run scripts/preprocess_nba.py before this experiment.")

    patch_threadpoolctl_openblas_version()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    feature_cols = load_feature_columns()
    x = data[feature_cols]
    true_pos = data[TARGET_COL].astype(str)
    true_labels = data[TARGET_LABEL_COL]
    labels = sorted(true_pos.unique())

    k_values = list(K_RANGE)
    inertias = []
    silhouette_scores = []
    for k in k_values:
        model = make_kmeans(k)
        cluster_labels = model.fit_predict(x)
        inertias.append(float(model.inertia_))
        silhouette_scores.append(
            float(
                silhouette_score(
                    x,
                    cluster_labels,
                    sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(data)),
                    random_state=RANDOM_STATE,
                )
            )
        )

    k_selection = pd.DataFrame({"K": k_values, "Inertia": inertias, "Silhouette_Score": silhouette_scores})
    k_selection.to_csv(RESULT_DIR / "k_selection_metrics.csv", index=False, encoding="utf-8")
    save_line_plot(k_values, inertias, "K-Means Elbow Curve", "Inertia", "elbow_curve.png")
    save_line_plot(k_values, silhouette_scores, "Silhouette Score by K", "Silhouette Score", "silhouette_scores.png")

    model = make_kmeans(MAIN_K)
    clusters = model.fit_predict(x)

    assignments = pd.DataFrame(
        {
            "Row_Index": data.index,
            TARGET_COL: true_pos,
            TARGET_LABEL_COL: true_labels,
            "Cluster": clusters,
        }
    )
    majority_mapping = get_majority_mapping(assignments)
    assignments["Mapped_Position"] = assignments["Cluster"].map(majority_mapping)
    assignments["Correct_By_Majority"] = assignments[TARGET_COL] == assignments["Mapped_Position"]
    assignments.to_csv(RESULT_DIR / "cluster_assignments.csv", index=False, encoding="utf-8")

    cluster_index = sorted(assignments["Cluster"].unique())
    crosstab = pd.crosstab(assignments["Cluster"], assignments[TARGET_COL]).reindex(index=cluster_index, columns=labels, fill_value=0)
    crosstab.to_csv(RESULT_DIR / "cluster_position_crosstab.csv", encoding="utf-8")
    save_heatmap(crosstab, "Cluster and True Position Crosstab", "cluster_position_heatmap.png")

    ratio_table = crosstab.div(crosstab.sum(axis=1), axis=0).fillna(0)
    ratio_table.to_csv(RESULT_DIR / "cluster_position_ratio.csv", encoding="utf-8")
    save_heatmap(ratio_table, "Position Ratio Within Each Cluster", "cluster_position_ratio_heatmap.png", fmt=".1%")

    majority_accuracy = float(assignments["Correct_By_Majority"].mean())
    ari = float(adjusted_rand_score(true_labels, clusters))
    nmi = float(normalized_mutual_info_score(true_labels, clusters))
    homogeneity = float(homogeneity_score(true_labels, clusters))
    silhouette_main = float(
        silhouette_score(
            x,
            clusters,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(data)),
            random_state=RANDOM_STATE,
        )
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(x)
    pca_df = pd.DataFrame(
        {
            "PC1": pca_values[:, 0],
            "PC2": pca_values[:, 1],
            "Cluster": assignments["Cluster"].astype(str),
            "True_Position": assignments[TARGET_COL],
        }
    )
    pca_df.to_csv(RESULT_DIR / "pca_projection.csv", index=False, encoding="utf-8")
    save_pca_plot(pca_df, "Cluster", "pca_clusters.png", "PCA Projection Colored by K-Means Cluster")
    save_pca_plot(pca_df, "True_Position", "pca_true_positions.png", "PCA Projection Colored by True Position")

    cluster_summary = crosstab.copy()
    cluster_summary["Total"] = cluster_summary.sum(axis=1)
    cluster_summary["Majority_Position"] = [majority_mapping[int(cluster)] for cluster in cluster_summary.index]
    cluster_summary["Majority_Count"] = [int(crosstab.loc[cluster, majority_mapping[int(cluster)]]) for cluster in cluster_summary.index]
    cluster_summary["Majority_Ratio"] = (cluster_summary["Majority_Count"] / cluster_summary["Total"]).round(4)
    cluster_summary.to_csv(RESULT_DIR / "cluster_summary.csv", encoding="utf-8")

    metrics = {
        "algorithm": "K-Means",
        "implementation": "sklearn.cluster.KMeans",
        "data": "processed/nba_scaled.csv",
        "rows": int(len(data)),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "k": MAIN_K,
        "random_state": RANDOM_STATE,
        "n_init": 10,
        "inertia": float(model.inertia_),
        "silhouette_score": silhouette_main,
        "adjusted_rand_index": ari,
        "normalized_mutual_information": nmi,
        "homogeneity": homogeneity,
        "majority_mapping_accuracy": majority_accuracy,
        "majority_mapping": {str(k): v for k, v in majority_mapping.items()},
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# 实验二：K-Means 聚类

## 方法

本实验使用原始基础版 K-Means。输入为 `processed/nba_scaled.csv` 中已经标准化的 26 个原始数值统计字段，不加入额外位置结构特征，不使用真实标签参与聚类。

## 实验结果

- K：{MAIN_K}
- Inertia：{model.inertia_:.4f}
- Silhouette Score：{silhouette_main:.4f}
- Adjusted Rand Index：{ari:.4f}
- Normalized Mutual Information：{nmi:.4f}
- Homogeneity：{homogeneity:.4f}
- Majority Mapping Accuracy：{majority_accuracy:.4f}

## 可视化结果

![Elbow Curve](elbow_curve.png)

![Silhouette Scores](silhouette_scores.png)

![Cluster Position Heatmap](cluster_position_heatmap.png)

![Cluster Position Ratio Heatmap](cluster_position_ratio_heatmap.png)

![PCA Clusters](pca_clusters.png)

![PCA True Positions](pca_true_positions.png)

## 聚类簇与真实位置交叉表

{crosstab.to_markdown()}

## 聚类簇主要对应位置

{cluster_summary.to_markdown()}

## 说明

K-Means 是无监督算法，训练时不使用 `Pos` 标签。多数映射准确率只是聚类完成后的辅助解释指标，不等价于监督分类准确率。该版本只展示 K-Means 对原始标准化统计空间的基础聚类效果。
"""
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("Experiment 2 completed: basic K-Means")
    print(f"K: {MAIN_K}")
    print(f"Silhouette Score: {silhouette_main:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Majority Mapping Accuracy: {majority_accuracy:.4f}")
    print(f"Result directory: {RESULT_DIR}")


if __name__ == "__main__":
    main()
