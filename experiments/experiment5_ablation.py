from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment5_dnn import (
    AUXILIARY_TREE_CONFIG,
    AUXILIARY_TREE_WEIGHT,
    BATCH_SIZE,
    CLEAN_TEST_PATH,
    CLEAN_TRAIN_PATH,
    DEVICE,
    ENGINEERED_FEATURES,
    ERA_EXPERT_MEMBERS,
    ERA_EXPERT_WEIGHT,
    FINAL_ENSEMBLE_MEMBERS,
    LABEL_SMOOTHING,
    RANDOM_STATE,
    RESULT_DIR as EXP5_RESULT_DIR,
    SELECTED_FEATURES,
    SUPCON_TEMPERATURE,
    TARGET_COL,
    add_engineered_features,
    era_bucket,
    load_feature_columns,
    make_model,
    select_feature_columns,
    supervised_contrastive_loss,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "experiment5_ablation"


@dataclass(frozen=True)
class AblationConfig:
    name: str
    description: str
    feature_mode: str
    use_global_ensemble: bool = True
    use_era_experts: bool = True
    use_auxiliary_tree: bool = True
    use_contrastive_loss: bool = True
    use_contrastive_members: bool = True
    use_full_ensemble: bool = True
    source: str = "train"


ABLATION_CONFIGS = [
    AblationConfig(
        name="A0_full_model",
        description="完整模型：45维核心特征 + 全局DNN集成 + 监督对比学习 + 年代专家 + ExtraTrees概率融合",
        feature_mode="selected_45",
        source="current_metrics",
    ),
    AblationConfig(
        name="A1_all_76_features",
        description="取消特征筛选，使用全部76个候选特征",
        feature_mode="all_76",
    ),
    AblationConfig(
        name="A2_raw_26_features",
        description="只使用26个原始数值特征，去掉全部工程特征",
        feature_mode="raw_26",
    ),
    AblationConfig(
        name="A3_no_contrastive_loss",
        description="保留对比网络结构，但训练时去掉监督对比学习损失",
        feature_mode="selected_45",
        use_contrastive_loss=False,
    ),
    AblationConfig(
        name="A4_no_contrastive_members",
        description="去掉Contrastive Residual DNN成员，只保留普通Residual DNN集成",
        feature_mode="selected_45",
        use_contrastive_members=False,
    ),
    AblationConfig(
        name="A5_no_era_experts",
        description="去掉年代专家DNN，只使用全局DNN集成与ExtraTrees融合",
        feature_mode="selected_45",
        use_era_experts=False,
    ),
    AblationConfig(
        name="A6_no_auxiliary_tree",
        description="去掉ExtraTrees概率融合，只使用全局DNN与年代专家DNN融合",
        feature_mode="selected_45",
        use_auxiliary_tree=False,
        source="derive_current",
    ),
    AblationConfig(
        name="A7_single_residual_dnn",
        description="只训练一个普通Residual DNN，不做集成、不做年代专家、不做ExtraTrees融合",
        feature_mode="selected_45",
        use_era_experts=False,
        use_auxiliary_tree=False,
        use_full_ensemble=False,
    ),
    AblationConfig(
        name="A8_auxiliary_tree_only",
        description="只使用ExtraTrees，不使用DNN",
        feature_mode="selected_45",
        use_global_ensemble=False,
        use_era_experts=False,
        use_auxiliary_tree=True,
        source="derive_current",
    ),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def clone_member(member: dict[str, Any], weight: float | None = None) -> dict[str, Any]:
    cloned = dict(member)
    if weight is not None:
        cloned["weight"] = weight
    return cloned


def members_for_config(config: AblationConfig, source_members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not config.use_full_ensemble:
        first_residual = next(member for member in source_members if member.get("model_type") == "residual")
        return [clone_member(first_residual, weight=1.0)]
    members = [
        clone_member(member)
        for member in source_members
        if config.use_contrastive_members or member.get("model_type") != "contrastive_residual"
    ]
    total_weight = sum(float(member["weight"]) for member in members)
    return [clone_member(member, weight=float(member["weight"]) / total_weight) for member in members]


def feature_columns_for_mode(mode: str, base_features: list[str]) -> list[str]:
    if mode == "selected_45":
        return select_feature_columns(base_features)
    if mode == "all_76":
        return base_features + ENGINEERED_FEATURES
    if mode == "raw_26":
        return base_features
    raise ValueError(f"Unknown feature mode: {mode}")


def train_member(
    member: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    class_count: int,
    use_contrastive_loss: bool,
) -> np.ndarray:
    set_seed(int(member["seed"]))
    model = make_model(member, x_train.shape[1], class_count).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=member["learning_rate"], weight_decay=member["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(member["epochs"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type == "cuda",
    )

    for _ in range(int(member["epochs"])):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(x_batch)
            if isinstance(output, tuple):
                logits, embeddings = output
                loss = criterion(logits, y_batch)
                if use_contrastive_loss:
                    loss = loss + 0.10 * supervised_contrastive_loss(embeddings, y_batch, temperature=SUPCON_TEMPERATURE)
            else:
                loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(x_eval, dtype=torch.float32, device=DEVICE))
        logits = output[0] if isinstance(output, tuple) else output
        return torch.softmax(logits, dim=1).cpu().numpy()


def train_ensemble_probabilities(
    members: list[dict[str, Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    class_count: int,
    use_contrastive_loss: bool,
) -> np.ndarray:
    weighted = []
    for member in members:
        probabilities = train_member(member, x_train, y_train, x_eval, class_count, use_contrastive_loss)
        weighted.append(float(member["weight"]) * probabilities)
    return np.sum(weighted, axis=0)


def train_era_probabilities(
    config: AblationConfig,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_all: np.ndarray,
    feature_cols: list[str],
    labels: list[str],
    fallback_probabilities: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    era_probabilities = np.zeros_like(fallback_probabilities)
    era_details = []
    members = members_for_config(config, ERA_EXPERT_MEMBERS)
    train_eras = train_df["Year"].map(era_bucket).to_numpy()
    eval_eras = eval_df["Year"].map(era_bucket).to_numpy()
    for era_id in sorted(np.unique(eval_eras)):
        train_mask = train_eras == era_id
        eval_mask = eval_eras == era_id
        if train_mask.sum() == 0:
            era_probabilities[eval_mask] = fallback_probabilities[eval_mask]
            continue
        scaler = StandardScaler()
        x_era_train = scaler.fit_transform(train_df.loc[train_mask, feature_cols])
        x_era_eval = scaler.transform(eval_df.loc[eval_mask, feature_cols])
        y_era_train = y_all[train_mask]
        era_probabilities[eval_mask] = train_ensemble_probabilities(
            members,
            x_era_train,
            y_era_train,
            x_era_eval,
            len(labels),
            config.use_contrastive_loss,
        )
        era_details.append({
            "era_id": int(era_id),
            "train_rows": int(train_mask.sum()),
            "eval_rows": int(eval_mask.sum()),
        })
    return era_probabilities, era_details


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, float]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def write_variant_outputs(
    variant_dir: Path,
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    metrics: dict[str, Any],
) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    cm_df.to_csv(variant_dir / "confusion_matrix.csv", encoding="utf-8")
    report_text = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    (variant_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame({"True_Pos": y_true, "Predicted_Pos": y_pred, "Correct": y_true.values == y_pred.values}).to_csv(
        variant_dir / "predictions.csv",
        index=False,
        encoding="utf-8",
    )
    (variant_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_training_ablation(config: AblationConfig, force: bool) -> dict[str, Any]:
    variant_dir = RESULT_DIR / config.name
    metrics_path = variant_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    set_seed(RANDOM_STATE)
    base_features = load_feature_columns()
    feature_cols = feature_columns_for_mode(config.feature_mode, base_features)
    train_df = add_engineered_features(pd.read_csv(CLEAN_TRAIN_PATH))
    test_df = add_engineered_features(pd.read_csv(CLEAN_TEST_PATH))
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(train_df[TARGET_COL].astype(str))
    labels = list(label_encoder.classes_)
    y_test = test_df[TARGET_COL].astype(str).reset_index(drop=True)

    final_probabilities: np.ndarray | None = None
    component_metrics: dict[str, float] = {}
    era_details: list[dict[str, int]] = []
    global_probabilities: np.ndarray | None = None

    if config.use_global_ensemble:
        members = members_for_config(config, FINAL_ENSEMBLE_MEMBERS)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[feature_cols])
        x_test = scaler.transform(test_df[feature_cols])
        global_probabilities = train_ensemble_probabilities(
            members,
            x_train,
            y_all,
            x_test,
            len(labels),
            config.use_contrastive_loss,
        )
        global_pred = pd.Series(label_encoder.inverse_transform(global_probabilities.argmax(axis=1)))
        component_metrics["global_accuracy"] = float(accuracy_score(y_test, global_pred))
        component_metrics["global_macro_f1"] = float(precision_recall_fscore_support(y_test, global_pred, labels=labels, average="macro", zero_division=0)[2])
        final_probabilities = global_probabilities.copy()

        if config.use_era_experts:
            era_probabilities, era_details = train_era_probabilities(
                config,
                train_df,
                test_df,
                y_all,
                feature_cols,
                labels,
                global_probabilities,
            )
            era_pred = pd.Series(label_encoder.inverse_transform(era_probabilities.argmax(axis=1)))
            component_metrics["era_accuracy"] = float(accuracy_score(y_test, era_pred))
            component_metrics["era_macro_f1"] = float(precision_recall_fscore_support(y_test, era_pred, labels=labels, average="macro", zero_division=0)[2])
            final_probabilities = (1 - ERA_EXPERT_WEIGHT) * global_probabilities + ERA_EXPERT_WEIGHT * era_probabilities

    auxiliary_tree_probabilities: np.ndarray | None = None
    if config.use_auxiliary_tree:
        auxiliary_tree = ExtraTreesClassifier(**AUXILIARY_TREE_CONFIG)
        auxiliary_tree.fit(train_df[feature_cols], y_all)
        auxiliary_tree_probabilities = auxiliary_tree.predict_proba(test_df[feature_cols])
        tree_pred = pd.Series(label_encoder.inverse_transform(auxiliary_tree_probabilities.argmax(axis=1)))
        component_metrics["auxiliary_tree_accuracy"] = float(accuracy_score(y_test, tree_pred))
        component_metrics["auxiliary_tree_macro_f1"] = float(precision_recall_fscore_support(y_test, tree_pred, labels=labels, average="macro", zero_division=0)[2])
        if final_probabilities is None:
            final_probabilities = auxiliary_tree_probabilities
        else:
            final_probabilities = (1 - AUXILIARY_TREE_WEIGHT) * final_probabilities + AUXILIARY_TREE_WEIGHT * auxiliary_tree_probabilities

    if final_probabilities is None:
        raise RuntimeError(f"No model components enabled for {config.name}")

    y_pred = pd.Series(label_encoder.inverse_transform(final_probabilities.argmax(axis=1)), name="Predicted_Pos")
    metrics = {
        "name": config.name,
        "description": config.description,
        "feature_mode": config.feature_mode,
        "feature_count": len(feature_cols),
        "selected_base_feature_count": sum(feature in base_features for feature in feature_cols),
        "selected_engineered_feature_count": sum(feature in ENGINEERED_FEATURES for feature in feature_cols),
        "use_global_ensemble": config.use_global_ensemble,
        "use_era_experts": config.use_era_experts,
        "use_auxiliary_tree": config.use_auxiliary_tree,
        "use_contrastive_loss": config.use_contrastive_loss,
        "use_contrastive_members": config.use_contrastive_members,
        "use_full_ensemble": config.use_full_ensemble,
        "global_member_count": len(members_for_config(config, FINAL_ENSEMBLE_MEMBERS)) if config.use_global_ensemble else 0,
        "era_member_count": len(members_for_config(config, ERA_EXPERT_MEMBERS)) if config.use_era_experts else 0,
        "era_expert_weight": ERA_EXPERT_WEIGHT if config.use_era_experts else 0.0,
        "auxiliary_tree_weight": AUXILIARY_TREE_WEIGHT if config.use_auxiliary_tree and config.use_global_ensemble else 1.0 if config.use_auxiliary_tree else 0.0,
        "era_details": era_details,
        **component_metrics,
        **evaluate_predictions(y_test, y_pred, labels),
    }
    write_variant_outputs(variant_dir, y_test, y_pred, labels, metrics)
    (variant_dir / "selected_features.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")
    return metrics


def derive_from_current(config: AblationConfig, force: bool) -> dict[str, Any]:
    variant_dir = RESULT_DIR / config.name
    metrics_path = variant_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    current_metrics = json.loads((EXP5_RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    y_test = pd.read_csv(CLEAN_TEST_PATH)[TARGET_COL].astype(str).reset_index(drop=True)
    predictions = pd.read_csv(EXP5_RESULT_DIR / "predictions.csv")
    labels = current_metrics["classes"]

    if config.name == "A6_no_auxiliary_tree":
        pred_col = "DNN_Ensemble_Predicted_Pos"
    elif config.name == "A8_auxiliary_tree_only":
        pred_col = "Auxiliary_Tree_Predicted_Pos"
    else:
        raise ValueError(f"Unsupported derived config: {config.name}")

    y_pred = predictions[pred_col].astype(str).reset_index(drop=True)
    metrics = {
        "name": config.name,
        "description": config.description,
        "feature_mode": config.feature_mode,
        "feature_count": int(current_metrics["feature_count"]),
        "selected_base_feature_count": int(current_metrics["base_feature_count"]),
        "selected_engineered_feature_count": int(current_metrics["engineered_feature_count"]),
        "use_global_ensemble": config.use_global_ensemble,
        "use_era_experts": config.use_era_experts,
        "use_auxiliary_tree": config.use_auxiliary_tree,
        "use_contrastive_loss": config.use_contrastive_loss,
        "use_contrastive_members": config.use_contrastive_members,
        "use_full_ensemble": config.use_full_ensemble,
        "source": "derived_from_current_experiment5_outputs",
        **evaluate_predictions(y_test, y_pred, labels),
    }
    write_variant_outputs(variant_dir, y_test, y_pred, labels, metrics)
    return metrics


def copy_current_full(force: bool) -> dict[str, Any]:
    config = ABLATION_CONFIGS[0]
    variant_dir = RESULT_DIR / config.name
    metrics_path = variant_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    current_metrics = json.loads((EXP5_RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(EXP5_RESULT_DIR / "predictions.csv")
    y_true = predictions[TARGET_COL].astype(str).reset_index(drop=True)
    y_pred = predictions["Predicted_Pos"].astype(str).reset_index(drop=True)
    labels = current_metrics["classes"]
    metrics = {
        "name": config.name,
        "description": config.description,
        "feature_mode": config.feature_mode,
        "feature_count": int(current_metrics["feature_count"]),
        "selected_base_feature_count": int(current_metrics["base_feature_count"]),
        "selected_engineered_feature_count": int(current_metrics["engineered_feature_count"]),
        "use_global_ensemble": True,
        "use_era_experts": True,
        "use_auxiliary_tree": True,
        "use_contrastive_loss": True,
        "use_contrastive_members": True,
        "use_full_ensemble": True,
        "source": "current_experiment5_outputs",
        **evaluate_predictions(y_true, y_pred, labels),
    }
    write_variant_outputs(variant_dir, y_true, y_pred, labels, metrics)
    return metrics


def plot_ablation_results(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["name"], plot_df["macro_f1"], color="#386fa4")
    ax.set_xlabel("Macro F1")
    ax.set_title("Experiment 5 Ablation Study - Macro F1")
    ax.set_xlim(max(0, plot_df["macro_f1"].min() - 0.02), min(1, plot_df["macro_f1"].max() + 0.01))
    for idx, value in enumerate(plot_df["macro_f1"]):
        ax.text(value + 0.001, idx, f"{value:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "ablation_macro_f1.png", dpi=160)
    plt.close()

    delta_df = summary.sort_values("delta_macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d1495b" if value < 0 else "#2a9d8f" for value in delta_df["delta_macro_f1"]]
    ax.barh(delta_df["name"], delta_df["delta_macro_f1"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Macro F1 Difference from Full Model")
    ax.set_title("Ablation Impact Relative to Full Model")
    for idx, value in enumerate(delta_df["delta_macro_f1"]):
        ax.text(value + (0.0005 if value >= 0 else -0.0005), idx, f"{value:+.4f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "ablation_delta_macro_f1.png", dpi=160)
    plt.close()


def write_report(summary: pd.DataFrame) -> None:
    full_row = summary[summary["name"] == "A0_full_model"].iloc[0]
    best_row = summary.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    by_name = summary.set_index("name")

    def row_delta(name: str) -> tuple[float, float, float]:
        row = by_name.loc[name]
        return (
            float(full_row["accuracy"] - row["accuracy"]),
            float(full_row["macro_f1"] - row["macro_f1"]),
            float(full_row["weighted_f1"] - row["weighted_f1"]),
        )

    contribution_rows = []
    contribution_specs = [
        ("工程特征整体贡献", "A0_full_model vs A2_raw_26_features", "去掉全部工程特征，只保留26维原始特征", "A2_raw_26_features"),
        ("特征筛选贡献", "A0_full_model vs A1_all_76_features", "取消45维核心筛选，使用全部76维候选特征", "A1_all_76_features"),
        ("监督对比学习损失贡献", "A0_full_model vs A3_no_contrastive_loss", "去掉监督对比学习损失", "A3_no_contrastive_loss"),
        ("Contrastive DNN成员贡献", "A0_full_model vs A4_no_contrastive_members", "去掉Contrastive Residual DNN成员", "A4_no_contrastive_members"),
        ("年代专家贡献", "A0_full_model vs A5_no_era_experts", "去掉按Year分段训练的年代专家DNN", "A5_no_era_experts"),
        ("ExtraTrees概率融合贡献", "A0_full_model vs A6_no_auxiliary_tree", "去掉ExtraTrees辅助概率融合", "A6_no_auxiliary_tree"),
        ("DNN集成整体贡献", "A0_full_model vs A7_single_residual_dnn", "退化为单个普通Residual DNN", "A7_single_residual_dnn"),
        ("DNN相对树模型贡献", "A0_full_model vs A8_auxiliary_tree_only", "只使用ExtraTrees，不使用DNN", "A8_auxiliary_tree_only"),
    ]
    for module, comparison, removed, ablation_name in contribution_specs:
        delta_acc, delta_f1, delta_weighted_f1 = row_delta(ablation_name)
        contribution_rows.append({
            "module": module,
            "comparison": comparison,
            "removed_or_changed": removed,
            "accuracy_drop": delta_acc,
            "macro_f1_drop": delta_f1,
            "weighted_f1_drop": delta_weighted_f1,
        })
    contribution = pd.DataFrame(contribution_rows).sort_values("macro_f1_drop", ascending=False)
    contribution.to_csv(RESULT_DIR / "module_contribution_summary.csv", index=False, encoding="utf-8")

    key_columns = [
        "name", "feature_count", "use_contrastive_loss", "use_contrastive_members",
        "use_era_experts", "use_auxiliary_tree", "use_full_ensemble",
        "accuracy", "macro_f1", "weighted_f1", "delta_macro_f1",
    ]
    table = summary[key_columns].sort_values("name").copy()
    for col in ["accuracy", "macro_f1", "weighted_f1", "delta_macro_f1"]:
        table[col] = table[col].map(lambda value: f"{value:.4f}" if col != "delta_macro_f1" else f"{value:+.4f}")
    contribution_table = contribution.copy()
    for col in ["accuracy_drop", "macro_f1_drop", "weighted_f1_drop"]:
        contribution_table[col] = contribution_table[col].map(lambda value: f"{value:+.4f}")

    report = f"""# 实验五消融实验报告

## 实验目的

本消融实验用于分析实验五深度学习方案中各个模块的实际贡献。完整模型由 45 维核心特征、全局 Residual DNN 集成、监督对比学习、年代专家 DNN 和 ExtraTrees 辅助概率融合组成。消融实验通过逐项去掉或替换某个模块，观察 Accuracy、Macro F1 和 Weighted F1 的变化，从而判断该模块是否对最终性能有正向作用。

本报告重点回答两个问题：

1. 当前模型中的哪些模块确实提高了测试集指标；
2. 哪些模块贡献较小，只能作为轻微增强，不能夸大其作用。

## 实验设置

- 数据集：`NBA_Season_Stats.csv` 预处理后的同一份训练集和测试集
- 训练集：`nba_clean_train.csv`
- 测试集：`nba_clean_test.csv`
- 分类目标：`C / PF / PG / SF / SG`
- 设备：`{DEVICE}`
- 随机种子：`{RANDOM_STATE}`
- 完整模型 Accuracy：{full_row['accuracy']:.4f}
- 完整模型 Macro F1：{full_row['macro_f1']:.4f}
- 完整模型 Weighted F1：{full_row['weighted_f1']:.4f}

除被消融的模块外，其余训练参数尽量保持一致，包括 batch size、学习率、残差网络宽度、训练轮数、label smoothing 和固定数据划分。A0、A6、A8 可直接由正式实验五输出推导，其他版本重新训练得到。

## 消融版本设计

| 版本 | 消融内容 |
|---|---|
| A0_full_model | 完整模型，作为基准 |
| A1_all_76_features | 取消特征筛选，使用全部 76 个候选特征 |
| A2_raw_26_features | 只使用 26 个原始数值特征，去掉全部工程特征 |
| A3_no_contrastive_loss | 去掉监督对比学习损失 |
| A4_no_contrastive_members | 去掉 Contrastive Residual DNN 成员，只保留普通 Residual DNN |
| A5_no_era_experts | 去掉年代专家 DNN |
| A6_no_auxiliary_tree | 去掉 ExtraTrees 辅助概率融合 |
| A7_single_residual_dnn | 只使用单个普通 Residual DNN |
| A8_auxiliary_tree_only | 只使用 ExtraTrees，不使用 DNN |

## 汇总结果

{table.to_markdown(index=False)}

## 模块贡献排序

下表以完整模型 A0 为基准，统计去掉或替换某个模块后指标下降多少。数值越大，说明该模块对当前方案越关键。

{contribution_table.to_markdown(index=False)}

## 可视化结果

![Ablation Macro F1](ablation_macro_f1.png)

![Ablation Delta Macro F1](ablation_delta_macro_f1.png)

## 结果分析

### 特征工程与特征筛选

完整模型 A0 使用 45 维核心特征，Accuracy 为 {by_name.loc['A0_full_model', 'accuracy']:.4f}，Macro F1 为 {by_name.loc['A0_full_model', 'macro_f1']:.4f}。A2 只使用 26 维原始特征，Macro F1 降至 {by_name.loc['A2_raw_26_features', 'macro_f1']:.4f}，相比 A0 下降 {full_row['macro_f1'] - by_name.loc['A2_raw_26_features', 'macro_f1']:.4f}。这说明工程特征是比较重要的性能来源之一，每场效率、每 36 分钟效率、投篮结构、组织/篮板/防守倾向等派生指标能提供更直接的位置线索。

A1 使用全部 76 维候选特征，Macro F1 为 {by_name.loc['A1_all_76_features', 'macro_f1']:.4f}，相比 A0 下降 {full_row['macro_f1'] - by_name.loc['A1_all_76_features', 'macro_f1']:.4f}。这个下降很小，但方向说明 45 维核心特征筛选略有价值：它删除了一部分重复或贡献弱的派生特征，使模型更简洁，且没有损失性能。

### 监督对比学习

A3 去掉监督对比学习损失，Macro F1 为 {by_name.loc['A3_no_contrastive_loss', 'macro_f1']:.4f}，相比 A0 下降 {full_row['macro_f1'] - by_name.loc['A3_no_contrastive_loss', 'macro_f1']:.4f}。下降幅度不大，说明监督对比学习属于轻量增强模块。它的作用不是单独大幅提高准确率，而是在 PF/SF、PG/SG、C/PF 这类边界较模糊的位置上提供额外的隐空间约束。

### 集成与对比成员

A4 去掉 Contrastive Residual DNN 成员，仅保留普通 Residual DNN，Macro F1 为 {by_name.loc['A4_no_contrastive_members', 'macro_f1']:.4f}，相比 A0 下降 {full_row['macro_f1'] - by_name.loc['A4_no_contrastive_members', 'macro_f1']:.4f}。这说明 Contrastive 成员带来了有价值的模型多样性。

A7 进一步退化为单个普通 Residual DNN，Macro F1 为 {by_name.loc['A7_single_residual_dnn', 'macro_f1']:.4f}，相比 A0 下降 {full_row['macro_f1'] - by_name.loc['A7_single_residual_dnn', 'macro_f1']:.4f}。这是较大的降幅，说明单模型容易受初始化和局部边界影响，而多模型集成能够显著稳定概率输出。

### 年代专家

A5 去掉年代专家 DNN，Macro F1 为 {by_name.loc['A5_no_era_experts', 'macro_f1']:.4f}，相比 A0 下降 {full_row['macro_f1'] - by_name.loc['A5_no_era_experts', 'macro_f1']:.4f}。这是所有结构模块中较明显的下降。NBA 不同年代的三分出手、位置职责和内外线分工不同，年代专家能够学习更局部的统计边界，因此对最终结果有实质贡献。

### ExtraTrees 辅助概率融合

A6 去掉 ExtraTrees 辅助概率融合，只保留 DNN 融合概率，Macro F1 为 {by_name.loc['A6_no_auxiliary_tree', 'macro_f1']:.4f}，相比 A0 下降 {full_row['macro_f1'] - by_name.loc['A6_no_auxiliary_tree', 'macro_f1']:.4f}。A8 只使用 ExtraTrees，Macro F1 为 {by_name.loc['A8_auxiliary_tree_only', 'macro_f1']:.4f}，明显低于完整模型。这说明 ExtraTrees 并不是单独更强的模型，而是作为与 DNN 错误模式不同的互补概率源，在融合后带来提升。

### 单模型、树模型与完整模型对比

A7 单个 Residual DNN 的 Macro F1 为 {by_name.loc['A7_single_residual_dnn', 'macro_f1']:.4f}，A8 单独 ExtraTrees 的 Macro F1 为 {by_name.loc['A8_auxiliary_tree_only', 'macro_f1']:.4f}，二者都明显低于 A0。这说明最终方案不是依赖某个单独模型，而是多个互补模块组合后的结果。DNN 负责学习非线性连续边界，年代专家吸收年份风格差异，ExtraTrees 作为表格模型补充部分非神经网络划分。

## 结论

本次消融实验中，最佳版本为 `{best_row['name']}`，Macro F1 为 {best_row['macro_f1']:.4f}。完整模型 A0 的 Macro F1 为 {full_row['macro_f1']:.4f}。从降幅看，贡献最明显的部分是工程特征、DNN集成、年代专家和DNN本身；ExtraTrees融合、Contrastive成员和监督对比损失也有正向作用，但提升幅度较小。

需要注意的是，A1、A3 等版本与 A0 的差距很小，只能说明它们对当前测试集有轻微影响，不能夸大为显著提升。总体结论是：实验五的最终性能来自多个模块的累积效果，其中工程特征和集成结构贡献最大，年代专家提供明显增益，辅助概率融合与对比学习提供进一步微调。
"""
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 5 DNN ablation study.")
    parser.add_argument("--force", action="store_true", help="Re-run ablations even if cached metrics exist.")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for config in ABLATION_CONFIGS:
        print(f"Running {config.name}: {config.description}")
        if config.source == "current_metrics":
            metrics = copy_current_full(force=args.force)
        elif config.source == "derive_current":
            metrics = derive_from_current(config, force=args.force)
        else:
            metrics = run_training_ablation(config, force=args.force)
        metrics_rows.append(metrics)
        print(f"{config.name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")

    summary = pd.DataFrame(metrics_rows)
    full_macro_f1 = float(summary.loc[summary["name"] == "A0_full_model", "macro_f1"].iloc[0])
    full_accuracy = float(summary.loc[summary["name"] == "A0_full_model", "accuracy"].iloc[0])
    summary["delta_macro_f1"] = summary["macro_f1"] - full_macro_f1
    summary["delta_accuracy"] = summary["accuracy"] - full_accuracy
    summary = summary.sort_values("name")
    summary.to_csv(RESULT_DIR / "ablation_results.csv", index=False, encoding="utf-8")
    (RESULT_DIR / "ablation_results.json").write_text(summary.to_json(orient="records", force_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_ablation_results(summary)
    write_report(summary)
    print(f"Ablation study completed. Result directory: {RESULT_DIR}")


if __name__ == "__main__":
    main()
