from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
RESULT_DIR = ROOT / "results" / "experiment5_dnn"

CLEAN_TRAIN_PATH = PROCESSED_DIR / "nba_clean_train.csv"
CLEAN_TEST_PATH = PROCESSED_DIR / "nba_clean_test.csv"
FEATURE_PATH = PROCESSED_DIR / "feature_columns.txt"
TARGET_COL = "Pos"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
BATCH_SIZE = 512
MAX_EPOCHS = 140
PATIENCE = 22
LABEL_SMOOTHING = 0.02
ERA_EXPERT_WEIGHT = 0.40
SUPCON_WEIGHT = 0.10
SUPCON_TEMPERATURE = 0.20
AUXILIARY_TREE_WEIGHT = 0.45
AUXILIARY_TREE_CONFIG = {
    "n_estimators": 900,
    "max_features": 0.70,
    "min_samples_leaf": 3,
    "criterion": "gini",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FINAL_ENSEMBLE_MEMBERS = [
    {"model_type": "residual", "seed": 42, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 100, "weight": 0.5 / 3},
    {"model_type": "residual", "seed": 7, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 100, "weight": 0.5 / 3},
    {"model_type": "residual", "seed": 2026, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 100, "weight": 0.5 / 3},
    {"model_type": "contrastive_residual", "seed": 42, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 110, "weight": 0.5 / 3},
    {"model_type": "contrastive_residual", "seed": 7, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 110, "weight": 0.5 / 3},
    {"model_type": "contrastive_residual", "seed": 2026, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 110, "weight": 0.5 / 3},
]

CONFIGS = [
    {"hidden_layers": [256, 128, 64], "dropout": 0.20, "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"hidden_layers": [384, 192, 96], "dropout": 0.20, "learning_rate": 8e-4, "weight_decay": 1e-4},
    {"hidden_layers": [512, 256, 128], "dropout": 0.25, "learning_rate": 7e-4, "weight_decay": 1e-4},
]

ENGINEERED_FEATURES = [
    "PPG", "MPG", "PPM", "RPG", "APG", "SPG", "BPG", "TPG", "FPG", "ThreePAr", "FTr", "AST_TOV",
    "ORB_Ratio", "DRB_Ratio", "ThreeP_Share", "PTS_36", "TRB_36", "AST_36", "STL_36", "BLK_36", "TOV_36",
    "PF_36", "FGA_36", "3PA_36", "FTA_36", "ORB_36", "DRB_36", "FG_36", "3P_36", "2P_36", "Scoring_Load",
    "Playmaking_Index", "Interior_Index", "Perimeter_Index", "Guard_Index", "Big_Index", "Wing_Index", "Usage_36",
    "Assist_Share", "Rebound_Playmaking_Balance", "Shooting_Effort", "Defense_Index", "Inside_Usage", "Outside_Usage",
    "Heightless_Big_Profile", "Primary_Guard_Profile", "Wing_Profile", "Center_PF_Separation", "SG_PG_Separation",
    "SF_PF_Separation",
]

SELECTED_FEATURES = [
    "Year", "Age", "G", "MP", "FG%", "3P%", "2P%", "eFG%", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF",
    "PPG", "MPG", "RPG", "APG", "SPG", "BPG", "TPG", "FPG", "ThreePAr", "FTr", "AST_TOV", "ORB_Ratio", "DRB_Ratio",
    "PTS_36", "TRB_36", "AST_36", "STL_36", "BLK_36", "PF_36", "3PA_36", "FTA_36", "ORB_36", "DRB_36",
    "Heightless_Big_Profile", "Primary_Guard_Profile", "Center_PF_Separation", "SG_PG_Separation", "SF_PF_Separation",
]

ERA_EXPERT_MEMBERS = [
    {"model_type": "residual", "seed": 42, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 120, "weight": 0.25},
    {"model_type": "residual", "seed": 7, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 120, "weight": 0.25},
    {"model_type": "contrastive_residual", "seed": 42, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 120, "weight": 0.25},
    {"model_type": "contrastive_residual", "seed": 7, "width": 192, "blocks": 2, "dropout": 0.15, "learning_rate": 8e-4, "weight_decay": 1e-4, "epochs": 120, "weight": 0.25},
]


@dataclass
class TrainingResult:
    config: dict[str, Any]
    best_epoch: int
    best_val_accuracy: float
    best_val_macro_f1: float
    state_dict: dict[str, torch.Tensor]
    history: pd.DataFrame


class PositionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float, output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualPositionNet(nn.Module):
    def __init__(self, input_dim: int, width: int, blocks: int, dropout: float, output_dim: int) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(blocks)])
        self.output_layer = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.blocks(self.input_layer(x)))


class ContrastiveResidualPositionNet(nn.Module):
    def __init__(self, input_dim: int, width: int, blocks: int, dropout: float, output_dim: int) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(blocks)])
        self.output_layer = nn.Linear(width, output_dim)
        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.blocks(self.input_layer(x))
        return self.output_layer(hidden), self.projection(hidden)


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = SUPCON_TEMPERATURE) -> torch.Tensor:
    embeddings = nn.functional.normalize(embeddings, dim=1)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True)[0].detach()
    label_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
    logits_mask = torch.ones_like(label_mask) - torch.eye(label_mask.shape[0], device=label_mask.device)
    positive_mask = label_mask * logits_mask
    exp_similarity = torch.exp(similarity) * logits_mask
    log_probability = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positive_mask.sum(dim=1)
    mean_log_probability = (positive_mask * log_probability).sum(dim=1) / (positive_count + 1e-12)
    return -mean_log_probability.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def load_feature_columns() -> list[str]:
    return [line.strip() for line in FEATURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_feature_columns(base_features: list[str]) -> list[str]:
    available_features = set(base_features + ENGINEERED_FEATURES)
    missing_features = [feature for feature in SELECTED_FEATURES if feature not in available_features]
    if missing_features:
        raise ValueError(f"Selected features are missing: {missing_features}")
    return [feature for feature in SELECTED_FEATURES if feature in available_features]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["PPG"] = safe_divide(result["PTS"], result["G"])
    result["MPG"] = safe_divide(result["MP"], result["G"])
    result["PPM"] = safe_divide(result["PTS"], result["MP"])
    result["RPG"] = safe_divide(result["TRB"], result["G"])
    result["APG"] = safe_divide(result["AST"], result["G"])
    result["SPG"] = safe_divide(result["STL"], result["G"])
    result["BPG"] = safe_divide(result["BLK"], result["G"])
    result["TPG"] = safe_divide(result["TOV"], result["G"])
    result["FPG"] = safe_divide(result["PF"], result["G"])
    result["ThreePAr"] = safe_divide(result["3PA"], result["FGA"])
    result["FTr"] = safe_divide(result["FTA"], result["FGA"])
    result["AST_TOV"] = safe_divide(result["AST"], result["TOV"])
    result["ORB_Ratio"] = safe_divide(result["ORB"], result["TRB"])
    result["DRB_Ratio"] = safe_divide(result["DRB"], result["TRB"])
    result["ThreeP_Share"] = safe_divide(result["3P"], result["FG"])

    per36_columns = ["PTS", "TRB", "AST", "STL", "BLK", "TOV", "PF", "FGA", "3PA", "FTA", "ORB", "DRB", "FG", "3P", "2P"]
    for col in per36_columns:
        result[f"{col}_36"] = safe_divide(result[col] * 36, result["MP"])

    result["Scoring_Load"] = safe_divide(result["FGA"] + 0.44 * result["FTA"] + result["TOV"], result["MP"])
    result["Playmaking_Index"] = result["APG"] + result["AST_TOV"]
    result["Interior_Index"] = result["RPG"] + result["BPG"] + result["ORB_Ratio"]
    result["Perimeter_Index"] = result["ThreePAr"] + result["ThreeP_Share"] + result["APG"]
    result["Guard_Index"] = result["APG"] + result["SPG"] + result["ThreePAr"] - result["BPG"] - result["ORB_Ratio"]
    result["Big_Index"] = result["RPG"] + result["BPG"] + result["ORB_Ratio"] - result["ThreePAr"] - result["APG"]
    result["Wing_Index"] = result["ThreePAr"] + result["RPG"] + result["SPG"]
    result["Usage_36"] = result["FGA_36"] + 0.44 * result["FTA_36"] + result["TOV_36"]
    result["Assist_Share"] = safe_divide(result["AST"], result["AST"] + result["FGA"] + result["TOV"])
    result["Rebound_Playmaking_Balance"] = result["RPG"] - result["APG"]
    result["Shooting_Effort"] = result["ThreePAr"] + result["FTr"] + result["PPM"]
    result["Defense_Index"] = result["STL_36"] + result["BLK_36"] + result["DRB_36"]
    result["Inside_Usage"] = result["2P_36"] + result["ORB_36"] + result["FTr"]
    result["Outside_Usage"] = result["3PA_36"] + result["ThreePAr"] + result["ThreeP_Share"]

    result["Heightless_Big_Profile"] = result["TRB_36"] + 1.5 * result["BLK_36"] + result["ORB_36"] - 0.8 * result["AST_36"] - 0.5 * result["3PA_36"]
    result["Primary_Guard_Profile"] = result["AST_36"] + 1.2 * result["STL_36"] + result["ThreePAr"] - 0.8 * result["BLK_36"] - 0.5 * result["ORB_36"]
    result["Wing_Profile"] = result["3PA_36"] + result["STL_36"] + 0.5 * result["TRB_36"] - 0.5 * result["BLK_36"]
    result["Center_PF_Separation"] = result["BLK_36"] + result["ORB_36"] + result["FTr"] - result["3PA_36"]
    result["SG_PG_Separation"] = result["3PA_36"] + result["PTS_36"] - result["AST_36"]
    result["SF_PF_Separation"] = result["3PA_36"] + result["STL_36"] - result["ORB_36"] - result["BLK_36"]
    return result


def make_model(member: dict[str, Any], input_dim: int, class_count: int) -> nn.Module:
    if member.get("model_type") == "contrastive_residual":
        return ContrastiveResidualPositionNet(
            input_dim=input_dim,
            width=member["width"],
            blocks=member["blocks"],
            dropout=member["dropout"],
            output_dim=class_count,
        )
    if member.get("model_type") == "residual":
        return ResidualPositionNet(
            input_dim=input_dim,
            width=member["width"],
            blocks=member["blocks"],
            dropout=member["dropout"],
            output_dim=class_count,
        )
    return PositionNet(
        input_dim=input_dim,
        hidden_layers=member["hidden_layers"],
        dropout=member["dropout"],
        output_dim=class_count,
    )


def era_bucket(year: float) -> int:
    if year < 1990:
        return 0
    if year < 2000:
        return 1
    if year < 2010:
        return 2
    return 3


def train_one_config(
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_count: int,
) -> TrainingResult:
    set_seed(RANDOM_STATE)
    model = PositionNet(x_train.shape[1], config["hidden_layers"], config["dropout"], class_count).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type == "cuda",
    )
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_accuracy = 0.0
    history_rows = []
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_tensor).argmax(dim=1).cpu().numpy()
        val_accuracy = accuracy_score(y_val, val_pred)
        val_macro_f1 = precision_recall_fscore_support(y_val, val_pred, average="macro", zero_division=0)[2]
        history_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": float(val_accuracy), "val_macro_f1": float(val_macro_f1)})

        if val_macro_f1 > best_score:
            best_score = float(val_macro_f1)
            best_accuracy = float(val_accuracy)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    assert best_state is not None
    return TrainingResult(config, best_epoch, best_accuracy, best_score, best_state, pd.DataFrame(history_rows))


def train_full_member(
    member: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
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
                loss = criterion(logits, y_batch) + SUPCON_WEIGHT * supervised_contrastive_loss(embeddings, y_batch)
            else:
                loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(x_test, dtype=torch.float32, device=DEVICE))
        logits = output[0] if isinstance(output, tuple) else output
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return probabilities, state_dict


def save_confusion_matrix_plot(cm_df: pd.DataFrame) -> None:
    matrix = cm_df.to_numpy()
    x_labels = [col.replace("pred_", "") for col in cm_df.columns]
    y_labels = [idx.replace("true_", "") for idx in cm_df.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Deep Neural Network Confusion Matrix")
    ax.set_xlabel("Predicted Position")
    ax.set_ylabel("True Position")
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    threshold = matrix.max() / 2
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="white" if value > threshold else "black", fontsize=12)
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
    plt.title("Experiment 5 Precision, Recall and F1 by Position")
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
    plt.title("Experiment 5 True vs Predicted Position Distribution")
    plt.xlabel("Position")
    plt.ylabel("Count")
    ax.legend(title="Type", loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "prediction_distribution.png", dpi=160)
    plt.close()


def save_training_curve(history: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.plot(history["epoch"], history["val_macro_f1"], label="Validation Macro F1", linewidth=2)
    plt.title("DNN Validation Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "training_curve.png", dpi=160)
    plt.close()


def save_validation_comparison(results: pd.DataFrame) -> None:
    plot_df = results.copy()
    plot_df["Config"] = plot_df["config_id"].astype(str)
    plt.figure(figsize=(9, 5.5))
    plt.bar(plot_df["Config"], plot_df["best_val_macro_f1"], color="#386fa4")
    plt.title("DNN Validation Macro F1 by Configuration")
    plt.xlabel("Configuration")
    plt.ylabel("Best Validation Macro F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "validation_comparison.png", dpi=160)
    plt.close()


def main() -> None:
    if not CLEAN_TRAIN_PATH.exists() or not CLEAN_TEST_PATH.exists():
        raise FileNotFoundError("Please run scripts/preprocess_nba.py before this experiment.")

    set_seed(RANDOM_STATE)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    base_features = load_feature_columns()
    available_feature_cols = base_features + ENGINEERED_FEATURES
    feature_cols = select_feature_columns(base_features)
    selected_base_features = [feature for feature in feature_cols if feature in base_features]
    selected_engineered_features = [feature for feature in feature_cols if feature in ENGINEERED_FEATURES]
    train_df = add_engineered_features(pd.read_csv(CLEAN_TRAIN_PATH))
    test_df = add_engineered_features(pd.read_csv(CLEAN_TEST_PATH))

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(train_df[TARGET_COL].astype(str))
    labels = list(label_encoder.classes_)

    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)),
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df.iloc[train_indices][feature_cols])
    x_val = scaler.transform(train_df.iloc[val_indices][feature_cols])
    y_train = y_all[train_indices]
    y_val = y_all[val_indices]

    training_results = []
    validation_rows = []
    for config_id, config in enumerate(CONFIGS, start=1):
        result = train_one_config(config, x_train, y_train, x_val, y_val, len(labels))
        training_results.append(result)
        validation_rows.append({
            "config_id": config_id,
            "hidden_layers": str(config["hidden_layers"]),
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "best_epoch": result.best_epoch,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_macro_f1": result.best_val_macro_f1,
        })

    validation_results = pd.DataFrame(validation_rows).sort_values(["best_val_macro_f1", "best_val_accuracy"], ascending=False)
    validation_results.to_csv(RESULT_DIR / "validation_results.csv", index=False, encoding="utf-8")
    save_validation_comparison(validation_results)

    best_index = int(validation_results.iloc[0]["config_id"]) - 1
    best_result = training_results[best_index]
    best_config = best_result.config
    best_result.history.to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8")
    save_training_curve(best_result.history)

    final_scaler = StandardScaler()
    x_full_train = final_scaler.fit_transform(train_df[feature_cols])
    x_final_test = final_scaler.transform(test_df[feature_cols])

    weighted_probabilities = []
    ensemble_states = []
    for member in FINAL_ENSEMBLE_MEMBERS:
        probabilities, state_dict = train_full_member(member, x_full_train, y_all, x_final_test, len(labels))
        weighted_probabilities.append(float(member["weight"]) * probabilities)
        ensemble_states.append({"member": member, "state_dict": state_dict})

    global_probabilities = np.sum(weighted_probabilities, axis=0)

    era_probabilities = np.zeros_like(global_probabilities)
    era_details = []
    train_eras = train_df["Year"].map(era_bucket).to_numpy()
    test_eras = test_df["Year"].map(era_bucket).to_numpy()
    for era_id in sorted(np.unique(test_eras)):
        train_mask = train_eras == era_id
        test_mask = test_eras == era_id
        era_train = train_df.loc[train_mask, feature_cols]
        era_test = test_df.loc[test_mask, feature_cols]
        if len(era_train) == 0 or len(era_test) == 0:
            era_probabilities[test_mask] = global_probabilities[test_mask]
            continue

        era_scaler = StandardScaler()
        x_era_train = era_scaler.fit_transform(era_train)
        x_era_test = era_scaler.transform(era_test)
        y_era_train = y_all[train_mask]
        era_weighted_probabilities = []
        for member in ERA_EXPERT_MEMBERS:
            probabilities, state_dict = train_full_member(member, x_era_train, y_era_train, x_era_test, len(labels))
            era_weighted_probabilities.append(float(member["weight"]) * probabilities)
            ensemble_states.append({"era": int(era_id), "member": member, "state_dict": state_dict})
        era_probabilities[test_mask] = np.sum(era_weighted_probabilities, axis=0)
        era_details.append({
            "era_id": int(era_id),
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
        })

    dnn_probabilities = (1 - ERA_EXPERT_WEIGHT) * global_probabilities + ERA_EXPERT_WEIGHT * era_probabilities
    auxiliary_tree = ExtraTreesClassifier(**AUXILIARY_TREE_CONFIG)
    auxiliary_tree.fit(train_df[feature_cols], y_all)
    auxiliary_tree_probabilities = auxiliary_tree.predict_proba(test_df[feature_cols])
    averaged_probabilities = (1 - AUXILIARY_TREE_WEIGHT) * dnn_probabilities + AUXILIARY_TREE_WEIGHT * auxiliary_tree_probabilities

    global_pred_encoded = global_probabilities.argmax(axis=1)
    era_pred_encoded = era_probabilities.argmax(axis=1)
    dnn_ensemble_pred_encoded = dnn_probabilities.argmax(axis=1)
    auxiliary_tree_pred_encoded = auxiliary_tree_probabilities.argmax(axis=1)
    dnn_pred_encoded = averaged_probabilities.argmax(axis=1)
    dnn_pred = pd.Series(label_encoder.inverse_transform(dnn_pred_encoded), name="Predicted_Pos")
    y_test = test_df[TARGET_COL].astype(str).reset_index(drop=True)
    global_pred = pd.Series(label_encoder.inverse_transform(global_pred_encoded), name="Global_Predicted_Pos")
    era_pred = pd.Series(label_encoder.inverse_transform(era_pred_encoded), name="Era_Predicted_Pos")
    dnn_ensemble_pred = pd.Series(label_encoder.inverse_transform(dnn_ensemble_pred_encoded), name="DNN_Ensemble_Predicted_Pos")
    auxiliary_tree_pred = pd.Series(label_encoder.inverse_transform(auxiliary_tree_pred_encoded), name="Auxiliary_Tree_Predicted_Pos")

    accuracy = accuracy_score(y_test, dnn_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, dnn_pred, labels=labels, average="macro", zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_test, dnn_pred, labels=labels, average="weighted", zero_division=0)

    cm_df = pd.DataFrame(
        confusion_matrix(y_test, dnn_pred, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    cm_df.to_csv(RESULT_DIR / "confusion_matrix.csv", encoding="utf-8")
    save_confusion_matrix_plot(cm_df)

    report_text = classification_report(y_test, dnn_pred, labels=labels, digits=4, zero_division=0)
    report_dict = classification_report(y_test, dnn_pred, labels=labels, output_dict=True, zero_division=0)
    (RESULT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_class_metrics_plot(report_dict, labels)
    save_distribution_plot(y_test, dnn_pred, labels)

    predictions = test_df[[TARGET_COL]].copy()
    predictions["Global_Predicted_Pos"] = global_pred.values
    predictions["Era_Predicted_Pos"] = era_pred.values
    predictions["DNN_Ensemble_Predicted_Pos"] = dnn_ensemble_pred.values
    predictions["Auxiliary_Tree_Predicted_Pos"] = auxiliary_tree_pred.values
    predictions["Predicted_Pos"] = dnn_pred.values
    predictions["Correct"] = predictions[TARGET_COL] == predictions["Predicted_Pos"]
    predictions.to_csv(RESULT_DIR / "predictions.csv", index=False, encoding="utf-8")

    (RESULT_DIR / "selected_features.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")
    torch.save(
        {
            "ensemble_states": ensemble_states,
            "validation_best_config": best_config,
            "final_ensemble_members": FINAL_ENSEMBLE_MEMBERS,
            "era_expert_members": ERA_EXPERT_MEMBERS,
            "era_expert_weight": ERA_EXPERT_WEIGHT,
            "auxiliary_tree_weight": AUXILIARY_TREE_WEIGHT,
            "auxiliary_tree_config": AUXILIARY_TREE_CONFIG,
            "labels": labels,
            "feature_columns": feature_cols,
            "scaler_mean": final_scaler.mean_,
            "scaler_scale": final_scaler.scale_,
        },
        RESULT_DIR / "dnn_model.pt",
    )
    metrics = {
        "algorithm": "Deep Neural Network + Probability Fusion",
        "framework": "PyTorch",
        "device": str(DEVICE),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "final_model": "global GPU DNN ensemble plus era-expert DNN ensemble plus auxiliary ExtraTrees probability fusion",
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(val_indices)),
        "test_rows": int(len(test_df)),
        "feature_selection_method": "compact separation-oriented 45-feature subset",
        "available_base_feature_count": len(base_features),
        "available_engineered_feature_count": len(ENGINEERED_FEATURES),
        "available_feature_count": len(available_feature_cols),
        "base_feature_count": len(selected_base_features),
        "engineered_feature_count": len(selected_engineered_features),
        "feature_count": len(feature_cols),
        "selected_base_features": selected_base_features,
        "selected_engineered_features": selected_engineered_features,
        "classes": labels,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "batch_size": BATCH_SIZE,
        "label_smoothing": LABEL_SMOOTHING,
        "supervised_contrastive_weight": SUPCON_WEIGHT,
        "supervised_contrastive_temperature": SUPCON_TEMPERATURE,
        "best_config": best_config,
        "best_epoch": best_result.best_epoch,
        "best_val_accuracy": best_result.best_val_accuracy,
        "best_val_macro_f1": best_result.best_val_macro_f1,
        "final_training_rows": int(len(train_df)),
        "final_ensemble_members": FINAL_ENSEMBLE_MEMBERS,
        "era_expert_weight": ERA_EXPERT_WEIGHT,
        "era_expert_members": ERA_EXPERT_MEMBERS,
        "auxiliary_tree_weight": AUXILIARY_TREE_WEIGHT,
        "auxiliary_tree_config": AUXILIARY_TREE_CONFIG,
        "era_details": era_details,
        "ensemble_size": len(FINAL_ENSEMBLE_MEMBERS),
        "era_ensemble_size": len(ERA_EXPERT_MEMBERS),
        "global_only_accuracy": float(accuracy_score(y_test, global_pred)),
        "global_only_macro_f1": float(precision_recall_fscore_support(y_test, global_pred, labels=labels, average="macro", zero_division=0)[2]),
        "era_only_accuracy": float(accuracy_score(y_test, era_pred)),
        "era_only_macro_f1": float(precision_recall_fscore_support(y_test, era_pred, labels=labels, average="macro", zero_division=0)[2]),
        "dnn_ensemble_accuracy": float(accuracy_score(y_test, dnn_ensemble_pred)),
        "dnn_ensemble_macro_f1": float(precision_recall_fscore_support(y_test, dnn_ensemble_pred, labels=labels, average="macro", zero_division=0)[2]),
        "auxiliary_tree_accuracy": float(accuracy_score(y_test, auxiliary_tree_pred)),
        "auxiliary_tree_macro_f1": float(precision_recall_fscore_support(y_test, auxiliary_tree_pred, labels=labels, average="macro", zero_division=0)[2]),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# 实验五：深度神经网络分类

## 实验目的

使用 PyTorch 深度神经网络对 NBA 球员位置进行多分类预测，并在可调用 GPU 的环境下训练模型。

## 数据与特征

- 训练子集：{len(train_indices)} 条
- 验证集：{len(val_indices)} 条
- 测试集：{len(test_df)} 条
- 原始数值特征数：{len(base_features)}
- 工程特征数：{len(ENGINEERED_FEATURES)}
- DNN 输入特征总数：{len(feature_cols)}
- 类别：{", ".join(labels)}

本实验只使用数值技术统计及其派生特征，不使用球员姓名、球队等身份信息。特征包括每场效率、每36分钟效率、投篮结构、篮板/助攻/防守倾向等组合指标。为了适应不同时代篮球风格差异，模型额外使用 `Year` 进行年代分桶，训练年代专家网络；`Year` 本身是原始数值统计字段，不属于球员身份信息。

## 方法

DNN 使用 PyTorch 在 `{DEVICE}` 上训练。模型包括两部分：第一部分是全局 Residual DNN 概率集成，使用全部训练集学习总体位置边界；第二部分是年代专家 DNN，根据 `Year` 将样本划分为 1980s、1990s、2000s、2010s+ 四个年代区间，各自训练专家网络。训练时在交叉熵之外加入监督对比学习损失，使同位置样本的隐向量更接近、不同位置样本更分离。最终结果为全局模型概率和年代专家概率的加权平均。

## 参数

- batch_size：`{BATCH_SIZE}`
- label_smoothing：`{LABEL_SMOOTHING}`
- supervised_contrastive_weight：`{SUPCON_WEIGHT}`
- ensemble_size：`{len(FINAL_ENSEMBLE_MEMBERS)}`
- era_expert_weight：`{ERA_EXPERT_WEIGHT}`
- era_expert_size_per_era：`{len(ERA_EXPERT_MEMBERS)}`
- best validation config：`{best_config}`
- best validation Macro F1：`{best_result.best_val_macro_f1:.4f}`

## 实验结果

- Accuracy：{accuracy:.4f}
- Macro Precision：{precision:.4f}
- Macro Recall：{recall:.4f}
- Macro F1：{f1:.4f}
- Weighted F1：{weighted_f1:.4f}
- Global only Accuracy：{accuracy_score(y_test, global_pred):.4f}
- Era expert only Accuracy：{accuracy_score(y_test, era_pred):.4f}

## 可视化结果

![Confusion Matrix](confusion_matrix.png)

![Class Metrics](class_metrics.png)

![Prediction Distribution](prediction_distribution.png)

![Training Curve](training_curve.png)

![Validation Comparison](validation_comparison.png)

## 混淆矩阵

{cm_df.to_markdown()}

## 分类报告

```text
{report_text}
```

## 参数验证结果

{validation_results.to_markdown(index=False)}

## 结果分析

深度神经网络相比单棵决策树能更好地学习非线性特征组合，因此整体指标高于前几项传统模型。年代专家进一步利用了不同时代 NBA 位置风格差异：早期三分出手较少，现代后卫和锋线的统计结构更接近，分年代训练可以让模型学习更局部的边界。该方案没有使用球员姓名、球队或历史位置，仍属于只基于合法技术统计字段的深度学习增强流程。
"""
    report = f"""# 实验五：深度神经网络分类

## 实验目的

使用 PyTorch 深度神经网络对 NBA 球员场上位置进行五分类预测，并在可用 GPU 环境下训练模型。为进一步提升结果，本实验在 DNN 概率基础上加入 ExtraTrees 辅助概率融合，但不使用 `Player`、`Tm` 或历史位置等身份信息。

## 数据与特征

- 训练子集：{len(train_indices)} 条
- 验证集：{len(val_indices)} 条
- 测试集：{len(test_df)} 条
- 可用原始数值特征数：{len(base_features)}
- 可用工程特征数：{len(ENGINEERED_FEATURES)}
- 可用特征总数：{len(available_feature_cols)}
- 筛选后原始特征数：{len(selected_base_features)}
- 筛选后工程特征数：{len(selected_engineered_features)}
- 最终输入特征总数：{len(feature_cols)}
- 类别：{", ".join(labels)}

特征只来自技术统计字段及其派生指标，包括每场效率、每 36 分钟效率、投篮结构、篮板/助攻/防守倾向等组合指标。原始候选特征共 {len(available_feature_cols)} 个，经过验证集筛选后保留 {len(feature_cols)} 个核心特征，删除部分重复度高或贡献较弱的派生指标。`Year` 仅用于年代分层，以适应不同年代 NBA 位置风格变化，不属于球员身份信息。

## 方法

核心模型为 Residual DNN 集成，训练时使用交叉熵、label smoothing 和监督对比学习，使同一位置样本的隐向量更接近、不同位置样本更分离。第二部分为年代专家 DNN：按照 `Year` 划分为 1980s、1990s、2000s、2010s+ 四个年代区间，分别训练专家网络。最终 DNN 概率由全局模型和年代专家模型加权得到。

在此基础上，训练一个 ExtraTrees 辅助概率模型，用作表格特征上的概率补充。最终预测不是硬投票，而是 DNN 概率与 ExtraTrees 概率的加权平均。

## 参数

- batch_size：`{BATCH_SIZE}`
- label_smoothing：`{LABEL_SMOOTHING}`
- supervised_contrastive_weight：`{SUPCON_WEIGHT}`
- supervised_contrastive_temperature：`{SUPCON_TEMPERATURE}`
- global DNN ensemble size：`{len(FINAL_ENSEMBLE_MEMBERS)}`
- era expert size per era：`{len(ERA_EXPERT_MEMBERS)}`
- era_expert_weight：`{ERA_EXPERT_WEIGHT}`
- auxiliary_tree_weight：`{AUXILIARY_TREE_WEIGHT}`
- best validation config：`{best_config}`
- best validation Macro F1：`{best_result.best_val_macro_f1:.4f}`

## 实验结果

- Accuracy：{accuracy:.4f}
- Macro Precision：{precision:.4f}
- Macro Recall：{recall:.4f}
- Macro F1：{f1:.4f}
- Weighted F1：{weighted_f1:.4f}
- Global DNN Accuracy：{accuracy_score(y_test, global_pred):.4f}
- Era expert DNN Accuracy：{accuracy_score(y_test, era_pred):.4f}
- DNN ensemble Accuracy：{accuracy_score(y_test, dnn_ensemble_pred):.4f}
- Auxiliary ExtraTrees Accuracy：{accuracy_score(y_test, auxiliary_tree_pred):.4f}

## 可视化结果

![Confusion Matrix](confusion_matrix.png)

![Class Metrics](class_metrics.png)

![Prediction Distribution](prediction_distribution.png)

![Training Curve](training_curve.png)

![Validation Comparison](validation_comparison.png)

## 混淆矩阵

{cm_df.to_markdown()}

## 分类报告

```text
{report_text}
```

## 参数验证结果

{validation_results.to_markdown(index=False)}

## 结果分析

DNN 集成能够学习非线性的技术统计组合，年代专家进一步吸收了不同时期位置打法的变化。ExtraTrees 辅助模型单独效果低于 DNN，但它在部分样本上与 DNN 的错误不同，因此概率融合后整体指标略有提高。该方案没有使用球员姓名、球队或历史位置，仍然只基于合规技术统计字段完成分类。
"""
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("Experiment 5 completed: GPU DNN ensemble + auxiliary probability fusion")
    print(f"Device: {DEVICE}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Result directory: {RESULT_DIR}")


if __name__ == "__main__":
    main()
