from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "NBA_Season_Stats.csv"
OUTPUT_DIR = ROOT / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Pos"
DROP_FEATURE_COLS = ["Player", "Tm"]
EXPECTED_POSITIONS = ["C", "PF", "PG", "SG", "SF"]


def fit_discretizer(series: pd.Series, bins: int = 3) -> tuple[np.ndarray | None, list[str]]:
    labels_by_count = {
        1: ["same"],
        2: ["low", "high"],
        3: ["low", "mid", "high"],
    }

    unique_count = series.nunique(dropna=True)
    if unique_count <= 1:
        return None, ["same"]

    bin_count = min(bins, unique_count)
    _, edges = pd.qcut(series, q=bin_count, labels=False, duplicates="drop", retbins=True)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return None, ["same"]

    edges[0] = -np.inf
    edges[-1] = np.inf
    actual_bins = len(edges) - 1
    labels = labels_by_count.get(actual_bins, [f"bin_{i}" for i in range(actual_bins)])
    return edges, labels


def apply_discretizer(series: pd.Series, edges: np.ndarray | None, labels: list[str]) -> pd.Series:
    if edges is None:
        return pd.Series([labels[0]] * len(series), index=series.index)
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True).astype(str)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(RAW_PATH)
    raw_columns = list(raw.columns)

    data = raw.copy()
    data = data[data[TARGET_COL].isin(EXPECTED_POSITIONS)].copy()
    data = data.reset_index(drop=True)

    feature_cols = [col for col in data.columns if col not in DROP_FEATURE_COLS + [TARGET_COL]]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    missing_before = data[feature_cols].isna().sum()
    data[feature_cols] = data[feature_cols].fillna(0)

    le = LabelEncoder()
    data["Pos_Label"] = le.fit_transform(data[TARGET_COL])
    label_mapping = {label: int(code) for code, label in enumerate(le.classes_)}

    clean = data[feature_cols + [TARGET_COL, "Pos_Label"]]

    train_idx, test_idx = train_test_split(
        clean.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean[TARGET_COL],
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    clean_train = clean.loc[train_idx].reset_index(drop=True)
    clean_test = clean.loc[test_idx].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(clean_train[feature_cols])

    def make_scaled(frame: pd.DataFrame) -> pd.DataFrame:
        scaled_features = pd.DataFrame(
            scaler.transform(frame[feature_cols]),
            columns=feature_cols,
            index=frame.index,
        )
        scaled_features[TARGET_COL] = frame[TARGET_COL].values
        scaled_features["Pos_Label"] = frame["Pos_Label"].values
        return scaled_features

    scaled_train = make_scaled(clean_train)
    scaled_test = make_scaled(clean_test)
    scaled_all = make_scaled(clean.reset_index(drop=True))

    discrete = clean.copy()
    for col in feature_cols:
        edges, labels = fit_discretizer(clean_train[col])
        discrete[col] = apply_discretizer(clean[col], edges, labels)
    discrete_train = discrete.loc[train_idx].reset_index(drop=True)
    discrete_test = discrete.loc[test_idx].reset_index(drop=True)

    clean.to_csv(OUTPUT_DIR / "nba_clean.csv", index=False)
    clean_train.to_csv(OUTPUT_DIR / "nba_clean_train.csv", index=False)
    clean_test.to_csv(OUTPUT_DIR / "nba_clean_test.csv", index=False)

    scaled_all.to_csv(OUTPUT_DIR / "nba_scaled.csv", index=False)
    scaled_train.to_csv(OUTPUT_DIR / "nba_scaled_train.csv", index=False)
    scaled_test.to_csv(OUTPUT_DIR / "nba_scaled_test.csv", index=False)

    discrete.to_csv(OUTPUT_DIR / "nba_discrete.csv", index=False)
    discrete_train.to_csv(OUTPUT_DIR / "nba_discrete_train.csv", index=False)
    discrete_test.to_csv(OUTPUT_DIR / "nba_discrete_test.csv", index=False)

    (OUTPUT_DIR / "feature_columns.txt").write_text(
        "\n".join(feature_cols) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "raw_path": str(RAW_PATH.name),
        "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "raw_columns": raw_columns,
        "drop_feature_columns": DROP_FEATURE_COLS,
        "target_column": TARGET_COL,
        "expected_positions": EXPECTED_POSITIONS,
        "clean_shape": [int(clean.shape[0]), int(clean.shape[1])],
        "feature_count": len(feature_cols),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(clean_train)),
        "test_rows": int(len(clean_test)),
        "class_counts": clean[TARGET_COL].value_counts().sort_index().to_dict(),
        "missing_values_filled_with_zero": {
            col: int(count)
            for col, count in missing_before.items()
            if int(count) > 0
        },
        "outputs": [
            "nba_clean.csv",
            "nba_clean_train.csv",
            "nba_clean_test.csv",
            "nba_scaled.csv",
            "nba_scaled_train.csv",
            "nba_scaled_test.csv",
            "nba_discrete.csv",
            "nba_discrete_train.csv",
            "nba_discrete_test.csv",
            "feature_columns.txt",
            "label_mapping.json",
            "preprocess_report.json",
        ],
    }
    (OUTPUT_DIR / "preprocess_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NBA preprocessing completed.")
    print(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"Clean shape: {clean.shape[0]} rows x {clean.shape[1]} columns")
    print(f"Train/Test: {len(clean_train)} / {len(clean_test)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
