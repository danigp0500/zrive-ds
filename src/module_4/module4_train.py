import pandas as pd
import os
import json
import datetime
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
)
from typing import Tuple, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

OUTPUT_PATH = Path("models/module_4_rf")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    dataset = pd.read_csv(os.path.join(data_dir, "feature_frame.csv"))
    dataset["created_at"] = pd.to_datetime(dataset["created_at"])
    dataset["order_date"] = pd.to_datetime(dataset["order_date"]).dt.date
    return dataset


def filter_orders(df: pd.DataFrame, min_items: int = 5) -> pd.DataFrame:
    order_sizes = df.groupby("order_id")["outcome"].sum()
    valid_orders = order_sizes[order_sizes >= min_items].index
    return df[df["order_id"].isin(valid_orders)]


def split_temporal(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_orders = df.groupby("created_at").order_id.nunique()
    cumsum = daily_orders.cumsum() / daily_orders.sum()
    train_end = cumsum[cumsum <= 0.7].idxmax()
    val_end = cumsum[cumsum <= 0.9].idxmax()
    train = df[df["created_at"] <= train_end]
    val = df[(df["created_at"] > train_end) & (df["created_at"] <= val_end)]
    return train, val


# --- Evaluation ---
def evaluate(y_true, y_pred, label=""):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_pred)
    logger.info(f"{label} | PR AUC: {pr_auc:.3f}, ROC AUC: {roc_auc:.3f}")


# --- Train logic as separate function ---
def train_model(train: pd.DataFrame, val: pd.DataFrame, model_param: Dict) -> str:
    required_keys = ["n_estimators", "max_depth", "min_samples_leaf"]
    for k in required_keys:
        if k not in model_param:
            raise ValueError(f"Missing required parameter: {k}")
    feature_cols = model_param.get(
        "feature_cols",
        [
            "user_order_seq",
            "ordered_before",
            "abandoned_before",
            "count_children",
            "count_babies",
            "count_pets",
            "people_ex_baby",
            "days_since_purchase_variant_id",
            "avg_days_to_buy_variant_id",
            "std_days_to_buy_variant_id",
            "days_since_purchase_product_type",
            "avg_days_to_buy_product_type",
            "std_days_to_buy_product_type",
        ],
    )

    X_train = train[feature_cols]
    y_train = train["outcome"]
    X_val = val[feature_cols]
    y_val = val["outcome"]

    rf = RandomForestClassifier(
        n_estimators=model_param["n_estimators"],
        max_depth=model_param["max_depth"],
        min_samples_leaf=model_param["min_samples_leaf"],
        max_features=model_param.get("max_features", "sqrt"),
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    evaluate(y_train, rf.predict_proba(X_train)[:, 1], "Train")
    evaluate(y_val, rf.predict_proba(X_val)[:, 1], "Validation")

    model_name = f"push_{datetime.date.today():%Y_%m_%d}.pkl"
    model_path = OUTPUT_PATH / model_name
    joblib.dump(rf, model_path)
    logger.info(f"Model saved to {model_path}")

    return str(model_path)


# --- API handler function ---
def handler_fit(event, _) -> Dict:
    if "model_parametrisation" not in event:
        raise ValueError("Missing 'model_parametrisation' in input event")
    model_param = event["model_parametrisation"]
    df = filter_orders(load_data())
    train, val = split_temporal(df)
    model_path = train_model(train, val, model_param)
    return {"statusCode": "200", "body": json.dumps({"model_path": model_path})}
