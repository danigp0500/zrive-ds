import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict
from joblib import load

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Ruta de modelos
OUTPUT_PATH = Path("models/module_4_rf")

# Features esperadas (deben coincidir con entrenamiento)
FEATURE_COLS = [
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
]


def get_latest_model_path() -> Path:
    model_files = list(OUTPUT_PATH.glob("push_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained model found in OUTPUT_PATH")
    return max(model_files, key=os.path.getmtime)


def handler_predict(event, _) -> Dict:
    if "users" not in event:
        raise ValueError("Missing 'users' field in input")

    # Convert JSON to DataFrame
    user_dict = (
        json.loads(event["users"])
        if isinstance(event["users"], str)
        else event["users"]
    )
    X = pd.DataFrame.from_dict(user_dict, orient="index")

    # Asegurar orden y columnas esperadas
    X = X[FEATURE_COLS]

    # Cargar modelo
    model_path = get_latest_model_path()
    model = load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Realizar predicciones
    y_pred = model.predict_proba(X)[:, 1]  # Probabilidad de outcome=1
    predictions = dict(zip(X.index, y_pred.round(4).tolist()))

    return {"statusCode": "200", "body": json.dumps({"prediction": predictions})}
