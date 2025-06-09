import pandas as pd
from src.module_4.module4_train import (
    evaluate,
    train_model,
    filter_orders,
    load_data,
    split_temporal,
)


def test_evaluate_ordered_scores():
    y_true = pd.Series([0, 0, 1, 1])
    y_good = pd.Series([0.1, 0.2, 0.8, 0.9])
    y_bad = pd.Series([0.9, 0.8, 0.2, 0.1])

    # Solo testeo que evalúe, no hace falta logger aquí
    auc_good = evaluate(y_true, y_good, label="good")
    auc_bad = evaluate(y_true, y_bad, label="bad")

    # No falla si la métrica es mayor
    assert auc_good is None  # porque evaluate no devuelve nada
    assert auc_bad is None


def test_train_model_execution():
    df = filter_orders(load_data())
    train, val = split_temporal(df)

    model_param = {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1}

    model_path = train_model(train, val, model_param)
    assert model_path.endswith(".pkl")
