from datetime import datetime
import os

METRICS_PATH = os.path.join(os.path.dirname(__file__), "logs", "metrics.txt")


def log_prediction(user_id: str, prediction: float, latency: float) -> None:
    with open(METRICS_PATH, "a") as f:
        f.write(
            f"[{datetime.now()}] user: {user_id}"
            + "| pred: {prediction} | latency: {latency} \n"
        )


def log_error(user_id: str, error: str) -> None:
    with open(METRICS_PATH, "a") as f:
        f.write(f"[{datetime.now()}] ERROR for user: {user_id} | reason: {error} \n")
