import uvicorn
import time
from fastapi import FastAPI, HTTPException
from utils.data_downloader import download_s3_files
from module_6.exceptions import PredictionException, UserNotFoundException
from module_6.basket_model.basket_model import BasketModel
from module_6.basket_model.feature_store import FeatureStore
from module_6.metrics import log_prediction, log_error
import pandas as pd

# Data managing
download_s3_files("zrive-ds-data", "groceries/sampled-datasets", local_subdir="data")
download_s3_files("zrive-ds-data", "groceries/trained-models", local_subdir="bin")


app = FastAPI()

model = BasketModel()
dataset = FeatureStore()


@app.get("/status")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def price_prediction(user_id: str):
    start_time = time.time()
    try:
        features = dataset.get_features(user_id)
        if isinstance(features, pd.DataFrame):
            features = features.sort_values("created_at").tail(1)
        features = features.values.astype(float).reshape(1, -1)
        prediction = model.predict(features)
        latency = time.time() - start_time
        log_prediction(user_id, prediction[0], latency)
        return {"predicted_basket_value": float(prediction[0])}
    except UserNotFoundException:
        log_error(user_id, "User not found")
        raise HTTPException(status_code=404, detail="User not found")
    except PredictionException:
        log_error(user_id, "Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    except Exception as e:
        log_error(user_id, f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
