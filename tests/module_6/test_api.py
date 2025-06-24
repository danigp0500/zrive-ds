from fastapi.testclient import TestClient
from src.module_6.app import app

client = TestClient(app)

TEST_USER_ID = "d3c1816b-066d-4cf2-8434-43021ac82bb3"


def test_status_ok():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}


def test_predict():
    response = client.post("/predict", params={"user_id": TEST_USER_ID})

    assert response.status_code == 200
    assert "predicted_basket_value" in response.json()
    assert isinstance(response.json()["predicted_basket_value"], float)
