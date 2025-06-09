from src.module_4.module4_inference import handler_predict
import json


def test_handler_predict_single_instance():
    test_event = {
        "users": {
            "test_user": {
                "user_order_seq": 3,
                "ordered_before": 1,
                "abandoned_before": 0,
                "count_children": 1,
                "count_babies": 0,
                "count_pets": 1,
                "people_ex_baby": 2,
                "days_since_purchase_variant_id": 10,
                "avg_days_to_buy_variant_id": 7.2,
                "std_days_to_buy_variant_id": 2.5,
                "days_since_purchase_product_type": 20,
                "avg_days_to_buy_product_type": 15.3,
                "std_days_to_buy_product_type": 4.1,
            }
        }
    }

    response = handler_predict(test_event, None)
    assert response["statusCode"] == "200"
    result = json.loads(response["body"])
    assert "test_user" in result["prediction"]
    assert 0.0 <= result["prediction"]["test_user"] <= 1.0
