import os
from app import create_app
from app.services.prediction_service import make_prediction

app = create_app()

with app.app_context():
    input_data = {
        "brand": "Kia",
        "model": "Seltos",
        "year": 2023,
        "seller_type": "Individual",
        "kilometers_driven": 15000,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "mileage": 16.8,
        "engine": 1497,
        "power": 113.4,
        "seats": 5
    }
    try:
        res = make_prediction(input_data)
        print("Success:", res)
    except Exception as e:
        print("Error:", repr(e))
        import traceback
        traceback.print_exc()
