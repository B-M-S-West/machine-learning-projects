from fastapi.testclient import TestClient

from ml_API_example.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the House Price Prediction API"}

def test_predict_price():
    response = client.post("/predict", json={"size_sqft": 1500, "bedrooms": 3})
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert isinstance(response.json()["predicted_price"], float)

def test_predict_batch():
    response = client.post("/predict_batch", json={"houses": [{"size_sqft": 1500, "bedrooms": 3}, {"size_sqft": 2000, "bedrooms": 4}, {"size_sqft": 2500, "bedrooms": 5}]})
    assert response.status_code == 200
    assert "predicted_prices" in response.json()
    assert isinstance(response.json()["predicted_prices"], list)
    assert all(isinstance(price, float) for price in response.json()["predicted_prices"])
