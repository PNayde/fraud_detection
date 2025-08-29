from starlette.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_dummy():
    r = client.post("/predict", json={"rows":[{"a":-1.0,"b":0.2},{"a":1.5,"b":0.3}]})
    assert r.status_code == 200
    body = r.json()
    assert "preds" in body and len(body["preds"]) == 2
