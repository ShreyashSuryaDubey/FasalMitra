"""Quick self-check:  python tests_smoke.py   (no server needed)"""
from app import app

c = app.test_client()
ok = {"N": 80, "P": 48, "K": 40, "temperature": 24, "humidity": 82, "ph": 6.4, "rainfall": 236}

r = c.post("/api/predict", json=ok)
assert r.status_code == 200 and r.json["recommendations"][0]["crop"] == "rice", r.json
print("PASS  rice-like conditions -> rice")

r = c.post("/api/predict", json={**ok, "ph": 70})
assert r.status_code == 400
print("PASS  impossible pH is rejected")

for page in ("/", "/insights", "/?lang=hi", "/insights?lang=hi"):
    assert c.get(page).status_code == 200
print("PASS  all pages load (English + Hindi)")

html = c.post("/", data={**ok, "lang": "en"}).get_data(as_text=True)
assert "Fertilizer plan" in html and "Rice" in html
print("PASS  form submission shows crops and a fertilizer plan")
