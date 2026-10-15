"""
app.py  -  FasalMitra: Smart Crop & Fertilizer Advisor (Flask web app)

Flow:  farmer enters soil-test + weather values
       -> trained ML model gives the top-3 suitable crops with a confidence score
       -> for each crop we compare the soil with that crop's "ideal profile" (learned from data)
       -> plain-language fertilizer advice (add / skip) in English or Hindi

Run locally:   python app.py      then open  http://127.0.0.1:5000
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from translations import CROP_HI, FEATURE_LABELS, TEXT

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

# if the model has not been trained yet (e.g. first run on a server), train it now
if not os.path.exists("model/crop_model.joblib"):
    import subprocess
    import sys
    subprocess.run([sys.executable, "train.py"], check=True)

MODEL = joblib.load("model/crop_model.joblib")
with open("model/metrics.json") as f:
    METRICS = json.load(f)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
PROFILES = METRICS["profiles"]
RANGES = METRICS["feature_ranges"]

# Approximate nutrient content of common fertilizers (share of the bag that is the nutrient)
UREA_N, SSP_P, MOP_K = 0.46, 0.16, 0.60

# Allowed input limits (generous, so a typo like pH=70 is caught)
LIMITS = {
    "N": (0, 300), "P": (0, 300), "K": (0, 400),
    "temperature": (0, 50), "humidity": (0, 100), "ph": (3, 10), "rainfall": (0, 500),
}

app = Flask(__name__)


def tr(lang):
    return TEXT.get(lang, TEXT["en"])


def crop_name(crop, lang):
    return CROP_HI.get(crop, crop) if lang == "hi" else crop.capitalize()


def round5(x):
    return int(5 * round(x / 5)) if x >= 5 else 0


def parse_inputs(form):
    """Read the 7 numbers from the form. Returns (values, errors)."""
    values, errors = {}, []
    for f in FEATURES:
        raw = (form.get(f) or "").strip()
        try:
            v = float(raw)
        except ValueError:
            errors.append(f)
            continue
        lo, hi = LIMITS[f]
        if not (lo <= v <= hi):
            errors.append(f)
        values[f] = v
    return values, errors


def feature_checks(values, crop):
    """Compare each input with the crop's normal range (10th-90th percentile)."""
    out = []
    for f in FEATURES:
        p = PROFILES[crop][f]
        v = values[f]
        status = "ok" if p["p10"] <= v <= p["p90"] else ("high" if v > p["p90"] else "low")
        out.append({"feature": f, "value": v, "low": p["p10"], "high": p["p90"], "status": status})
    return out


def fertilizer_advice(values, crop, lang):
    """Turn the gap between the soil and the crop's ideal N-P-K into simple advice."""
    t = tr(lang)
    advice = []
    for nutrient, share, fert_key in (("N", UREA_N, "urea"), ("P", SSP_P, "ssp"), ("K", MOP_K, "mop")):
        ideal = PROFILES[crop][nutrient]["median"]
        have = values[nutrient]
        if have < 0.8 * ideal:                      # soil is low -> add
            kg = round5((ideal - have) / share)
            advice.append({"nutrient": nutrient, "status": "low",
                           "text": t["adv_low"].format(n=nutrient, gap=round(ideal - have), kg=kg, fert=t[fert_key])})
        elif have > 1.2 * ideal:                    # soil is already rich -> skip, save money
            kg = round5((have - ideal) / share)
            advice.append({"nutrient": nutrient, "status": "high",
                           "text": t["adv_high"].format(n=nutrient, kg=kg, fert=t[fert_key])})
        else:
            advice.append({"nutrient": nutrient, "status": "ok", "text": t["adv_ok"].format(n=nutrient)})

    ph = values["ph"]
    p = PROFILES[crop]["ph"]
    if ph < p["p10"]:
        advice.append({"nutrient": "pH", "status": "low", "text": t["ph_low"]})
    elif ph > p["p90"]:
        advice.append({"nutrient": "pH", "status": "high", "text": t["ph_high"]})
    else:
        advice.append({"nutrient": "pH", "status": "ok", "text": t["ph_ok"]})
    return advice


def recommend(values, lang="en"):
    """Run the ML model and build the full result."""
    row = pd.DataFrame([values])[FEATURES]
    proba = MODEL.predict_proba(row)[0]
    classes = list(MODEL.classes_)
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    # top 3 crops, but skip ones with a negligible score (always keep at least the best one)
    top = [x for x in ranked[:3] if x[1] >= 0.01] or ranked[:1]

    results = []
    for crop, p in top:
        results.append({
            "crop": crop,
            "name": crop_name(crop, lang),
            "score": round(float(p) * 100, 1),
            "checks": feature_checks(values, crop),
            "advice": fertilizer_advice(values, crop, lang),
        })

    warnings = []
    t = tr(lang)
    for f in FEATURES:
        lo, hi = RANGES[f]
        if values[f] < lo * 0.5 or values[f] > hi * 1.5:
            warnings.append(t["warn_range"].format(f=FEATURE_LABELS[lang][f]))
    if results[0]["score"] < 50:
        warnings.append(t["warn_low_conf"])
    return results, warnings


@app.route("/", methods=["GET", "POST"])
def index():
    lang = request.values.get("lang", "en")
    if lang not in TEXT:
        lang = "en"
    ctx = dict(t=tr(lang), lang=lang, labels=FEATURE_LABELS[lang], features=FEATURES,
               values={}, results=None, warnings=[], errors=[],
               limits=LIMITS)
    if request.method == "POST":
        values, errors = parse_inputs(request.form)
        ctx["values"] = request.form
        if errors:
            ctx["errors"] = [FEATURE_LABELS[lang][e] for e in errors]
        else:
            ctx["results"], ctx["warnings"] = recommend(values, lang)
    return render_template("index.html", **ctx)


@app.route("/insights")
def insights():
    lang = request.args.get("lang", "en")
    if lang not in TEXT:
        lang = "en"
    return render_template("insights.html", t=tr(lang), lang=lang, m=METRICS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API, e.g.  curl -X POST -H 'Content-Type: application/json' -d '{...}' /api/predict"""
    data = request.get_json(silent=True) or {}
    values, errors = parse_inputs({k: str(v) for k, v in data.items()})
    if errors or len(values) != len(FEATURES):
        return jsonify({"error": "Provide valid numbers for: " + ", ".join(FEATURES)}), 400
    results, warnings = recommend(values, data.get("lang", "en"))
    return jsonify({"recommendations": [{"crop": r["crop"], "score": r["score"]} for r in results],
                    "warnings": warnings})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
