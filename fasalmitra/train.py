"""
train.py
--------
Trains the crop recommendation model and saves what the web app needs.

Steps (as in "Steps of Machine Learning", Masterclass 3):
  1. load the data        2. clean it         3. train four models
  4. test them            5. keep the best and save it, plus charts

Needs data/Crop_recommendation.csv (Kaggle "Crop Recommendation Dataset").
Run:  python train.py
"""
import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DATA_CSV = "data/Crop_recommendation.csv"
GREEN, GREY = "#2E7D32", "#9AA5A0"

os.makedirs("model", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)


# ---------------------------------------------------------------- data
def load_data():
    if not os.path.exists(DATA_CSV):
        raise SystemExit(f"Dataset not found. Save the Kaggle file as {DATA_CSV} and run again.")
    df = pd.read_csv(DATA_CSV)
    # accept any capitalisation of the column names
    by_lower = {c.strip().lower(): c for c in df.columns}
    for want in FEATURES + ["label"]:
        if want.lower() not in by_lower:
            raise SystemExit(f"Column '{want}' not found in {DATA_CSV}. Found: {list(df.columns)}")
    df = df.rename(columns={by_lower[w.lower()]: w for w in FEATURES + ["label"]})
    df = df[FEATURES + ["label"]].dropna().drop_duplicates()
    df["label"] = df["label"].str.strip().str.lower()
    return df


df = load_data()

# Crops to keep. Leave the list EMPTY to use every crop in the data file.
# Put your own state's crops to focus the app, for example:
#   KEEP_CROPS = ["rice", "maize", "cotton", "chickpea", "lentil", "mungbean"]
KEEP_CROPS = []
if KEEP_CROPS:
    df = df[df["label"].isin([c.strip().lower() for c in KEEP_CROPS])]
    if df["label"].nunique() < 2:
        raise SystemExit("KEEP_CROPS must match at least 2 crops that exist in the data file.")

print(f"Loaded {DATA_CSV}: rows={len(df)}, crops={df['label'].nunique()}")

X, y = df[FEATURES], df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------- models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv = cross_val_score(model, X_train, y_train, cv=5)
    results[name] = {"test_accuracy": round(float(test_acc), 4), "cv_mean": round(float(cv.mean()), 4)}
    print(f"{name:20s} test={test_acc:.3f}  cv={cv.mean():.3f}")

# ----------------------------------------------------------  pick best
best_name = max(results, key=lambda n: (results[n]["cv_mean"], results[n]["test_accuracy"]))
best = models[best_name]
print("Best model:", best_name)

# the final model is re-trained on ALL data before saving
final = type(best)(**best.get_params()) if not hasattr(best, "steps") else best
final.fit(X, y)
joblib.dump(final, "model/crop_model.joblib")

# feature importance always comes from a Random Forest 
rf = models["Random Forest"]
importances = dict(zip(FEATURES, [round(float(v), 4) for v in rf.feature_importances_]))

# ideal profile per crop: median and the 10th-90th percentile range of each feature
profiles = {}
for crop, g in df.groupby("label"):
    profiles[crop] = {
        f: {
            "median": round(float(g[f].median()), 2),
            "p10": round(float(g[f].quantile(0.10)), 2),
            "p90": round(float(g[f].quantile(0.90)), 2),
        }
        for f in FEATURES
    }

feature_ranges = {f: [round(float(df[f].min()), 2), round(float(df[f].max()), 2)] for f in FEATURES}

metrics = {
    "data_source": "Kaggle Crop Recommendation Dataset",
    "rows": int(len(df)),
    "n_crops": int(df["label"].nunique()),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "models": results,
    "best_model": best_name,
    "feature_importance": importances,
    "feature_ranges": feature_ranges,
    "classes": sorted(df["label"].unique().tolist()),
    "profiles": profiles,
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ----------------------------------------------------------------- charts
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# a) model comparison
fig, ax = plt.subplots(figsize=(7, 3.6))
names = list(results)
vals = [results[n]["test_accuracy"] * 100 for n in names]
bars = ax.barh(names, vals, color=[GREEN if n == best_name else GREY for n in names])
ax.set_xlim(max(0, min(vals) - 10), 101)
ax.set_xlabel("Accuracy on unseen test data (%)")
ax.invert_yaxis()
for b, v in zip(bars, vals):
    ax.text(v + 0.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center")
fig.tight_layout()
fig.savefig("static/charts/model_comparison.png", dpi=160)
plt.close(fig)

# b) feature importance
order = sorted(importances, key=importances.get)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(order, [importances[k] * 100 for k in order], color=GREEN)
ax.set_xlabel("Importance for choosing the crop (%)")
fig.tight_layout()
fig.savefig("static/charts/feature_importance.png", dpi=160)
plt.close(fig)

# c) confusion matrix of the Random Forest on the test set
labels = sorted(y.unique())
cm = confusion_matrix(y_test, rf.predict(X_test), labels=labels)
fig, ax = plt.subplots(figsize=(8.2, 7))
im = ax.imshow(cm, cmap="Greens")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Predicted crop")
ax.set_ylabel("Actual crop")
fig.colorbar(im, fraction=0.04)
fig.tight_layout()
fig.savefig("static/charts/confusion_matrix.png", dpi=160)
plt.close(fig)

# d) data exploration: temperature vs rainfall, one colour per crop family is too busy,
#    so show every crop as a faint dot and label the crop centres.
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.scatter(df["rainfall"], df["temperature"], s=6, color=GREY, alpha=0.35)
for crop, g in df.groupby("label"):
    ax.text(g["rainfall"].median(), g["temperature"].median(), crop, fontsize=7.5,
            ha="center", va="center", color="#1B4D1F", fontweight="bold")
ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Temperature (°C)")
fig.tight_layout()
fig.savefig("static/charts/temp_vs_rain.png", dpi=160)
plt.close(fig)

print("Saved model/crop_model.joblib, model/metrics.json and 4 charts in static/charts/")