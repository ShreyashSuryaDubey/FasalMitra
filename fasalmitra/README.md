# FasalMitra – Smart Crop & Fertilizer Advisor

**SDG 2 – Zero Hunger** (also supports SDG 12 Responsible Consumption and SDG 13 Climate Action by reducing fertilizer waste)

A farmer enters the numbers from a Soil Health Card (N, P, K, pH) plus local temperature, humidity and rainfall.
A machine-learning model suggests the **top 3 suitable crops**, explains *why*, and gives a simple
**fertilizer plan** (what to add, and what to skip to save money). Works in **English and Hindi**.

## Problem
Small farmers often pick crops and fertilizer by habit or guesswork. A wrong crop gives a low yield;
over-fertilizing wastes money and damages soil and water. Soil-test cards exist, but most farmers cannot interpret them.

## Concepts Used
| Supervised learning (labelled data → learn mapping) | `train.py` – soil/weather → crop label |
| Classification & decision trees | Decision Tree and Random Forest models |
| Steps of ML (data → prepare → train → test → improve) | Sections 1–5 in `train.py` |
| Train / test split | 80 % train, 20 % unseen test data |
| Error / accuracy | Test accuracy + 5-fold cross-validation, model comparison chart |
| Problem-first thinking (Lean Canvas) | Solution designed around a clear farmer problem |

## Run it (5 minutes)
```bash
pip install -r requirements.txt
python train.py              # trains the model, saves charts
python app.py                # open http://127.0.0.1:5000
python test_app.py           # optional: quick self-check
```

## Project files
```
app.py               Flask web app + recommendation logic
train.py             trains and compares models, saves model + charts
translations.py              English / Hindi text
templates/, static/  web pages and styling
model/               saved model + metrics (created by train.py)
screenshots/         ready-made screenshots for the PPT
```

## Limitations
* Fertilizer quantities are simple arithmetic estimates, **not** agronomist-validated; the app tells users to confirm with a KVK.
* The dataset is a general benchmark, not local to a state/district; accuracy on real farms will be lower than on the test split.
* No yield or price prediction yet – a natural next step (linear regression on yield vs fertilizer, as in Masterclass 3).
