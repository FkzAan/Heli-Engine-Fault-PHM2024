"""
Prototype Testing and Evaluation (Phase 3.9)

Checks that the prototype gives the same predictions as the notebook,
and that it handles invalid input values.

How to run (from the Heli-Engine-Fault-PHM2024 folder):
    python prototype/prototype_test.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
model = joblib.load(MODEL_DIR / "xgboost_model.joblib")
info = json.loads((MODEL_DIR / "model_info.json").read_text())
samples = pd.read_csv(MODEL_DIR / "test_samples.csv")
FEATURES = info["features"]


def prototype_predict(values):
    """Same prediction steps as app.py."""
    row = pd.DataFrame([values], columns=FEATURES)
    prob_faulty = float(model.predict_proba(row)[0][1])
    return (1 if prob_faulty >= 0.5 else 0), prob_faulty


def check_inputs(values):
    """Return 'error', 'warning' or 'ok' for one set of inputs."""
    for f in FEATURES:
        v = values.get(f)
        if v is None:
            return "error"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "error"
        low, high = info["ranges"][f]
        if not (low <= v <= high):
            return "warning"
    return "ok"


# ---------- Test 1: prototype output vs notebook output ----------
rows = []
for _, r in samples.iterrows():
    pred, prob = prototype_predict({f: r[f] for f in FEATURES})
    rows.append({
        "actual_label": int(r["actual_label"]),
        "notebook_prediction": int(r["notebook_prediction"]),
        "prototype_prediction": pred,
        "prob_faulty": round(prob, 4),
    })
out = pd.DataFrame(rows)
out["same_as_notebook"] = out["prototype_prediction"] == out["notebook_prediction"]
out["correct"] = out["prototype_prediction"] == out["actual_label"]
out.to_csv(MODEL_DIR / "prototype_test_results.csv", index=False)

print("===== Test 1: prototype vs notebook =====")
print(f"Records tested                     : {len(out)}")
print(f"Prototype output same as notebook  : {out['same_as_notebook'].mean():.1%}")
print(f"Prototype prediction matches actual: {out['correct'].mean():.1%}")

# ---------- Test 2: input validation ----------
valid = {f: samples.iloc[0][f] for f in FEATURES}
cases = [
    ("TC01", "All seven values valid", dict(valid), "ok"),
    ("TC02", "One value missing", {**valid, FEATURES[0]: None}, "error"),
    ("TC03", "Non-numeric value", {**valid, FEATURES[1]: "abc"}, "error"),
    ("TC04", "Value outside training range", {**valid, FEATURES[2]: 99999}, "warning"),
]
results = []
for cid, desc, vals, expected in cases:
    actual = check_inputs(vals)
    results.append({"Test case": cid, "Description": desc, "Expected": expected,
                    "Actual": actual, "Result": "Pass" if actual == expected else "Fail"})
results = pd.DataFrame(results)
results.to_csv(MODEL_DIR / "prototype_validation_tests.csv", index=False)

print("\n===== Test 2: input validation =====")
print(results.to_string(index=False))