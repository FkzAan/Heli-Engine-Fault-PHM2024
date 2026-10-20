"""
Helicopter Engine Fault Detection - Streamlit prototype (PHM 2024 dataset)

How to run (from the Heli-Engine-Fault-PHM2024 folder):
    streamlit run prototype/app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# Friendly names for the 7 sensors, in two groups
SENSOR_NAMES = {
    "trq_measured": "Output torque",
    "mgt": "Mean gas temperature",
    "ng": "Gas generator speed",
    "np": "Power turbine speed",
    "oat": "Outside air temperature",
    "pa": "Pressure altitude",
    "ias": "Indicated airspeed",
}
GROUPS = {
    "Engine readings": ["trq_measured", "mgt", "ng", "np"],
    "Flight conditions": ["oat", "pa", "ias"],
}
LABELS = {0: "Normal", 1: "Faulty"}

st.set_page_config(page_title="Helicopter Engine Fault Detection", page_icon="🚁", layout="wide")


# ------------------------------------------------------------
# 2. Load the saved model (only once, then cached)
# ------------------------------------------------------------
@st.cache_resource
def load_files():
    model = joblib.load(MODEL_DIR / "xgboost_model.joblib")
    with open(MODEL_DIR / "model_info.json") as f:
        info = json.load(f)
    samples = pd.read_csv(MODEL_DIR / "test_samples.csv")
    return model, info, samples


try:
    model, info, samples = load_files()
except FileNotFoundError:
    st.error("Model files not found in the model folder. "
             "Run the 'save model' cell (Cell 19) in the notebook first, then refresh this page.")
    st.stop()

FEATURES = info["features"]

# Start each input at the middle of its training range
for f in FEATURES:
    low, high = info["ranges"][f]
    st.session_state.setdefault(f, round((low + high) / 2, 2))
st.session_state.setdefault("actual_label", None)


# ------------------------------------------------------------
# 3. Button actions
# ------------------------------------------------------------
def load_random_sample():
    """Fill the inputs with one real record from the testing set."""
    row = samples.sample(1).iloc[0]
    for f in FEATURES:
        st.session_state[f] = round(float(row[f]), 2)
    st.session_state.actual_label = int(row["actual_label"])


def reset_inputs():
    for f in FEATURES:
        low, high = info["ranges"][f]
        st.session_state[f] = round((low + high) / 2, 2)
    st.session_state.actual_label = None


# ------------------------------------------------------------
# 4. Page header and sidebar
# ------------------------------------------------------------
st.title("Helicopter engine fault detection")
st.caption("FYP prototype: predicts whether a helicopter turbine engine is normal or faulty from 7 sensor "
           "readings. For academic demonstration only, not for real maintenance decisions.")

with st.sidebar:
    st.subheader("Model")
    st.write(f"{info['model_name']}, trained on the PHM 2024 Data Challenge dataset")
    st.subheader("Test set performance")
    scores = info["test_scores"]
    st.metric("Accuracy (all 7 sensors)", f"{scores['accuracy']:.2%}")
    st.metric("F1-score", f"{scores['f1_score']:.4f}")
    st.metric("Accuracy (engine sensors only)", f"{scores['engine_sensors_only_accuracy']:.2%}")
    st.caption("Accuracy with all sensors is likely optimistic, because the dataset links faults to "
               "flight conditions. The engine-sensors-only result is a more conservative estimate.")

# ------------------------------------------------------------
# 5. Input form (left) and result (right)
# ------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Sensor readings")
    st.caption("Enter values in the same units as the dataset. The range under each box is the range "
               "seen in the training data.")

    b1, b2 = st.columns(2)
    b1.button("Load a test sample", on_click=load_random_sample)
    b2.button("Reset", on_click=reset_inputs)

    for group_name, sensors in GROUPS.items():
        st.markdown(f"**{group_name}**")
        # At most 3 boxes per row, so every number fits
        for start in range(0, len(sensors), 3):
            cols = st.columns(3)
            for col, sensor in zip(cols, sensors[start:start + 3]):
                low, high = info["ranges"][sensor]
                col.number_input(SENSOR_NAMES[sensor], key=sensor, step=0.1, format="%.2f",
                                 help=f"Training range: {low} to {high}")
                col.caption(f"Range {low} to {high}")

    predict_clicked = st.button("Predict engine condition", type="primary")

with right:
    st.subheader("Prediction result")

    if not predict_clicked:
        st.info("Enter sensor readings or load a test sample, then click Predict engine condition.")
    else:
        values = {f: st.session_state[f] for f in FEATURES}
        input_row = pd.DataFrame([values], columns=FEATURES)

        # Warn if a value is outside what the model saw in training
        for f in FEATURES:
            low, high = info["ranges"][f]
            if not (low <= values[f] <= high):
                st.warning(f"{SENSOR_NAMES[f]} = {values[f]:.2f} is outside the training range "
                           f"({low} to {high}). The prediction may be less reliable.")

        # Predict
        prob_faulty = float(model.predict_proba(input_row)[0][1])
        predicted = 1 if prob_faulty >= 0.5 else 0
        confidence = prob_faulty if predicted == 1 else 1 - prob_faulty

        if predicted == 0:
            st.success(f"Predicted condition: **{LABELS[predicted]}**")
        else:
            st.error(f"Predicted condition: **{LABELS[predicted]}**")
        st.metric("Confidence", f"{confidence:.1%}")

        # If the inputs came from a test sample, show the real answer
        if st.session_state.actual_label is not None:
            actual = LABELS[st.session_state.actual_label]
            result = "correct" if predicted == st.session_state.actual_label else "wrong"
            st.write(f"Actual label of this sample: **{actual}** (prediction is {result})")

        st.markdown("**Probability for each condition**")
        st.progress(1 - prob_faulty, text=f"Normal: {1 - prob_faulty:.1%}")
        st.progress(prob_faulty, text=f"Faulty: {prob_faulty:.1%}")