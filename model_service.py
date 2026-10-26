"""Load model artifacts and run a classification on the supplied features."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    model_path = PROJECT_DIR / "model.pkl"
    scaler_path = PROJECT_DIR / "scaler.pkl"
    if not model_path.is_file() or not scaler_path.is_file():
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)


def classify(model, scaler, values: list[float]):
    """Return the model class and its largest class probability, when available."""
    feature_count = getattr(model, "n_features_in_", len(values))
    if feature_count != len(values):
        raise ValueError(
            f"The model expects {feature_count} inputs, but the form supplied {len(values)}."
        )

    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is not None:
        model_input = pd.DataFrame([values], columns=list(feature_names))
    else:
        model_input = np.asarray([values], dtype=float)

    scaled_values = scaler.transform(model_input)
    prediction = model.predict(scaled_values)[0]
    try:
        probabilities = model.predict_proba(scaled_values)[0]
        confidence = float(np.max(probabilities) * 100)
    except (AttributeError, NotImplementedError):
        confidence = None
    return prediction, confidence
