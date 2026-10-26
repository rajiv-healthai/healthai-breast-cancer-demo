"""Breast cancer classification demonstration built with Streamlit."""

import base64
from html import escape
from pathlib import Path
import re
from string import Template

import streamlit as st

from model_service import classify, load_artifacts


PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = PROJECT_DIR / "templates.html"
STYLE_PATH = PROJECT_DIR / "assets" / "styles.css"

# Keep the display labels, ranges, defaults, and model feature order together.
FEATURES = [
    {
        "key": "mean_radius",
        "label": "Mean radius",
        "minimum": 0.0,
        "maximum": 40.0,
        "default": 14.0,
        "step": 0.1,
        "help": "Average distance from the center to points on the cell nucleus perimeter.",
    },
    {
        "key": "mean_texture",
        "label": "Mean texture",
        "minimum": 0.0,
        "maximum": 50.0,
        "default": 20.0,
        "step": 0.1,
        "help": "Standard deviation of grayscale intensity values in the cell region.",
    },
    {
        "key": "mean_perimeter",
        "label": "Mean perimeter",
        "minimum": 0.0,
        "maximum": 250.0,
        "default": 90.0,
        "step": 0.5,
        "help": "Mean length of the cell nucleus boundary.",
    },
    {
        "key": "mean_area",
        "label": "Mean area",
        "minimum": 0.0,
        "maximum": 3000.0,
        "default": 600.0,
        "step": 5.0,
        "help": "Mean area enclosed by the cell nucleus boundary.",
    },
    {
        "key": "mean_concavity",
        "label": "Mean concavity",
        "minimum": 0.0,
        "maximum": 1.0,
        "default": 0.1,
        "step": 0.01,
        "help": "Average severity of concave portions of the cell nucleus boundary.",
    },
]


def apply_styles() -> None:
    # Embed the animated neural-brain illustration with the theme stylesheet.
    css = STYLE_PATH.read_text(encoding="utf-8-sig")
    artwork = base64.b64encode(
        (PROJECT_DIR / "assets" / "neural_brain.svg").read_bytes()
    ).decode("ascii")
    css = css.replace("__BRAIN_ART_URL__", f"data:image/svg+xml;base64,{artwork}")
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


def render_html(name: str, **values: str) -> None:
    source = TEMPLATE_PATH.read_text(encoding="utf-8-sig")
    pattern = rf"<!-- BEGIN:{re.escape(name)} -->(.*?)<!-- END:{re.escape(name)} -->"
    match = re.search(pattern, source, flags=re.DOTALL)
    if not match:
        st.warning(f"The page section '{name}' could not be loaded.")
        return
    safe_values = {key: escape(str(value)) for key, value in values.items()}
    html = Template(match.group(1)).safe_substitute(safe_values)
    st.markdown(html, unsafe_allow_html=True)


def render_result(is_benign: bool, confidence: float | None) -> None:
    render_html("result_benign" if is_benign else "result_malignant")
    if is_benign:
        st.markdown(
            "The model classified this sample as **benign** from the entered "
            "measurements. This is an educational model output, not a clinical finding."
        )
    else:
        st.markdown(
            "The model classified this sample as **malignant** from the entered "
            "measurements. This is not a diagnosis; consult a qualified clinician."
        )

    if confidence is not None:
        st.markdown("**Model score**")
        st.progress(min(max(int(round(confidence)), 0), 100))
        st.caption(
            f"Largest predicted class probability: {confidence:.1f}%. "
            "This score is not a measure of clinical certainty."
        )

    with st.expander("How this model works and its limitations"):
        st.markdown(
            "- **Classifier:** Logistic Regression.\n"
            "- **Preprocessing:** StandardScaler transforms the five inputs.\n"
            "- **Limitations:** The model cannot account for a patient's full clinical context."
        )


st.set_page_config(
    page_title="Breast Cancer Classification Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_styles()
render_html("header")
render_html("hero")
render_html("metrics")

model, scaler = load_artifacts()
if model is None or scaler is None:
    st.error("Place `model.pkl` and `scaler.pkl` in the project folder to run the demo.")
    st.stop()

form_column, report_column = st.columns([1.2, 0.8], gap="medium")
with form_column:
    render_html(
        "section_header",
        section_id="measurements",
        index="01",
        title="Try a sample",
        subtitle="Enter measurements to explore the model output.",
    )
    with st.container(border=True):
        st.markdown('<div class="param-group-label">Cell nucleus measurements</div>', unsafe_allow_html=True)
        input_columns = st.columns(3, gap="small")
        values = [0.0] * len(FEATURES)
        for index, feature in enumerate(FEATURES):
            with input_columns[index % 2]:
                values[index] = st.number_input(
                    feature["label"],
                    min_value=feature["minimum"],
                    max_value=feature["maximum"],
                    value=feature["default"],
                    step=feature["step"],
                    help=feature["help"],
                    key=feature["key"],
                )
        analyze = st.button("Analyze sample", type="primary", use_container_width=True)
        st.caption("Enter measurements from a dataset or lab source; defaults are illustrative.")

with report_column:
    render_html(
        "section_header",
        section_id="model-output",
        index="02",
        title="Model output",
        subtitle="See the predicted class and model score.",
    )
    with st.container(border=True):
        if not analyze:
            render_html("empty_state")
        else:
            try:
                with st.spinner("Processing the sample…"):
                    prediction, confidence = classify(model, scaler, values)
                render_result(is_benign=int(prediction) == 1, confidence=confidence)
            except (ValueError, AttributeError, TypeError) as error:
                st.error(f"Could not analyze these inputs: {error}")

render_html("notice")
render_html("footer")
