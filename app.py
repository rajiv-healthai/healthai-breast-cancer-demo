import streamlit as st
import numpy as np
import joblib
import os
import time

# 1. Page Configuration
st.set_page_config(
    page_title="Breast Cancer Classification Demo • HealthAI Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom Precision Scientific Instrument & Dark Health-Tech CSS
st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background-color: #080c14;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1280px;
    }

    /* Top Precision Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 2rem;
    }
    .nav-brand {
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: -0.02em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .nav-badge {
        font-size: 0.65rem;
        background: rgba(14, 165, 233, 0.1);
        color: #38bdf8;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .nav-meta {
        font-size: 0.8rem;
        color: #64748b;
        font-family: monospace;
    }

    /* Editorial Hero Section with Fine-Line Scientific Motif */
    .hero-container {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 2.25rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease;
    }
    .hero-container:hover {
        border-color: rgba(56, 189, 248, 0.25);
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 2rem;
        align-items: center;
    }
    @media (max-width: 900px) {
        .hero-grid { grid-template-columns: 1fr; }
    }
    .hero-eyebrow {
        font-family: monospace;
        font-size: 0.75rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }
    .hero-description {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 780px;
        margin-bottom: 1.25rem;
    }
    .hero-meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        font-size: 0.775rem;
        color: #94a3b8;
        font-family: monospace;
        border-top: 1px solid #1e293b;
        padding-top: 1rem;
    }
    .hero-meta-row span {
        color: #38bdf8;
    }

    /* Scientific Motif Illustration SVG styling */
    .scientific-motif {
        width: 140px;
        height: 140px;
        opacity: 0.85;
    }

    /* Educational Notice Banner */
    .notice-banner {
        background: #0f172a;
        border: 1px solid #312e81;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 0.875rem 1.25rem;
        font-size: 0.85rem;
        color: #c7d2fe;
        margin-bottom: 2rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        line-height: 1.5;
    }

    /* Section Header with Localized Hover Lighting on Accent Line */
    .section-header-wrap {
        margin-bottom: 1rem;
        cursor: default;
    }
    .section-title-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.2rem;
    }
    .section-index {
        font-family: monospace;
        font-size: 0.75rem;
        color: #38bdf8;
        background: rgba(14, 165, 233, 0.1);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .section-main-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .section-accent-line {
        height: 2px;
        width: 24px;
        background: #334155;
        border-radius: 2px;
        margin-top: 0.4rem;
        transition: width 0.25s ease, background-color 0.25s ease;
    }
    .section-header-wrap:hover .section-accent-line {
        width: 48px;
        background: #38bdf8;
    }
    .section-subtitle {
        font-size: 0.825rem;
        color: #94a3b8;
    }

    /* Workspace Surfaces & Parameter Groups with Localized Hover Borders */
    .instrument-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.5rem;
        transition: border-color 0.2s ease;
    }
    .instrument-card:hover {
        border-color: #334155;
    }

    .param-group {
        border-left: 2px solid #334155;
        padding-left: 1rem;
        margin-bottom: 1.25rem;
        transition: border-color 0.2s ease;
    }
    .param-group:hover {
        border-left-color: #38bdf8;
    }
    .param-group-label {
        font-size: 0.725rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.75rem;
        font-family: monospace;
        transition: color 0.2s ease;
    }
    .param-group:hover .param-group-label {
        color: #38bdf8;
    }

    /* Empty State Box */
    .empty-state-box {
        text-align: center;
        padding: 4.5rem 1.5rem;
        color: #64748b;
    }
    .empty-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
        opacity: 0.7;
    }

    /* Result Status Badges with Precise Borders */
    .badge-benign {
        background: rgba(6, 95, 70, 0.25);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 0.5rem 0.875rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .badge-malignant {
        background: rgba(153, 27, 27, 0.25);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.35);
        padding: 0.5rem 0.875rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Report Subsection Separators */
    .report-divider {
        height: 1px;
        background: #1e293b;
        margin: 1.25rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1e293b;
        font-size: 0.775rem;
        color: #64748b;
        font-family: monospace;
        line-height: 1.6;
    }

    /* Streamlit Control Polish & Focus States */
    div[data-baseweb="input"] {
        background-color: #080c14 !important;
        border-color: #334155 !important;
        color: #f1f5f9 !important;
        transition: border-color 0.2s ease;
    }
    div[data-baseweb="input"]:hover {
        border-color: #475569 !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }
    div[data-baseweb="base-input"] input {
        color: #f1f5f9 !important;
    }
    .stButton button {
        background: #0284c7 !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    .stButton button:hover {
        background: #0ea5e9 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
    }

    /* Visible Keyboard Focus */
    button:focus-visible, input:focus-visible, [tabindex]:focus-visible {
        outline: 2px solid #38bdf8 !important;
        outline-offset: 2px !important;
    }

    /* Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        .hero-container, .section-accent-line, .stButton button {
            transition: none !important;
            transform: none !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# 3. Top Precision Navigation Bar
st.markdown("""
    <div class="top-nav">
        <div class="nav-brand">
            <span>🔬 HealthAI Research Lab</span>
            <span class="nav-badge">Secure Demo</span>
        </div>
        <div class="nav-meta">
            SYS.ID: 26ECP-102 // CHANDIGARH.UNI
        </div>
    </div>
""", unsafe_allow_html=True)

# 4. Editorial Hero Section with Scientific Motif SVG & Updated Headline/Subtitle
st.markdown("""
    <div class="hero-container">
        <div class="hero-grid">
            <div>
                <div class="hero-eyebrow">Cellular Morphology Classifier</div>
                <div class="hero-title">Breast Cancer Classification Demo</div>
                <div class="hero-description">
                    Explore how a machine learning model classifies breast tumor samples as benign or malignant using cell measurements. 
                    This precision demonstration evaluates digitized Fine Needle Aspiration (FNA) feature vectors via a trained Logistic Regression pipeline.
                </div>
                <div class="hero-meta-row">
                    <div>ARCH: <span>Logistic Regression</span></div>
                    <div>NORM: <span>StandardScaler</span></div>
                    <div>DB: <span>Wisconsin Diagnostic</span></div>
                    <div>DEV: <span>Divyanshu Kumar Thakur (26BCS10287)</span></div>
                </div>
            </div>
            <div style="display: flex; justify-content: center;">
                <svg class="scientific-motif" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="50" cy="50" r="42" stroke="#1e293b" stroke-width="1.5" stroke-dasharray="4 4"/>
                    <circle cx="50" cy="50" r="30" stroke="#0284c7" stroke-width="1.5" opacity="0.4"/>
                    <circle cx="50" cy="50" r="18" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
                    <circle cx="50" cy="50" r="6" fill="#38bdf8" opacity="0.9"/>
                    <line x1="50" y1="8" x2="50" y2="92" stroke="#1e293b" stroke-width="1"/>
                    <line x1="8" y1="50" x2="92" y2="50" stroke="#1e293b" stroke-width="1"/>
                    <circle cx="71" cy="29" r="3" fill="#0ea5e9"/>
                    <circle cx="29" cy="71" r="2" fill="#0284c7"/>
                </svg>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 5. Educational Notice Banner
st.markdown("""
    <div class="notice-banner">
        <span style="font-size: 1rem; color: #818cf8;">ⓘ</span>
        <div>
            <b>Academic Demonstration Notice:</b> This software is built strictly for educational evaluation in the AI Applications Lab. 
            Outputs are generated by a statistical benchmark model using benchmark datasets and <b>do not constitute formal medical diagnosis, 
            clinical prognosis, or personal health advice</b>.
        </div>
    </div>
""", unsafe_allow_html=True)

# 6. Load Model & Scaler Artifacts with Caching & Error Handling
@st.cache_resource
def load_artifacts():
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
        return None, None
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("⚠️ **Model Artifacts Missing:** `model.pkl` and `scaler.pkl` were not detected in the project directory. Please place them alongside `app.py`.")
    st.stop()

# 7. Two-Column Workspace (Inputs vs. Report Output)
col_form, col_report = st.columns([1.1, 1.2], gap="large")

with col_form:
    st.markdown("""
        <div class="section-header-wrap">
            <div class="section-title-row">
                <span class="section-index">01</span>
                <span class="section-main-title">Specimen Parameters</span>
            </div>
            <div class="section-accent-line"></div>
            <div class="section-subtitle" style="margin-top: 0.5rem;">Input digitized cell nucleus dimensions and topological attributes.</div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="instrument-card" style="border:none; padding:0;">', unsafe_allow_html=True)
        
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        st.markdown('<div class="param-group-label">Morphological Dimensions</div>', unsafe_allow_html=True)
        mean_radius = st.number_input(
            "Mean Radius (mm)", 
            min_value=0.0, max_value=40.0, value=14.0, step=0.1, 
            help="Average distance from center to points on the nuclear perimeter."
        )
        mean_perimeter = st.number_input(
            "Mean Perimeter (mm)", 
            min_value=0.0, max_value=250.0, value=90.0, step=0.5, 
            help="Total outer perimeter length of the cellular nucleus core."
        )
        mean_area = st.number_input(
            "Mean Area (mm²)", 
            min_value=0.0, max_value=3000.0, value=600.0, step=5.0, 
            help="Computed surface area enclosed by the nuclear boundary."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        st.markdown('<div class="param-group-label">Surface Topology & Texture</div>', unsafe_allow_html=True)
        mean_texture = st.number_input(
            "Mean Texture (Standard Deviation)", 
            min_value=0.0, max_value=50.0, value=20.0, step=0.1, 
            help="Standard deviation of gray-scale intensity values across the cell region."
        )
        mean_concavity = st.number_input(
            "Mean Concavity", 
            min_value=0.0, max_value=1.0, value=0.10, step=0.01, 
            help="Severity and depth of concave portions along the nuclear contour."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("⚡ Execute Classification Pipeline", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_report:
    st.markdown("""
        <div class="section-header-wrap">
            <div class="section-title-row">
                <span class="section-index">02</span>
                <span class="section-main-title">Diagnostic Analysis Report</span>
            </div>
            <div class="section-accent-line"></div>
            <div class="section-subtitle" style="margin-top: 0.5rem;">Real-time model inference, probability estimation, and evaluation metrics.</div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="instrument-card" style="border:none; padding:0;">', unsafe_allow_html=True)
        if not analyze_btn:
            # Designed Empty State
            st.markdown("""
                <div class="empty-state-box">
                    <div class="empty-icon">🔬</div>
                    <div style="font-weight: 600; font-size: 1rem; color: #f1f5f9; margin-bottom: 0.35rem;">Awaiting Specimen Analysis</div>
                    <div style="font-size: 0.85rem; max-width: 320px; margin: 0 auto; line-height: 1.5; color: #94a3b8;">
                        Configure the clinical parameters on the left and execute the pipeline to generate a structured diagnostic readout.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Loading State
            with st.spinner("Processing feature vector through StandardScaler & Logistic Regression..."):
                time.sleep(0.3)
                input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_concavity]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                
                try:
                    probabilities = model.predict_proba(input_scaled)[0]
                    confidence = np.max(probabilities) * 100
                except AttributeError:
                    confidence = None

            st.markdown('<div class="report-divider"></div>', unsafe_allow_html=True)
            
            # Render Result Report
            if prediction[0] == 1:
                st.markdown('<div class="badge-benign">🟢 Benign Specimen (Non-Cancerous)</div>', unsafe_allow_html=True)
                st.markdown(
                    "<p style='font-size: 0.9rem; line-height: 1.6; color: #cbd5e1; margin-top: 0.875rem;'>"
                    "<b>Interpretation:</b> Morphological parameters align within standard non-malignant thresholds. "
                    "Cellular boundaries and concavity indices show normal structural stability."
                    "</p>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="badge-malignant">🔴 Malignant Specimen (Cancerous)</div>', unsafe_allow_html=True)
                st.markdown(
                    "<p style='font-size: 0.9rem; line-height: 1.6; color: #cbd5e1; margin-top: 0.875rem;'>"
                    "<b>Interpretation:</b> Anomalous nuclear configurations and elevated concavity severity detected. "
                    "In a clinical context, this specimen would require secondary pathological confirmation."
                    "</p>", 
                    unsafe_allow_html=True
                )
            
            if confidence is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Model Confidence Estimation:** `{confidence:.2f}%`")
                st.progress(int(confidence))
            
            st.markdown('<div class="report-divider"></div>', unsafe_allow_html=True)
            with st.expander("🔍 Pipeline Transparency & Technical Limitations"):
                st.markdown(
                    "* **Classifier:** Supervised Logistic Regression binary classifier.\n"
                    "* **Normalization:** Zero-mean and unit-variance feature scaling via `StandardScaler`.\n"
                    "* **Model Limitation:** As a linear classifier, the model relies on linear hyperplanes in feature space and cannot model complex multi-dimensional non-linear interactions without polynomial feature expansion."
                )
        st.markdown('</div>', unsafe_allow_html=True)

# 8. Footer Credits
st.markdown("""
    <div class="footer">
        DEVELOPED BY DIVYANSHU KUMAR THAKUR (UID: 26BCS10287) • CSE DEPT • CHANDIGARH UNIVERSITY<br>
        AI APPLICATIONS LAB (26ECP-102) • ACADEMIC RESEARCH DEMONSTRATION
    </div>
""", unsafe_allow_html=True)