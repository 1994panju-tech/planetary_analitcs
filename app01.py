import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI freezes
import matplotlib.pyplot as plt
import shap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Asteroid Intelligence System",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded",)

st.title("☄️ Planetary Defense & Asteroid Intelligence System")
st.markdown(
    """
    This interactive dashboard analyzes real-time user inputs across three machine learning cases:
    * **Case 1 (Classification):** Potentially Hazardous Asteroid (PHA) Risk Assessment + SHAP Explanation
    * **Case 2 (Regression):** Physical Diameter Estimation ($\text{km}$) + SHAP Explanation
    * **Case 3 (Clustering):** Keplerian Orbital Population Mapping
    """)

# ------------------------------------------------------------------------------
# 2. LOAD TRAINED MODELS & ARTIFACTS
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    clf_path = "saved_models/best_case1_pha_classifier.joblib"
    reg_path = "saved_models/best_case2_diameter_regressor.joblib"

    clf = joblib.load(clf_path) if os.path.exists(clf_path) else None
    reg = joblib.load(reg_path) if os.path.exists(reg_path) else None
    return clf, reg

model_clf, model_reg = load_models()

# ------------------------------------------------------------------------------
# 3. SIDEBAR USER INPUTS & DYNAMIC FEATURE ENGINEERING
# ------------------------------------------------------------------------------
st.sidebar.header("📥 Asteroid Physical & Orbital Inputs")

# Physical & Orbital Inputs
H = st.sidebar.number_input("Absolute Magnitude (H)", min_value=1.0, max_value=35.0, value=18.5, step=0.1)
a = st.sidebar.number_input("Semi-Major Axis (a) [AU]", min_value=0.1, max_value=100.0, value=1.45, step=0.01)
e = st.sidebar.number_input("Eccentricity (e)", min_value=0.0, max_value=0.99, value=0.22, step=0.01)
i = st.sidebar.number_input("Inclination (i) [deg]", min_value=0.0, max_value=180.0, value=10.5, step=0.1)
q = st.sidebar.number_input("Perihelion Distance (q) [AU]", min_value=0.01, max_value=50.0, value=1.05, step=0.01)
ad = st.sidebar.number_input("Aphelion Distance (ad) [AU]", min_value=0.1, max_value=200.0, value=1.85, step=0.01)
n = st.sidebar.number_input("Mean Motion (n) [deg/day]", min_value=0.0001, max_value=10.0, value=0.52, step=0.01)
albedo = st.sidebar.slider("Geometric Albedo (Reflectivity)", min_value=0.01, max_value=0.90, value=0.15, step=0.01)

# Angular Elements (Default standard values)
om = st.sidebar.number_input("Longitude of Ascending Node (om) [deg]", value=100.0, step=1.0)
w = st.sidebar.number_input("Argument of Perihelion (w) [deg]", value=150.0, step=1.0)
ma = st.sidebar.number_input("Mean Anomaly (ma) [deg]", value=200.0, step=1.0)

# Orbital Class Selection (Includes IEO)
selected_class = st.sidebar.selectbox(
    "Asteroid Classification Class",
    ["AMO", "APO", "AST", "ATE", "CEN", "IEO", "IMB", "MBA", "MCA", "OMB", "TJN", "TNO"],
    index=1,)  # Default to APO (Apollo)


# --- BASE FEATURE CALCULATIONS ---
log_a = np.log(a) if a > 0 else 0
log_q = np.log(q) if q > 0 else 0
log_ad = np.log(ad) if ad > 0 else 0
log_n = np.log(n) if n > 0 else 0
per_y = a ** 1.5  # Kepler's 3rd Law period in years
log_per = np.log(per_y) if per_y > 0 else 0

# Base continuous features
base_features = {
    "e": e, "i": i, "H": H, "om": om, "w": w, "ma": ma,
    "log_a": log_a, "log_q": log_q, "log_ad": log_ad, "log_per": log_per, "log_n": log_n}

# --- 1. BUILD 23-FEATURE DATAFRAME FOR CLASSIFIER (Includes class_IEO) ---
classes_23 = [
    "class_AMO", "class_APO", "class_AST", "class_ATE", "class_CEN", "class_IEO",
    "class_IMB", "class_MBA", "class_MCA", "class_OMB", "class_TJN", "class_TNO"]

class_encoded_23 = {c: 1 if c == f"class_{selected_class}" else 0 for c in classes_23}
input_dict_23 = {**base_features, **class_encoded_23}

expected_23_features = [
    "e", "i", "H", "om", "w", "ma",
    "log_a", "log_q", "log_ad", "log_per", "log_n",
    "class_AMO", "class_APO", "class_AST", "class_ATE", "class_CEN", "class_IEO",
    "class_IMB", "class_MBA", "class_MCA", "class_OMB", "class_TJN", "class_TNO"]

df_input_clf = pd.DataFrame([input_dict_23])[expected_23_features]

# --- 2. BUILD 22-FEATURE DATAFRAME FOR REGRESSOR (Excludes class_IEO) ---
classes_22 = [
    "class_AMO", "class_APO", "class_AST", "class_ATE", "class_CEN",
    "class_IMB", "class_MBA", "class_MCA", "class_OMB", "class_TJN", "class_TNO"]

class_encoded_22 = {c: 1 if c == f"class_{selected_class}" else 0 for c in classes_22}
input_dict_22 = {**base_features, **class_encoded_22}

expected_22_features = [
    "e", "i", "H", "om", "w", "ma",
    "log_a", "log_q", "log_ad", "log_per", "log_n",
    "class_AMO", "class_APO", "class_AST", "class_ATE", "class_CEN",
    "class_IMB", "class_MBA", "class_MCA", "class_OMB", "class_TJN", "class_TNO"]

df_input_reg = pd.DataFrame([input_dict_22])[expected_22_features]

# --- 3. INPUT DATAFRAME FOR CLUSTERING ---
df_cluster_input = pd.DataFrame([[a, e, i, q, ad, n]], columns=["a", "e", "i", "q", "ad", "n"])

# ------------------------------------------------------------------------------
# 4. DASHBOARD TABS & OUTPUTS
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🛡️ Case 1: PHA Classification & SHAP",
    "📏 Case 2: Diameter Regression & SHAP",
    "🌌 Case 3: Orbital Population Clustering"])

# --- TAB 1: CLASSIFICATION ---
with tab1:
    st.subheader("Case 1: Potentially Hazardous Asteroid (PHA) Classification")

    if model_clf is not None:
        try:
            prob_pha = model_clf.predict_proba(df_input_clf)[0][1]
            is_pha = prob_pha >= 0.50

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Predicted Hazard Probability", value=f"{prob_pha * 100:.2f}%")
                if is_pha:
                    st.error("🚨 ALERT: Classified as Potentially Hazardous (PHA)")
                else:
                    st.success("✅ SAFE: Classified as Non-Hazardous")

                st.caption("A body is defined as a PHA if its minimum orbit intersection distance (MOID) is small and its absolute magnitude $H \\le 22.0$.")

            with col2:
                st.markdown("#### Local SHAP Prediction Explanation (Waterfall)")
                try:
                    explainer = shap.TreeExplainer(model_clf)
                    shap_vals = explainer(df_input_clf)
                    if len(shap_vals.shape) == 3:
                        shap_vals = shap_vals[:, :, 1]

                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.plots.waterfall(shap_vals[0], show=False)
                    st.pyplot(fig)
                except Exception as ex:
                    st.error(f"SHAP Error: {ex}")
        except Exception as ex:
            st.error(f"Classification Execution Error: {ex}")
    else:
        st.warning("Classifier model file missing at `saved_models/best_case1_pha_classifier.joblib`.")

# --- TAB 2: REGRESSION ---
with tab2:
    st.subheader("Case 2: Asteroid Physical Diameter Estimation")

    if model_reg is not None:
        try:
            log_d = model_reg.predict(df_input_reg)[0]
            est_d = np.exp(log_d) if log_d < 15 else log_d

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Estimated Diameter", value=f"{est_d:.3f} km")
                st.caption("Estimated via LightGBM/XGBoost using engineered physical & orbital features.")

            with col2:
                st.markdown("#### Local SHAP Prediction Explanation (Waterfall)")
                try:
                    explainer_reg = shap.TreeExplainer(model_reg)
                    shap_vals_reg = explainer_reg(df_input_reg)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.plots.waterfall(shap_vals_reg[0], show=False)
                    st.pyplot(fig)
                except Exception as ex:
                    st.error(f"SHAP Error: {ex}")

        except Exception as ex:
            st.error(f"Regression Execution Error: {ex}")
    else:
        # Physical Rule-of-Thumb fallback calculation
        d_calc = (1328 / np.sqrt(albedo)) * (10 ** (-0.2 * H))
        st.metric(label="Estimated Diameter (Physical Formula)", value=f"{d_calc:.3f} km")

# --- TAB 3: CLUSTERING ---
with tab3:
    st.subheader("Case 3: Keplerian Orbital Population Mapping")

    # Generate synthetic Keplerian background population for reference mapping
    np.random.seed(42)
    ref_a = np.random.uniform(0.8, 4.5, 400)
    ref_e = np.random.uniform(0.01, 0.6, 400)
    ref_data = pd.DataFrame({
        "a": ref_a,
        "e": ref_e,
        "i": np.random.uniform(0.5, 30.0, 400),
        "q": ref_a * (1 - ref_e),
        "ad": ref_a * (1 + ref_e),
        "n": 0.9856 / (ref_a**1.5),})

    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(ref_data)
    user_scaled = scaler.transform(df_cluster_input)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    ref_clusters = kmeans.fit_predict(ref_scaled)
    user_cluster = kmeans.predict(user_scaled)[0]

    st.markdown(f"### Assigned Cluster: **Cluster {user_cluster}**")

    # Render 2D Orbital Map (Semi-Major Axis 'a' vs. Eccentricity 'e')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        ref_data["a"], ref_data["e"],
        c=ref_clusters, cmap="viridis", alpha=0.4, s=25, label="Background Asteroid Population")
    ax.scatter(
        [a], [e],
        color="red", s=200, marker="*", edgecolor="black", linewidth=1.5,
        label=f"Input Asteroid (Cluster {user_cluster})" )

    ax.set_title("Orbital Parameter Map (Semi-Major Axis 'a' vs. Eccentricity 'e')", fontsize=12)
    ax.set_xlabel("Semi-Major Axis a (AU)", fontsize=10)
    ax.set_ylabel("Eccentricity e", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    st.pyplot(fig)




