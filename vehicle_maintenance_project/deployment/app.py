import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import logging

# =========================================================
# LOGGING CONFIGURATION
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================================================
# FEATURE ENGINEERING FUNCTION
# =========================================================
def add_engine_features(df):
    """
    Engine feature engineering for predictive maintenance.
    """

    df_fe = df.copy()

    # =====================================================
    # 1. THERMAL SYSTEM FEATURES
    # =====================================================

    df_fe["Temp Diff"] = (
        df_fe["Coolant temp"] - df_fe["lub oil temp"]
    )

    df_fe["Temp Ratio"] = (
        df_fe["Coolant temp"] /
        (df_fe["lub oil temp"] + 1e-6)
    )

    df_fe["Temp Excess"] = np.clip(
        df_fe["Coolant temp"] - 90,
        0,
        None
    )

    df_fe["Thermal Load"] = (
        df_fe["Coolant temp"] *
        df_fe["Coolant pressure"]
    )

    # =====================================================
    # 2. ENGINE RPM FEATURES
    # =====================================================

    df_fe["RPM Deviation"] = np.abs(
        df_fe["Engine rpm"] - 650
    )

    df_fe["Log RPM"] = np.log1p(
        df_fe["Engine rpm"]
    )

    df_fe["Inverse RPM"] = 1 / (
        df_fe["Engine rpm"] + 1
    )

    # =====================================================
    # 3. LUBRICATION SYSTEM FEATURES
    # =====================================================

    df_fe["OilPressure per RPM"] = (
        df_fe["Lub oil pressure"] *
        df_fe["Inverse RPM"]
    )

    df_fe["Oil Pressure Deficit"] = np.clip(
        2 - df_fe["Lub oil pressure"],
        0,
        None
    )

    df_fe["Lubrication Stress"] = (
        df_fe["Temp Diff"] /
        (df_fe["Lub oil pressure"] + 1)
    )

    # =====================================================
    # 4. FUEL SYSTEM FEATURES
    # =====================================================

    df_fe["Fuel Pressure Log"] = np.log1p(
        df_fe["Fuel pressure"]
    )

    df_fe["Fuel Deficit"] = np.clip(
        3 - df_fe["Fuel pressure"],
        0,
        None
    )

    df_fe["Fuel Excess"] = np.clip(
        df_fe["Fuel pressure"] - 15,
        0,
        None
    )

    # =====================================================
    # 5. ENGINE STRESS INDEX
    # =====================================================

    df_fe["Engine Stress"] = (
        (
            df_fe["Temp Excess"] + 1
        ) *
        np.log1p(df_fe["Fuel pressure"])
    ) / (
        df_fe["Lub oil pressure"] + 1
    )

    # =====================================================
    # 6. INTERACTION FEATURES
    # =====================================================

    df_fe["RPM_Temp Interaction"] = (
        df_fe["Log RPM"] *
        np.log1p(df_fe["Coolant temp"])
    )

    df_fe["Fuel_RPM_Interaction"] = (
        np.log1p(df_fe["Fuel pressure"]) *
        df_fe["Inverse RPM"]
    )

    df_fe["Fuel_Thermal Interaction"] = (
        np.log1p(df_fe["Fuel pressure"]) *
        np.log1p(df_fe["Temp Excess"] + 1)
    )

    df_fe["Oil_Thermal Interaction"] = (
        df_fe["Lub oil pressure"] *
        df_fe["Temp Excess"]
    )

    # =====================================================
    # 7. CRITICAL FAILURE RISK
    # =====================================================

    df_fe["Critical Thermal Stress"] = (
        df_fe["Temp Excess"] *
        df_fe["Oil Pressure Deficit"]
    )

    return df_fe


# =========================================================
# LOAD MODEL FROM HUGGING FACE
# =========================================================
@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="bkrishnamukund/Vehicle-Engine-Maintenance-Prediction",
        filename="best_Vehicle_Engine_Maintenance_Prediction_model_v1.joblib"
    )

    return joblib.load(model_path)


model = load_model()

# =========================================================
# ORIGINAL INPUT FEATURES
# =========================================================
base_features = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp'
]

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Vehicle Engine Maintenance Prediction",
    page_icon="🚗",
    layout="wide"
)

# =========================================================
# APP TITLE
# =========================================================
st.title("🚗 Vehicle Engine Maintenance Prediction")

st.write("""
Predict vehicle engine health using sensor readings and advanced
engineered features for predictive maintenance.
""")

# =========================================================
# FEATURE DESCRIPTION
# =========================================================
with st.expander("📘 Feature Description"):

    st.markdown("""
    ### Base Features

    - **Engine rpm** → Engine speed in RPM  
    - **Lub oil pressure** → Lubrication oil pressure  
    - **Fuel pressure** → Fuel system pressure  
    - **Coolant pressure** → Cooling system pressure  
    - **lub oil temp** → Lubricating oil temperature  
    - **Coolant temp** → Coolant temperature  

    ### Engineered Features

    - Thermal imbalance indicators
    - Lubrication stress metrics
    - Fuel system stability metrics
    - RPM nonlinear transformations
    - Cross-system interaction features
    - Critical engine failure indicators
    """)

# =========================================================
# SINGLE PREDICTION
# =========================================================
st.header("🔍 Single Engine Prediction")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        engine_rpm = st.number_input(
            "Engine RPM",
            min_value=0.0,
            value=1500.0,
            step=10.0
        )

        lub_oil_pressure = st.number_input(
            "Lub Oil Pressure",
            min_value=0.0,
            value=3.0,
            step=0.1
        )

        fuel_pressure = st.number_input(
            "Fuel Pressure",
            min_value=0.0,
            value=15.0,
            step=0.1
        )

    with col2:

        coolant_pressure = st.number_input(
            "Coolant Pressure",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

        lub_oil_temp = st.number_input(
            "Lub Oil Temperature (°C)",
            value=80.0,
            step=1.0
        )

        coolant_temp = st.number_input(
            "Coolant Temperature (°C)",
            value=90.0,
            step=1.0
        )

    submit = st.form_submit_button(
        "Predict Engine Condition"
    )

    if submit:

        # ===============================================
        # CREATE INPUT DATAFRAME
        # ===============================================

        input_df = pd.DataFrame([{
            'Engine rpm': engine_rpm,
            'Lub oil pressure': lub_oil_pressure,
            'Fuel pressure': fuel_pressure,
            'Coolant pressure': coolant_pressure,
            'lub oil temp': lub_oil_temp,
            'Coolant temp': coolant_temp
        }])

        # ===============================================
        # FEATURE ENGINEERING
        # ===============================================

        input_df_fe = add_engine_features(input_df)

        # ===============================================
        # PREDICTION
        # ===============================================

        prediction = model.predict(input_df_fe)[0]

        probability = model.predict_proba(
            input_df_fe
        )[0][1]

        # ===============================================
        # LOGGING
        # ===============================================

        logging.info(
            "Input Features: %s",
            input_df_fe.to_dict(orient="records")
        )

        logging.info(
            "Prediction=%s | Probability=%.4f",
            prediction,
            probability
        )

        # ===============================================
        # DISPLAY RESULT
        # ===============================================

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"⚠️ Engine Fault Detected "
                f"(Probability: {probability:.2f})"
            )

        else:

            st.success(
                f"✅ Engine Operating Normally "
                f"(Confidence: {1 - probability:.2f})"
            )

        # ===============================================
        # SHOW ENGINEERED FEATURES
        # ===============================================

        with st.expander("🧠 Engineered Features"):

            engineered_cols = [
                col for col in input_df_fe.columns
                if col not in base_features
            ]

            st.dataframe(
                input_df_fe[engineered_cols].T.rename(
                    columns={0: "Value"}
                )
            )

# =========================================================
# BATCH PREDICTION
# =========================================================
st.header("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")

    st.dataframe(df.head())

    st.write("### Required Columns")

    st.code("""
Engine rpm
Lub oil pressure
Fuel pressure
Coolant pressure
lub oil temp
Coolant temp
""")

    if st.button("Predict Batch"):

        try:

            # ===========================================
            # KEEP ONLY REQUIRED FEATURES
            # ===========================================

            df_input = df[base_features]

            # ===========================================
            # FEATURE ENGINEERING
            # ===========================================

            df_fe = add_engine_features(df_input)

            # ===========================================
            # PREDICTIONS
            # ===========================================

            preds = model.predict(df_fe)

            probs = model.predict_proba(df_fe)[:, 1]

            # ===========================================
            # OUTPUT DATAFRAME
            # ===========================================

            df_out = df.copy()

            df_out["Predicted_Engine_Condition"] = preds

            df_out["Fault_Probability"] = np.round(
                probs,
                2
            )

            # ===========================================
            # DISPLAY RESULTS
            # ===========================================

            st.success(
                "✅ Batch Prediction Completed!"
            )

            st.dataframe(df_out)

            # ===========================================
            # DOWNLOAD BUTTON
            # ===========================================

            csv = df_out.to_csv(index=False)

            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="vehicle_engine_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(f"Error: {str(e)}")

