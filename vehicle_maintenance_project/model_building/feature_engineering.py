import numpy as np

def add_engine_features(df):
    """
    Engine feature engineering for predictive maintenance.

    Creates stable nonlinear transformations for thermal, fuel, RPM,
    and lubrication systems while improving signal quality for ML models.
    """

    df_fe = df.copy()

    # ----------------------------
    # 1. Thermal system features
    # Capture heat imbalance and overheating intensity across engine subsystems
    # ----------------------------

    # Difference between cooling system and lubrication oil temperature
    df_fe["Temp Diff"] = df_fe["Coolant temp"] - df_fe["lub oil temp"]

    # Ratio captures relative thermal imbalance between coolant and oil system
    df_fe["Temp Ratio"] = df_fe["Coolant temp"] / (df_fe["lub oil temp"] + 1e-6)

    # Captures overheating beyond normal operating threshold (90°C baseline)
    df_fe["Temp Excess"] = np.clip(df_fe["Coolant temp"] - 90, 0, None)

    # Represents overall thermal load combining heat and cooling pressure
    df_fe["Thermal Load"] = df_fe["Coolant temp"] * df_fe["Coolant pressure"]

    # ----------------------------
    # 2. Engine RPM features
    # Capture operating regime, engine load intensity, and nonlinear scaling effects
    # ----------------------------

    # Distance from optimal operating RPM (650 baseline)
    df_fe["RPM Deviation"] = np.abs(df_fe["Engine rpm"] - 650)

    # Log transform reduces skewness of RPM distribution
    df_fe["Log RPM"] = np.log1p(df_fe["Engine rpm"])

    # Inverse scaling captures stress at low RPM conditions
    df_fe["Inverse RPM"] = 1 / (df_fe["Engine rpm"] + 1)

    # ----------------------------
    # 3. Lubrication system features
    # Measure oil system efficiency and detect lubrication failure risk
    # ----------------------------

    # Oil pressure efficiency relative to engine speed
    df_fe["OilPressure per RPM"] = df_fe["Lub oil pressure"] * df_fe["Inverse RPM"]

    # Measures how far oil pressure drops below safe operating threshold (2 bar)
    df_fe["Oil Pressure Deficit"] = np.clip(2 - df_fe["Lub oil pressure"], 0, None)

    # Represents lubrication stress due to thermal and pressure imbalance
    df_fe["Lubrication Stress"] = (
        df_fe["Temp Diff"] / (df_fe["Lub oil pressure"] + 1)
    )

    # ----------------------------
    # 4. Fuel system features
    # Capture fuel supply stability, underfeeding, and overpressure conditions
    # ----------------------------

    # Log transform stabilizes skewed fuel pressure distribution
    df_fe["Fuel Pressure Log"] = np.log1p(df_fe["Fuel pressure"])

    # Measures fuel starvation below minimum safe threshold (3 bar)
    df_fe["Fuel Deficit"] = np.clip(3 - df_fe["Fuel pressure"], 0, None)

    # Measures excessive fuel pressure above safe operating range (15 bar)
    df_fe["Fuel Excess"] = np.clip(df_fe["Fuel pressure"] - 15, 0, None)

    # ----------------------------
    # 5. Engine stress index
    # Combined nonlinear representation of thermal + fuel + lubrication stress
    # ----------------------------

    # Combines thermal excess and fuel pressure with lubrication stability
    df_fe["Engine Stress"] = (
        (df_fe["Temp Excess"] + 1) *
        np.log1p(df_fe["Fuel pressure"])
    ) / (df_fe["Lub oil pressure"] + 1)

    # ----------------------------
    # 6. Interaction features
    # Capture cross-system nonlinear relationships between RPM, fuel, and thermal states
    # ----------------------------

    # Interaction between engine speed and thermal load
    df_fe["RPM_Temp Interaction"] = (
        df_fe["Log RPM"] * np.log1p(df_fe["Coolant temp"])
    )

    # Interaction between fuel system and engine speed stress
    df_fe["Fuel_RPM_Interaction"] = (
        np.log1p(df_fe["Fuel pressure"]) * df_fe["Inverse RPM"]
    )

    # Interaction capturing fuel stress amplified by overheating conditions
    df_fe["Fuel_Thermal Interaction"] = (
        np.log1p(df_fe["Fuel pressure"]) * np.log1p(df_fe["Temp Excess"] + 1)
    )

    # Interaction between lubrication system and thermal overload
    df_fe["Oil_Thermal Interaction"] = (
        df_fe["Lub oil pressure"] * df_fe["Temp Excess"]
    )

    # ----------------------------
    # 7. Critical failure risk indicator
    # Captures high-risk condition when overheating and oil stress occur together
    # ----------------------------

    df_fe["Critical Thermal Stress"] = (
        df_fe["Temp Excess"] * df_fe["Oil Pressure Deficit"]
    )

    return df_fe
