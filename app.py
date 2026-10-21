import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Battery Digital Twin",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 AI-Powered Battery Digital Twin")
st.write("AI-powered battery health prediction and scientific analysis")

# Load the sample dataset
@st.cache_data
def load_data():
    return pd.read_csv("nasa_battery_data.csv")

df = load_data()

# Train the machine learning model
features = ["cycle", "voltage", "current", "temperature"]
X = df[features]
y = df["soh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Dashboard metrics
st.subheader("📊 Battery Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Battery Records", len(df))
col2.metric("Latest Recorded SOH", f"{df['soh'].iloc[-1]:.1f}%")
col3.metric("Model Test MAE", f"{mae:.2f} percentage points")

st.info(
    "Dataset: NASA battery aging experimental data (cell B0025). "
    "SOH is calculated relative to the first recorded discharge capacity. "
    "This is an exploratory prototype, not a validated battery safety tool."
)

# Dataset chart
st.subheader("🔋 Battery Health Degradation")

fig = px.line(
    df,
    x="cycle",
    y="soh",
    markers=True,
    title="Battery State of Health Across Charging Cycles",
    labels={
        "cycle": "Charging Cycle",
        "soh": "State of Health (%)"
    }
)
st.plotly_chart(fig, use_container_width=True)

# Prediction interface
st.subheader("🧪 Predict Battery Health")

col1, col2 = st.columns(2)

with col1:
    cycle = st.number_input(
        "Charging cycle",
        min_value=1,
        max_value=10000,
        value=4
    )
    voltage = st.number_input(
        "Voltage (V)",
        min_value=0.0,
        max_value=5.0,
        value=3.41,
        step=0.01
    )
    current = st.number_input(
        "Current (A)",
        min_value=-5.0,
        max_value=100.0,
        value=-1.07,
        step=0.1
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=100.0,
        value=33.1,
        step=1.0
    )
    capacity = st.number_input(
        "Measured capacity (Ah)",
        min_value=0.1,
        max_value=1000.0,
        value=1.85,
        step=0.01
    )

if st.button("Predict Battery Health", type="primary"):
    input_data = pd.DataFrame([{
        "cycle": cycle,
        "voltage": voltage,
        "current": current,
        "temperature": temperature
        
    }])

    predicted_soh = float(model.predict(input_data)[0])
    predicted_soh = np.clip(predicted_soh, 0, 100)

    st.metric("Estimated Battery SOH", f"{predicted_soh:.2f}%")

    if predicted_soh >= 80:
        st.success("The model estimates battery health at 80% or above.")
    else:
        st.warning("The model estimates battery health below 80%.")

    st.caption(
        "This estimate comes from a small synthetic dataset and is not "
        "suitable for actual battery safety or maintenance decisions."
    )

# Show the data
with st.expander("View sample battery dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("View model evaluation"):
    st.write(f"Mean Absolute Error (MAE): {mae:.3f} percentage points")
    st.write(f"R² score: {r2:.3f}")