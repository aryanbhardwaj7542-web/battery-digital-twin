import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Battery Digital Twin",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 AI-Powered Battery Digital Twin")
st.markdown(
    "Explore battery degradation and estimate relative "
    "State of Health using NASA experimental data."
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("nasa_battery_data.csv")

df = load_data()

required_columns = [
    "cycle", "voltage", "current", "temperature", "capacity", "soh"
]

if df.empty or not set(required_columns).issubset(df.columns):
    st.error("Dataset is empty or required columns are missing.")
    st.stop()

df = df.sort_values("cycle").reset_index(drop=True)

# ---------------- TRAIN MODEL ----------------
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
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# ---------------- OVERVIEW ----------------
st.subheader("📊 Battery Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Battery Records", f"{len(df)}")
col2.metric("Latest Recorded SOH", f"{df['soh'].iloc[-1]:.2f}%")
col3.metric("Test MAE", f"{mae:.3f} pp")

st.info(
    "Dataset: NASA battery aging experimental data, cell B0025. "
    "SOH is calculated relative to the first recorded discharge capacity. "
    "This prototype is exploratory and is not a battery safety tool."
)

# ---------------- DEGRADATION CHART ----------------
st.subheader("🔋 Battery Health Degradation")

fig_health = px.line(
    df,
    x="cycle",
    y="soh",
    markers=True,
    title="Relative Battery State of Health Across Discharge Cycles",
    labels={
        "cycle": "Experimental Cycle Number",
        "soh": "Relative SOH (%)"
    },
    template="plotly_dark"
)

fig_health.update_traces(
    line=dict(width=3),
    marker=dict(size=7)
)
fig_health.update_layout(
    height=430,
    margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig_health, use_container_width=True)

# ---------------- PREDICTION INTERFACE ----------------
st.subheader("🧪 Predict Battery Health")
st.caption(
    "Enter measurements to get an exploratory estimate. "
    "Use values in a range similar to the training data."
)

col1, col2 = st.columns(2)

with col1:
    cycle = st.number_input(
        "Experimental cycle number",
        min_value=1,
        max_value=10000,
        value=int(df["cycle"].iloc[-1]),
        step=1
    )

    voltage = st.number_input(
        "Measured voltage (V)",
        min_value=0.0,
        max_value=5.0,
        value=float(df["voltage"].median()),
        step=0.01
    )

with col2:
    current = st.number_input(
        "Measured current (A)",
        min_value=-5.0,
        max_value=100.0,
        value=float(df["current"].median()),
        step=0.01
    )

    temperature = st.number_input(
        "Measured temperature (°C)",
        min_value=-20.0,
        max_value=100.0,
        value=float(df["temperature"].median()),
        step=0.1
    )

if st.button("🔍 Predict Battery Health", type="primary"):
    input_data = pd.DataFrame([{
        "cycle": cycle,
        "voltage": voltage,
        "current": current,
        "temperature": temperature
    }])

    predicted_soh = float(model.predict(input_data)[0])
    predicted_soh = float(np.clip(predicted_soh, 0, 100))

    st.metric("Estimated Relative SOH", f"{predicted_soh:.2f}%")

    if predicted_soh >= 90:
        st.success("🟢 Illustrative status: Excellent")
    elif predicted_soh >= 80:
        st.info("🔵 Illustrative status: Good")
    elif predicted_soh >= 70:
        st.warning("🟠 Illustrative status: Degraded")
    else:
        st.error("🔴 Illustrative status: Poor")

    st.caption(
        "Status thresholds are illustrative, not validated diagnostic "
        "criteria. This prediction cannot establish battery safety."
    )

# ---------------- ACTUAL VS PREDICTED ----------------
st.subheader("📈 Actual vs Predicted Battery Health")

comparison_df = pd.DataFrame({
    "Actual SOH (%)": y_test.to_numpy(),
    "Predicted SOH (%)": predictions
}).sort_values("Actual SOH (%)").reset_index(drop=True)

fig_comparison = go.Figure()

fig_comparison.add_trace(go.Scatter(
    x=comparison_df["Actual SOH (%)"],
    y=comparison_df["Predicted SOH (%)"],
    mode="markers",
    name="Test predictions",
    marker=dict(size=11, symbol="circle")
))

axis_min = float(min(
    comparison_df["Actual SOH (%)"].min(),
    comparison_df["Predicted SOH (%)"].min()
))
axis_max = float(max(
    comparison_df["Actual SOH (%)"].max(),
    comparison_df["Predicted SOH (%)"].max()
))

padding = max((axis_max - axis_min) * 0.1, 0.1)
line_min = axis_min - padding
line_max = axis_max + padding

fig_comparison.add_trace(go.Scatter(
    x=[line_min, line_max],
    y=[line_min, line_max],
    mode="lines",
    name="Ideal prediction",
    line=dict(dash="dash", width=2)
))

fig_comparison.update_layout(
    title="Test Set: Actual SOH vs Predicted SOH",
    xaxis_title="Actual Relative SOH (%)",
    yaxis_title="Predicted Relative SOH (%)",
    template="plotly_dark",
    height=500,
    margin=dict(l=30, r=30, t=60, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
)

fig_comparison.update_xaxes(range=[line_min, line_max])
fig_comparison.update_yaxes(range=[line_min, line_max], scaleanchor="x")
st.plotly_chart(fig_comparison, use_container_width=True)

# ---------------- MODEL METRICS ----------------
st.subheader("🧠 Model Evaluation")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.3f} percentage points")
col2.metric("RMSE", f"{rmse:.3f} percentage points")
col3.metric("R² Score", f"{r2:.3f}")

st.warning(
    "Evaluation uses a random holdout split from only 28 discharge records "
    "of one battery cell. Nearby cycles may be similar, so these results "
    "do not prove generalization to new batteries."
)

# ---------------- DATA TABLE ----------------
with st.expander("📁 View NASA battery dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("ℹ️ About this model"):
    st.write(
        "The Random Forest model uses cycle number, voltage, current and "
        "temperature. Capacity is not an input feature. SOH is relative to "
        "the first recorded discharge capacity, not manufacturer-rated capacity."
    )

# ---------------- BATTERY DEGRADATION FORECAST ----------------
st.subheader("🔮 Future Battery Degradation Forecast")

try:
    forecast_df = pd.read_csv("battery_forecast.csv")

    if not forecast_df.empty:
        fig_forecast = go.Figure()

        fig_forecast.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["soh"],
            mode="lines+markers",
            name="Historical SOH",
            line=dict(width=3)
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_df["cycle"],
            y=forecast_df["forecast_soh"],
            mode="lines+markers",
            name="Experimental Forecast",
            line=dict(dash="dash", width=3)
        ))

        fig_forecast.update_layout(
            title="Historical and Forecast Battery Health",
            xaxis_title="Experimental Cycle Number",
            yaxis_title="Relative SOH (%)",
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        st.caption(
            "Forecast uses a simple linear trend fitted to historical SOH. "
            "It is experimental and is not validated for real-world "
            "battery-life or safety predictions."
        )

        with st.expander("View forecast values"):
            st.dataframe(forecast_df, use_container_width=True)
    else:
        st.warning("Forecast file is empty. Run forecast_battery.py first.")

except FileNotFoundError:
    st.warning(
        "Forecast file not found. Run `python forecast_battery.py` "
        "from your project folder to generate it."
    )
# ---------------- BATTERY HEALTH INSIGHTS ----------------
st.subheader("🧠 Automatic Battery Health Insights")

first_soh = float(df["soh"].iloc[0])
latest_soh = float(df["soh"].iloc[-1])
highest_soh = float(df["soh"].max())
lowest_soh = float(df["soh"].min())
soh_change = latest_soh - first_soh

col1, col2, col3 = st.columns(3)

col1.metric(
    "First Recorded SOH",
    f"{first_soh:.2f}%"
)

col2.metric(
    "Highest Recorded SOH",
    f"{highest_soh:.2f}%"
)

col3.metric(
    "Lowest Recorded SOH",
    f"{lowest_soh:.2f}%"
)

if soh_change < -0.1:
    st.warning(
        f"📉 Relative SOH has decreased by "
        f"{abs(soh_change):.2f} percentage points "
        "since the first recorded cycle."
    )
elif soh_change > 0.1:
    st.info(
        f"📈 Relative SOH is {soh_change:.2f} percentage points "
        "above the first recorded value."
    )
else:
    st.info(
        "Relative SOH is approximately unchanged from the first record."
    )

st.caption(
    "Insights describe the recorded NASA B0025 data only. "
    "Short-term fluctuations can occur, and these measurements "
    "do not establish real-world battery safety or remaining life."
)
