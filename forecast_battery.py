import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def forecast_battery(data, future_cycles=20):
    """
    Estimate future relative battery SOH using a linear trend.
    This is an experimental trend forecast, not a validated
    battery-life or safety prediction.
    """

    required_columns = {"cycle", "soh"}

    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must contain cycle and soh columns.")

    history = (
        data[["cycle", "soh"]]
        .dropna()
        .groupby("cycle", as_index=False)["soh"]
        .mean()
        .sort_values("cycle")
    )

    if len(history) < 3:
        raise ValueError("At least 3 historical cycles are required.")

    if future_cycles < 1:
        raise ValueError("future_cycles must be at least 1.")

    X = history[["cycle"]]
    y = history["soh"]

    model = LinearRegression()
    model.fit(X, y)

    last_cycle = int(history["cycle"].max())

    future = pd.DataFrame({
        "cycle": np.arange(
            last_cycle + 1,
            last_cycle + future_cycles + 1
        )
    })

    future["forecast_soh"] = model.predict(future[["cycle"]])

    # Keep estimates within the relative SOH percentage range.
    future["forecast_soh"] = future["forecast_soh"].clip(0, 100)

    return history, future


if __name__ == "__main__":
    data = pd.read_csv("nasa_battery_data.csv")

    history, forecast = forecast_battery(data, future_cycles=20)

    print("\nRecent historical battery health:")
    print(history.tail())

    print("\nExperimental future SOH forecast:")
    print(forecast.to_string(index=False))

    forecast.to_csv("battery_forecast.csv", index=False)
    print("\nForecast saved to battery_forecast.csv")
