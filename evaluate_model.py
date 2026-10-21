import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

df = pd.read_csv("nasa_battery_data.csv")

features = ["cycle", "voltage", "current", "temperature", "capacity"]
X = df[features]
y = df["soh"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
predictions = cross_val_predict(model, X, y, cv=LeaveOneOut())

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

print("NASA Battery Model Evaluation")
print("-----------------------------")
print("Records:", len(df))
print(f"MAE: {mae:.4f} percentage points")
print(f"RMSE: {rmse:.4f} percentage points")
print(f"R2 score: {r2:.4f}")

print("\nFirst 5 predictions:")
for actual, predicted in zip(y.iloc[:5], predictions[:5]):
    print(f"Actual: {actual:.2f}% | Predicted: {predicted:.2f}%")

print("\nNote: Results are exploratory, using one battery's data.")
