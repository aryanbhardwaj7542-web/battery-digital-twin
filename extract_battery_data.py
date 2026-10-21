from scipy.io import loadmat
import pandas as pd

# Load the NASA battery dataset
mat = loadmat("B0025.mat", simplify_cells=True)
cycles = mat["B0025"]["cycle"]

records = []

for cycle_number, cycle in enumerate(cycles, start=1):
    if cycle["type"] != "discharge":
        continue

    data = cycle["data"]
    capacity = data.get("Capacity")

    if capacity is None:
        continue

    capacity = float(capacity)

    records.append({
        "cycle": cycle_number,
        "voltage": float(pd.Series(data["Voltage_measured"]).mean()),
        "current": float(pd.Series(data["Current_measured"]).mean()),
        "temperature": float(pd.Series(data["Temperature_measured"]).mean()),
        "capacity": capacity,
    })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No discharge capacity records were extracted.")

# Use the first recorded discharge capacity as the reference.
reference_capacity = df["capacity"].iloc[0]
df["soh"] = (df["capacity"] / reference_capacity) * 100

df.to_csv("nasa_battery_data.csv", index=False)

print("NASA data extraction completed.")
print("Discharge records:", len(df))
print("Reference capacity (Ah):", round(reference_capacity, 4))
print(df.head())
print("Saved as nasa_battery_data.csv")