# 🔋 AI-Powered Battery Digital Twin

An interactive **Streamlit** dashboard that explores lithium-ion battery degradation using real NASA experimental data. It estimates relative **State of Health (SOH)** with a Random Forest model, forecasts future degradation with a linear trend, and presents everything through interactive Plotly charts.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

> ⚠️ **Disclaimer:** This is an exploratory, educational prototype built on a single battery cell. It is **not** a battery safety tool and must not be used for real-world safety decisions or remaining-useful-life guarantees.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Model Performance](#-model-performance)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

A *digital twin* is a virtual model that mirrors a physical system. This project builds a lightweight digital twin of a lithium-ion battery: it ingests measured discharge data, learns how health relates to operating conditions, and lets you query the model interactively.

The pipeline covers the full workflow from raw lab data to a user-facing app:

1. **Extract** per-cycle discharge features from the raw NASA `.mat` file
2. **Compute** relative SOH from discharge capacity
3. **Train** a Random Forest regressor to estimate SOH
4. **Evaluate** it with leave-one-out cross-validation
5. **Forecast** future SOH with a linear trend
6. **Visualize** everything in a Streamlit dashboard

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Dataset overview** | Record count, latest SOH, and live test MAE at a glance |
| 📉 **Degradation chart** | Interactive line chart of SOH across experimental cycles |
| 🧪 **SOH predictor** | Enter cycle, voltage, current and temperature to get an estimated SOH with an illustrative status label |
| 📈 **Actual vs. predicted** | Parity plot of test-set predictions against the ideal line |
| 🧠 **Model metrics** | MAE, RMSE and R² displayed in the app |
| 🔮 **Degradation forecast** | Historical SOH plus a 20-cycle linear-trend projection |
| 💡 **Automatic insights** | First, highest and lowest recorded SOH with change-since-start summary |
| 📁 **Data explorer** | Expandable tables for the dataset and forecast values |

---

## 🛠 Tech Stack

- **Language:** Python
- **Web app:** [Streamlit](https://streamlit.io/)
- **Machine learning:** [scikit-learn](https://scikit-learn.org/) (`RandomForestRegressor`, `LinearRegression`)
- **Data handling:** pandas, NumPy
- **Visualization:** [Plotly](https://plotly.com/python/) (Express and Graph Objects)
- **Data extraction:** SciPy (`scipy.io.loadmat`)

---

## 📂 Project Structure

```
battery-digital-twin/
├── app.py                    # Streamlit dashboard (model training, charts, predictor)
├── extract_battery_data.py   # Converts raw NASA B0025.mat → nasa_battery_data.csv
├── evaluate_model.py         # Leave-one-out cross-validation of the Random Forest
├── forecast_battery.py       # Linear-trend SOH forecast → battery_forecast.csv
├── nasa_battery_data.csv     # Processed NASA B0025 discharge records (used by the app)
├── battery_data.csv          # Small illustrative sample dataset (not used by the app)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

**Files generated or excluded from version control** (see `.gitignore`):

- `B0025.mat` is the raw NASA dataset, which you download yourself
- `battery_forecast.csv` is produced by `forecast_battery.py`

---

## 🗃 Dataset

- **Source:** NASA Prognostics Center of Excellence, Li-ion battery aging experiments
- **Cell:** `B0025`
- **Records used:** 28 discharge cycles (experimental cycle indices 4 → 78)

### Columns in `nasa_battery_data.csv`

| Column | Unit | Description |
|---|---|---|
| `cycle` | – | Experimental cycle index in the original dataset |
| `voltage` | V | Mean measured voltage during discharge |
| `current` | A | Mean measured current during discharge |
| `temperature` | °C | Mean measured temperature during discharge |
| `capacity` | Ah | Measured discharge capacity |
| `soh` | % | Relative State of Health (see below) |

### How SOH is defined

```
SOH (%) = (discharge capacity at cycle n / first recorded discharge capacity) × 100
```

SOH here is **relative to the first recorded discharge capacity** in this dataset, not the manufacturer-rated capacity. Because capacity can fluctuate slightly between cycles, SOH may occasionally exceed 100%.

### Dataset snapshot

| Metric | Value |
|---|---|
| Discharge records | 28 |
| Cycle range | 4 – 78 |
| First SOH | 100.00% |
| Latest SOH | ≈ 95.71% |
| Highest SOH | ≈ 100.11% |
| Lowest SOH | ≈ 95.71% |

---

## ⚙️ How It Works

### 1. Data extraction (`extract_battery_data.py`)
Loads `B0025.mat`, iterates through every cycle, keeps only `discharge` cycles that report a capacity, averages voltage, current and temperature over each cycle, and computes SOH relative to the first discharge capacity. Output: `nasa_battery_data.csv`.

### 2. SOH prediction model (`app.py`)
- **Algorithm:** `RandomForestRegressor` (100 trees, `random_state=42`)
- **Input features:** `cycle`, `voltage`, `current`, `temperature`
- **Target:** `soh`
- **Split:** 75% train / 25% test
- **Note:** `capacity` is deliberately **not** used as an input feature, since SOH is derived directly from it and including it would leak the answer.

Predictions are clipped to the 0–100% range, and an illustrative status label is shown:

| Predicted SOH | Status |
|---|---|
| ≥ 90% | 🟢 Excellent |
| 80% – 90% | 🔵 Good |
| 70% – 80% | 🟠 Degraded |
| < 70% | 🔴 Poor |

These thresholds are illustrative only and are not validated diagnostic criteria.

### 3. Evaluation (`evaluate_model.py`)
Because there are only 28 records, a single random split is noisy. This script runs **leave-one-out cross-validation** and reports MAE, RMSE and R² over all records for a more stable estimate.

### 4. Forecasting (`forecast_battery.py`)
Fits a `LinearRegression` to SOH versus cycle and projects the next **20 cycles** beyond the last recorded one, clipped to 0–100%. Output: `battery_forecast.csv`, which the dashboard plots alongside the history.

---

## 📈 Model Performance

Leave-one-out cross-validation on all 28 records (`python evaluate_model.py`):

| Metric | Value |
|---|---|
| MAE | 0.3682 percentage points |
| RMSE | 0.4789 percentage points |
| R² | 0.8367 |

> These figures come from one cell with a small sample. Neighbouring cycles are similar, so the results do **not** demonstrate generalization to other batteries. The dashboard's own metrics use a 75/25 random holdout, so they will differ slightly from the figures above.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or newer
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/aryanbhardwaj7542-web/battery-digital-twin.git
cd battery-digital-twin
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the forecast file

The forecast chart reads `battery_forecast.csv`, which is not committed to the repo:

```bash
python forecast_battery.py
```

### 5. Launch the app

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 💻 Usage

### Run the dashboard

```bash
streamlit run app.py
```

Use the **Predict Battery Health** section to enter a cycle number, voltage, current and temperature, then click **🔍 Predict Battery Health**. Defaults are pre-filled from the dataset, and you should keep inputs close to the training range for sensible results.

### Re-evaluate the model

```bash
python evaluate_model.py
```

### Regenerate the forecast

```bash
python forecast_battery.py
```

### Rebuild the dataset from raw NASA data (optional)

1. Download the NASA Li-ion battery aging dataset and place `B0025.mat` in the project root
2. Run:

```bash
python extract_battery_data.py
```

---

## ⚠️ Limitations

- **Single cell:** all data comes from battery `B0025`, so the model is not validated on other cells or chemistries
- **Tiny dataset:** only 28 discharge records, which makes the metrics sensitive and optimistic
- **Random split leakage:** neighbouring cycles are similar, so a random train/test split can overstate accuracy
- **Relative SOH:** measured against the first recorded capacity, not the rated capacity
- **Linear forecast:** real degradation is typically non-linear (including knee-point behaviour), so the projection is only a rough trend
- **Not a safety tool:** status labels are illustrative and cannot establish safety or remaining useful life

---

## 🔮 Future Improvements

- [ ] Train and validate on multiple NASA cells (B0005, B0006, B0007, B0018, …)
- [ ] Use cell-wise or time-aware validation instead of random splits
- [ ] Add non-linear and time-series forecasting models (e.g. LSTM, Gaussian process, exponential fit)
- [ ] Engineer richer features (discharge curve shape, internal resistance, incremental capacity)
- [ ] Add prediction uncertainty estimates
- [ ] Include unit tests and a CI workflow
- [ ] Deploy to Streamlit Community Cloud

---

## 👤 Author

**Aryan Bhardwaj**
2nd-year B.Tech student, Artificial Intelligence & Machine Learning, Dayananda Sagar University, Bengaluru

- GitHub: [@aryanbhardwaj7542-web](https://github.com/aryanbhardwaj7542-web)

---

## 🙏 Acknowledgements

- [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) for the publicly available battery aging dataset
- The open-source communities behind Streamlit, scikit-learn, pandas and Plotly

---

## 📄 License

No license has been specified yet. Add a `LICENSE` file (for example MIT) to the repository to define how others may use this project.
