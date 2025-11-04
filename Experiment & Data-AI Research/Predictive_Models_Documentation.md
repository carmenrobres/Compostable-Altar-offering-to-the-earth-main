
# 🌱 Predictive Models — Compostable Altar Project

This directory contains the full predictive modeling system for "Compostable Altar: Offering to the Earth", developed under the Hungry EcoCities initiative in collaboration with Le Terre di Zoé and Betiana Pavón.It brings together AI, soil science, and biomaterial research to forecast how the soil evolves in response to biomaterial interventions.

## Repository Structure

Experiment & Data-AI Research/
│
├── Experiment - Barcelona/
│   └── SoilSensorLog.csv                    # Original sensor dataset
│
├── Predictive Models/
│   ├── forecast_app/
│   │   ├── static/                          # Plotly.js, CSS, JavaScript
│   │   ├── templates/                       # index.html for web interface
│   │   └── app.py                           # Flask web app backend
│   │
│   ├── lstm_models/                         # Trained LSTM models (.keras) and scalers
│   ├── lstm_predictions/                    # LSTM forecast outputs (CSV + PNG)
│   ├── SARIMA_models/                       # SARIMA models and metadata.json
│   ├── SARIMA_predictions/                  # SARIMA forecast outputs (CSV + PNG)
│   ├── models/                              # Legacy model backups
│   │
│   ├── clean_data.py                        # Data cleaning and preprocessing
│   ├── inputdata.csv                        # 1st-week soil input data for forecasts
│   ├── inputdata_expanded.csv               # Auto-expanded version of inputdata.csv
│   │
│   ├── LSTM_train.py                        # Trains LSTM models
│   ├── LSTM_forecast.py                     # Forecasts using trained LSTM models
│   ├── SARIMA_train.py                      # Trains SARIMA models
│   ├── SARIMA_forecast.py                   # Forecasts using SARIMA models
│   ├── replicate_inputdata.py               # Expands base input week for forecasting
│   │
│   ├── soil2_training_history.png           # LSTM training curve (soil2 model)
│   ├── soil3_training_history.png           # LSTM training curve (soil3 model)
│   │
│   ├── SoilSensorLog_clean.csv              # Cleaned dataset used for training
│   └── SoilSensorLog.csv                    # Raw soil data from experiment
│
├── images/
│   └── Correlation.png                      # Correlation visualization
│
├── Experiment_Analysis.md                   # Barcelona soil experiment report
├── Predictive_Models_Analysis.md            # Model performance and comparison
└── Predictive_Models_Documentation.md       # Predictive system study and description


## Setup Instructions

### Setup Instructions
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

### Install dependencies
Create a requirements.txt with the following content:
``` 
flask
pandas
numpy
matplotlib
scikit-learn
tensorflow
statsmodels
joblib
plotly
```
Then run

```
pip install -r requirements.txt
```

---

## System Overview

The predictive model suite provides two approaches for soil forecasting:

| Model Type | Purpose |
|------------|---------|
| **SARIMA** | Statistical - Captures repetitive irrigation and day–night cycles |
| **LSTM** | Neural Network - Learns nonlinear and long-term nutrient and moisture trends |

Both models predict **humidity, pH, electrical conductivity (EC), and nutrients (N, P, K)** from sensor data.

---

##  How the System Works

###  1. Data Collection

- **Sensor used:** CWT NPK-PH-CTH-S probe
- **Measurements:** moisture, EC, pH, N, P, K
- **Interval:** every 6 hours (03:00, 09:00, 15:00, 21:00)

Collected data is stored in:
```
SoilSensorLog.csv
```
The file is cleaned with:
```
python clean_data.py
```

to generate:
```
SoilSensorLog_clean.csv
```

### 2.Model Training

**LSTM_train.py**

Builds and trains LSTM neural networks for soil2 and soil3 (different biomaterials).
Uses 3 stacked LSTM layers with dropout regularization.
Automatically scales and sequences the data.
Saves trained models and scalers in /lstm_models/.
Run:
```
python LSTM_train.py
```

**Outputs:**
```
lstm_models/
├── soil2_lstm.keras
├── soil3_lstm.keras
├── validation_errors.json
├── soil2_targets_scaler.npy
└── soil3_targets_scaler.npy

soil2_training_history.png
soil3_training_history.png
```

**SARIMA_train.py**

Fits SARIMA models to capture cyclical irrigation patterns.
One model per variable (humidity, pH, EC, N, P, K) and per soil type.
Saves models and scalers in /SARIMA_models/.

Run:
```
python SARIMA_train.py
```

**Outputs:**
```
SARIMA_models/
├── soil2_humidity_sarima.pkl
├── soil3_ec_sarima.pkl
└── metadata.json
```

### 3. Forecast Generation
**LSTM_forecast.py**

Reads the user's 1-week input file (inputdata.csv)
Predicts 7, 14, and 58 days of soil behavior.
Adds cyclic time features (day_sin, irrig_cos, etc.)
Saves forecasts and graphs.
Run:
```
python LSTM_forecast.py
```

**Outputs:**
```
lstm_predictions/
├── predictions_7_14_58_scenarios.csv
└── forecast_with_input.png
```

**SARIMA_forecast.py**

Loads SARIMA models and forecasts future soil behavior.
Generates an overlay plot comparing observed vs forecasted data.

Run:
```
bashpython SARIMA_forecast.py
```

**Outputs:**
```
SARIMA_predictions/
├── sarima_predictions_7_14_58.csv
└── sarima_forecast_overlay.png
```
### 4. Web Interface (forecast_app/)
app.py
Runs a local Flask web app that lets users forecast data without coding.
Features:

Upload input CSV
Choose model: LSTM or SARIMA
Choose forecast horizon: 7, 14, or 58 days
Get instant Plotly visualizations in the browser

Start the app:
```
cd forecast_app
python app.py
```

**Visit:**
```
http://127.0.0.1:5000
```

**script.js**
Handles frontend behavior:

Uploads files to the /upload endpoint
Calls /predict for forecasts
Uses Plotly.js to draw 6 dynamic graphs:

Humidity
EC
Nitrogen
pH
Phosphorus
Potassium

## Input Data Requirements
--------------------------

The uploaded CSV file must contain:

| Column | Description |
| --- | --- |
| `timestamp` | Date/time of measurement (6-hour intervals) |
| `soil_humidity` | Soil humidity (%) |
| `soil_ph` | Soil pH |
| `soil_ec` | Electrical conductivity (μS/cm) |
| `soil_n` | Nitrogen index |
| `soil_p` | Phosphorus index |
| `soil_k` | Potassium index |

**Optional columns:**

-   `air_temp`, `dewpoint`, `rain`, `sunshine_duration`


## Model OUtputs
-----------

| File | Description |
| --- | --- |
| `lstm_predictions/predictions_7_14_58_scenarios.csv` | LSTM forecast results |
| `lstm_predictions/forecast_with_input.png` | Input week + LSTM predictions |
| `SARIMA_predictions/sarima_predictions_7_14_58.csv` | SARIMA forecast results |
| `SARIMA_predictions/sarima_forecast_overlay.png` | SARIMA overlay plot |
| `SARIMA_models/metadata.json` | MAE & RMSE metrics summary |

* * * * *

## Python LIbraries Used
------------------------

| Category | Libraries |
| --- | --- |
| **Web Interface** | flask, jinja2 |
| **Data Processing** | pandas, numpy, scikit-learn |
| **Time Series Modeling** | tensorflow, keras, statsmodels |
| **Visualization** | matplotlib, plotly |
| **File Management** | joblib, os, json, pathlib |

* * * * *

## Workflow SUmmary
-------------------------

1.  **Clean your data**


```
   python clean_data.py
```

2. **Train both models**


```
   python LSTM_train.py
   python SARIMA_train.py
```

3. **Run forecasts**

```
   python LSTM_forecast.py
   python SARIMA_forecast.py
```

4. **Launch web interface**

```
   cd forecast_app
   python app.py
```

5.  **Use the interface**
    -   Upload `inputdata.csv`
    -   Choose model (LSTM/SARIMA)
    -   Choose prediction duration (7, 14, 58 days)
    -   View interactive graphs

* * * * *

## Model COmparison
-------------------

| Feature | SARIMA | LSTM |
| --- | --- | --- |
| Captures daily cycles | ✅ | ✅ |
| Handles nonlinear effects | ❌ | ✅ |
| Easy to interpret | ✅ | ⚠️ Less interpretable |
| Needs large dataset | ❌ | ✅ |
| **Best for** | Periodic patterns | Complex time dependencies |



## Conceptual Summary
---------------------

-   **SARIMA** captures repeating patterns in soil behavior (e.g., daily irrigation cycles).
-   **LSTM** learns nonlinear and long-term effects like nutrient release over time.
-   The **Flask app** allows local, offline predictions and visualization.

Together, these tools turn experimental soil data into interactive, data-driven insights about how biomaterials affect soil regeneration.

* * * * *


> *"Here, AI does not seek to control but to listen --- to translate the rhythms of the soil into possible futures."*