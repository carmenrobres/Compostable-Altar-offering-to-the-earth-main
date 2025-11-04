import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

def add_cycles(df):
    df["day"] = np.arange(len(df)) // 4 + 1
    df["hour"] = (np.arange(len(df)) % 4) * 6 + 3
    df["day_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["day_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["irrig_sin"] = np.sin(2 * np.pi * df["day"] / 3)
    df["irrig_cos"] = np.cos(2 * np.pi * df["day"] / 3)
    df["day_norm"] = df["day"] / df["day"].max()
    return df

def main():
    data_path = "SoilSensorLog_clean.csv"
    model_dir = "SARIMA_models"
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    # convert rain to numeric and filter
    df["daily_rain_sum"] = pd.to_numeric(df["daily_rain_sum"], errors="coerce")
    df = df[df["daily_rain_sum"] <= 10]
    df = df.dropna().reset_index(drop=True)

    df = add_cycles(df)
    exog_vars = ["day_sin", "day_cos", "irrig_sin", "irrig_cos", "day_norm", "hour"]
    exog = df[exog_vars]
    exog = (exog - exog.min()) / (exog.max() - exog.min())

    soils = ["soil", "soil2", "soil3"]
    variables = ["humidity", "ph", "ec", "n", "p", "k"]

    metrics = {}

    for s in soils:
        for v in variables:
            col = f"{s}_{v}"
            if col not in df.columns:
                continue

            series = df[col].astype(float).fillna(0)
            if series.std() == 0:
                print(f"Skipping {col}: no variance")
                continue

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

            split_idx = int(0.8 * len(scaled))
            train, test = scaled[:split_idx], scaled[split_idx:]
            exog_train, exog_test = exog.iloc[:split_idx], exog.iloc[split_idx:]

            try:
                model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4),
                                exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                forecast = res.predict(start=split_idx, end=len(scaled) - 1, exog=exog_test)
                mae = mean_absolute_error(test, forecast)
                rmse = np.sqrt(mean_squared_error(test, forecast))
                metrics[col] = {"mae": float(mae), "rmse": float(rmse)}

                print(f"{col}: MAE={mae:.3f}, RMSE={rmse:.3f}")

                joblib.dump(res, os.path.join(model_dir, f"{col}_sarima.pkl"))
                joblib.dump(scaler, os.path.join(model_dir, f"{col}_scaler.pkl"))
            except Exception as e:
                print(f"Error training {col}: {e}")
                continue

    metadata = {
        "variables": variables,
        "soils": soils,
        "metrics": metrics
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved models and error summary in SARIMA_models/")

if __name__ == "__main__":
    main()
