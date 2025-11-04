import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    model_dir = "SARIMA_models"
    pred_dir = "SARIMA_predictions"
    os.makedirs(pred_dir, exist_ok=True)

    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("metadata.json not found in SARIMA_models. Please run SARIMA_train.py first.")

    with open(meta_path) as f:
        meta = json.load(f)

    variables = meta["variables"]
    soils = meta["soils"]

    # Load input baseline (week 1)
    df_in = pd.read_csv("inputdata.csv")
    df_in = df_in[df_in["timestamp"].str.contains(":03|:09|:15|:21", na=False)]
    df_in = df_in.reset_index(drop=True)
    df_in = add_cycles(df_in)

    total_days = 58
    total_steps = total_days * 4
    df_future = pd.DataFrame(np.arange(total_steps), columns=["t"])
    df_future = add_cycles(df_future)
    exog_future = df_future[["day_sin", "day_cos", "irrig_sin", "irrig_cos", "day_norm", "hour"]]
    exog_future = (exog_future - exog_future.min()) / (exog_future.max() - exog_future.min())

    results = {}

    for s in soils:
        for v in variables:
            col = f"{s}_{v}"
            model_path = os.path.join(model_dir, f"{col}_sarima.pkl")
            scaler_path = os.path.join(model_dir, f"{col}_scaler.pkl")

            if not os.path.exists(model_path):
                print(f"Skipping {col}: missing model.")
                continue
            if not os.path.exists(scaler_path):
                print(f"Skipping {col}: missing scaler.")
                continue

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            try:
                pred_scaled = model.forecast(steps=total_steps, exog=exog_future)
                pred_scaled = np.maximum(np.array(pred_scaled), 0)
                pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            except Exception as e:
                print(f"Forecast failed for {col}: {e}")
                pred = np.zeros(total_steps)

            results[col] = pred

    out_df = pd.DataFrame(results)
    # --- Add timestamp column for Flask plotting ---
    timestamps = pd.date_range(
        start=pd.Timestamp(df_in["timestamp"].iloc[-1]) + pd.Timedelta(hours=6),
        periods=len(out_df),
        freq="6H"
    )
    out_df.insert(0, "timestamp", timestamps)

    # Save final forecast file
    out_df.to_csv(os.path.join(pred_dir, "sarima_predictions_7_14_58.csv"), index=False)

    # --- Overlay plot ---
    labels = {"soil": "soil", "soil2": "soil2", "soil3": "soil3"}
    colors = {"soil": "black", "soil2": "orange", "soil3": "brown"}

    days_obs = np.arange(1, len(df_in) / 4 + 1, 0.25)
    days_pred = np.arange(len(df_in) / 4 + 1, len(df_in) / 4 + total_days + 0.25, 0.25)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, var in enumerate(["humidity", "ph", "ec", "n", "p", "k"]):
        ax = axes[i]
        for s in ["soil", "soil2", "soil3"]:
            col = f"{s}_{var}"
            if col not in out_df.columns:
                continue

            if col in df_in.columns:
                ax.plot(days_obs[:len(df_in)], df_in[col][:len(days_obs)],
                        color=colors[s], alpha=0.7, label=f"{labels[s]} (week 1)")
            ax.plot(days_pred, out_df[col][:len(days_pred)],
                    color=colors[s], linestyle="--", label=f"{labels[s]} forecast")

        ax.set_title(var.upper())
        ax.set_xlabel("Day")
        ax.set_ylabel(var)
        ax.axvline(7, color="blue", linestyle="--", alpha=0.3)
        ax.axvline(14, color="blue", linestyle="--", alpha=0.3)
        ax.axvline(58, color="blue", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(pred_dir, "sarima_forecast_overlay.png"), dpi=300)
    plt.close()

    # --- Print validation error ranges ---
    print("\nValidation Error Ranges (MAE):")
    for var in ["humidity", "ph", "ec", "n", "p", "k"]:
        maes = []
        for s in ["soil", "soil2", "soil3"]:
            col = f"{s}_{var}"
            if col in meta["metrics"]:
                maes.append(meta["metrics"][col]["mae"])
        if maes:
            print(f"  {var}: {min(maes):.3f} .. {max(maes):.3f}")

    print("\nValidation Error Ranges (RMSE):")
    for var in ["humidity", "ph", "ec", "n", "p", "k"]:
        rmses = []
        for s in ["soil", "soil2", "soil3"]:
            col = f"{s}_{var}"
            if col in meta["metrics"]:
                rmses.append(meta["metrics"][col]["rmse"])
        if rmses:
            print(f"  {var}: {min(rmses):.3f} .. {max(rmses):.3f}")

    print("\nSaved forecast CSV and overlay plot in SARIMA_predictions")

if __name__ == "__main__":
    main()
