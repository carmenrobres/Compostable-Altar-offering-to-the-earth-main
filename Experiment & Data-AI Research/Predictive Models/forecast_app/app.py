import os
import json
import subprocess
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

# Paths relative to this file
APP_DIR = Path(__file__).resolve().parent          # ...\FINAL\forecast_app
FINAL_DIR = APP_DIR.parent                         # ...\FINAL

# Where your scripts live
LSTM_SCRIPT = FINAL_DIR / "LSTM_forecast.py"
SARIMA_SCRIPT = FINAL_DIR / "SARIMA_forecast.py"

# Outputs created by your scripts
LSTM_PRED_PATH = FINAL_DIR / "lstm_predictions" / "predictions_7_14_58_scenarios.csv"
SARIMA_PRED_PATH = FINAL_DIR / "SARIMA_predictions" / "sarima_predictions_7_14_58.csv"

# Optional error file
ERRORS_JSON = FINAL_DIR / "lstm_models" / "validation_errors.json"

# Variables we will try to plot
VARS = ["humidity", "ec", "n", "ph", "p", "k"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No selected file"}), 400

    save_path = FINAL_DIR / "inputdata.csv"
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    f.save(save_path)

    print("DEBUG saved input to:", save_path, "exists?", save_path.exists())
    # --- Clear old prediction files when a new input CSV is uploaded ---
    import shutil

    for pred_dir in ["lstm_predictions", "SARIMA_predictions"]:
        folder = FINAL_DIR / pred_dir
        if folder.exists():
            for item in folder.iterdir():
                try:
                    if item.is_file():
                        item.unlink()  # delete file
                    elif item.is_dir():
                        shutil.rmtree(item)  # delete subfolder if any
                except Exception as e:
                    print(f"Warning: could not remove {item}: {e}")

    try:
        df = pd.read_csv(save_path)
        cols = list(df.columns)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    return jsonify({"ok": True, "columns": cols})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    print("DEBUG incoming data:", data)
    model = data.get("model")
    horizon = str(data.get("horizon"))

    if model not in ["LSTM", "SARIMA"]:
        return jsonify({"error": f"Bad request: {data}"}), 400
    if horizon not in ["7", "14", "58"]:
        return jsonify({"error": "horizon must be 7, 14, or 58"}), 400

    # Choose script and prediction file
    script_path = LSTM_SCRIPT if model == "LSTM" else SARIMA_SCRIPT
    pred_path = LSTM_PRED_PATH if model == "LSTM" else SARIMA_PRED_PATH

    # Always rerun the forecast script
    print(f"Running {script_path.name} ...")
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(FINAL_DIR),  # run inside FINAL
            check=True,
            capture_output=True,
            text=True,
        )


        # Only keep actual error messages
        stderr_clean = "\n".join(
            line for line in result.stderr.splitlines()
            if not line.strip().startswith("2025") and "oneDNN" not in line
        )

        if result.returncode != 0 and stderr_clean.strip():
            return jsonify({"error": f"Script failed: {stderr_clean}"}), 500
    except Exception as e:
      print(f"Error: {e}")
    # Load input and prediction data
    input_csv = FINAL_DIR / "inputdata.csv"
    if not input_csv.exists():
        return jsonify({"error": "Upload inputdata.csv first"}), 400

    input_df = pd.read_csv(input_csv)
    # Load prediction CSV
    # Read forecast and input to merge for plotting
    pred_df = pd.read_csv(pred_path)

    # Ensure both have the same timestamp column name
    time_col = "timestamp"
    if time_col not in input_df.columns:
        time_col = input_df.columns[0]

    # Convert timestamps
    input_df[time_col] = pd.to_datetime(input_df[time_col])
    pred_df[time_col] = pd.to_datetime(pred_df[time_col])

    # Merge on timestamp
    merged_df = pd.concat([input_df, pred_df], ignore_index=True, sort=False)
    merged_df = merged_df.fillna(0)  # avoid NaN for plotting

    # Detect timestamp column automatically
    possible_time_cols = ["timestamp", "date", "time", "datetime"]
    time_col = next((c for c in pred_df.columns if c.lower() in possible_time_cols), None)

    if time_col is None:
        # Get last timestamp from inputdata.csv
        input_df = pd.read_csv(input_csv)
        input_df_time_col = next((c for c in input_df.columns if "time" in c.lower() or "date" in c.lower()), None)
        if input_df_time_col:
            last_time = pd.to_datetime(input_df[input_df_time_col]).max()
            # Create forecast timestamps starting right after last_time
            pred_df["timestamp"] = pd.date_range(
                start=last_time + pd.Timedelta(days=1),
                periods=len(pred_df),
                freq="D"
            )
        else:
            pred_df["timestamp"] = pd.date_range(start=pd.Timestamp.today(), periods=len(pred_df), freq="D")
        time_col = "timestamp"
    else:
        pred_df[time_col] = pd.to_datetime(pred_df[time_col], errors="coerce")

    # Sort everything by timestamp just in case
    pred_df = pred_df.sort_values(by=time_col)


    # Make sure both have consistent datetime columns
    time_col = None
    for c in input_df.columns:
        if 'date' in c.lower() or 'time' in c.lower():
            time_col = c
            break

    if time_col:
        input_df[time_col] = pd.to_datetime(input_df[time_col])
        # Detect timestamp column automatically
        possible_time_cols = ["timestamp", "date", "time", "datetime"]
        time_col = next((c for c in pred_df.columns if c.lower() in possible_time_cols), None)

        if time_col is None:
            # Create a synthetic time column if missing
            pred_df.insert(0, "timestamp", range(len(pred_df)))
            time_col = "timestamp"
        else:
            pred_df[time_col] = pd.to_datetime(pred_df[time_col], errors="coerce")


        # Find the last timestamp from input data
        last_time = input_df[time_col].iloc[-1]

        # Compute expected time step (difference between last two)
        if len(input_df) > 1:
            step = input_df[time_col].iloc[-1] - input_df[time_col].iloc[-2]
        else:
            step = pd.Timedelta(days=1)

        # Shift prediction timestamps to continue right after input
        pred_df[time_col] = [last_time + step * (i + 1) for i in range(len(pred_df))]

    # Include the input week when plotting (7 + forecast days)
    steps_per_day = 4
    adjusted_days = 7 + int(horizon)
    limit = adjusted_days * steps_per_day
    pred_df = pred_df.head(limit)

    # Combine input + forecast for continuous plotting
    combined_df = pd.concat([input_df, pred_df], ignore_index=True)

    # Identify the x-axis column (time)
    def pick_x(df):
        for c in df.columns:
            if c.lower() in ["date", "timestamp", "time", "step", "index", "day"]:
                return c
        return df.columns[0]

    x_col = pick_x(combined_df)

    # Prepare all variable columns (humidity_soil, humidity_soil2, etc.)
    combined_series = {}
    for col in combined_df.columns:
        lc = col.lower()
        # skip non-numeric / time columns
        if any(word in lc for word in ["date", "time", "timestamp", "day", "index", "step"]):
            continue
        if any(v in lc for v in VARS):
            try:
                combined_series[col] = pd.to_numeric(combined_df[col], errors="coerce").fillna(0).tolist()
            except Exception:
                continue

    print(f"DEBUG sending {len(combined_df)} rows, {len(combined_series)} series to frontend.")

    # --- Load error metrics if available ---
    error_info = {}
    if model.upper() == "LSTM":
        err_file = FINAL_DIR / "lstm_predictions" / "validation_errors.json"
    elif model.upper() == "SARIMA":
        err_file = FINAL_DIR / "SARIMA_models" / "sarima_errors.json"
    else:
        err_file = None

    if err_file and err_file.exists():
        try:
            with open(err_file, "r") as f:
                error_info = json.load(f)
        except Exception:
            error_info = {}

    return jsonify({
        "model": model,
        "horizon": horizon,
        "x": combined_df[x_col].astype(str).tolist(),
        "series": combined_series,
        "error": error_info,
        "info": f"{model} forecast for {adjusted_days} total days (input + forecast)"
    })



if __name__ == "__main__":
    app.run(debug=True)

