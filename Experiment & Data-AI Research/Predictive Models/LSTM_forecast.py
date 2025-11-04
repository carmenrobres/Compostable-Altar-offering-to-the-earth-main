#!/usr/bin/env python3
import argparse, os, json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

TARGETS = ["humidity","ph","ec","n","p","k"]
GROUPS = ["soil","soil2","soil3"]  # soil=no altar, soil2=clementine, soil3=wood+clementine
COVARS = ["day_sin","day_cos","irrig_sin","irrig_cos","day_norm","hour"]

def find_datetime(df):
    # try best column
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", utc=False)
        if s.notna().sum() > len(df)*0.5:
            return s
    raise ValueError("No valid datetime column found")

def ensure_four(df):
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    return df[df["hour"].isin([3,9,15,21])].copy()

def add_covars(df, start_ts=None):
    if start_ts is None:
        start_ts = df["timestamp"].min()
    df = df.copy()
    df["day_index"] = (df["timestamp"].dt.floor("D") - pd.Timestamp(start_ts).floor("D")).dt.days.astype(int)
    df["hour"] = df["timestamp"].dt.hour
    df["day_phase"] = 2*np.pi*(df["hour"]/24.0)
    df["day_sin"] = np.sin(df["day_phase"])
    df["day_cos"] = np.cos(df["day_phase"])
    hours_since = (df["timestamp"] - start_ts) / pd.Timedelta(hours=1)
    shifted = (hours_since - 13.0) % 72.0  # irrigation every 72h around 13:00
    df["irrig_phase"] = 2*np.pi*(shifted/72.0)
    df["irrig_sin"] = np.sin(df["irrig_phase"])
    df["irrig_cos"] = np.cos(df["irrig_phase"])
    max_day = max(df["day_index"].max(), 1)
    df["day_norm"] = df["day_index"] / max_day
    return df

def build_time_grid(start_ts, total_days=58):
    tlist = []
    ts = start_ts.floor("D")
    end = start_ts.floor("D") + pd.Timedelta(days=total_days)
    while ts < end:
        for h in [3,9,15,21]:
            t = ts + pd.Timedelta(hours=h)
            if t >= start_ts:
                tlist.append(t)
        ts += pd.Timedelta(days=1)
    return pd.to_datetime(sorted(set(tlist)))

def load_scaler_dict(path):
    d = np.load(path, allow_pickle=True).item()
    sc = MinMaxScaler()
    sc.min_ = d["min"]; sc.scale_ = d["scale"]
    sc.data_min_ = d["data_min"]; sc.data_max_ = d["data_max"]
    sc.data_range_ = sc.data_max_ - sc.data_min_
    return sc

def pick_base_week(df):
    """
    Return a frame with timestamp and the 6 base variables for the given week.
    Accepts either plain names (humidity, ph, ...) or any soil_* variant.
    Priority order for each variable:
      1) plain name
      2) soil_* or soil2_* or soil3_* found first
    """
    base = {"timestamp": df["timestamp"]}
    cols = {c.lower(): c for c in df.columns}
    for b in TARGETS:
        # plain first
        if b in cols:
            base[b] = df[cols[b]]
            continue
        # any group column
        cand = [k for k in cols if k.endswith("_"+b) or k.startswith(b+"_")]
        if cand:
            base[b] = df[cols[cand[0]]]
        else:
            base[b] = np.nan
    base = pd.DataFrame(base)
    return base

def smooth_direction(prev, pred):
    # keep direction, damp jumps
    return np.maximum(prev + 0.7*(pred - prev), 0.0)

def recursive_forecast(cov_grid, seed_targets, model, scaler, seq_len):
    """
    cov_grid: DataFrame with timestamp + covars for full horizon
    seed_targets: DataFrame with timestamp + 6 target cols for the first week
    """
    df = cov_grid.copy()
    for b in TARGETS:
        df[b] = np.nan

    # align seed into df
    seed = seed_targets[["timestamp"] + TARGETS].copy()
    df = pd.merge(df, seed, on="timestamp", how="left", suffixes=("", "_seed"))
    for b in TARGETS:
        df[b] = df[f"{b}_seed"]
        df.drop(columns=[f"{b}_seed"], inplace=True)

    # lists
    hist = []
    preds6 = []
    last_vals = None
    exp_feats = len(COVARS) + len(TARGETS)

    for i in range(len(df)):
        row = df.iloc[i]
        covar_vec = row[COVARS].values.astype(np.float32)
        y_true = row[TARGETS].values.astype(np.float32)
        if not np.any(np.isnan(y_true)):
            y_scaled = scaler.transform(y_true.reshape(1,-1))[0]
            last_vals = y_true
        else:
            if last_vals is None:
                # until we have something, assume zeros
                y_scaled = np.zeros(len(TARGETS), dtype=np.float32)
            else:
                y_scaled = scaler.transform(last_vals.reshape(1,-1))[0]

        step = np.concatenate([covar_vec, y_scaled]).astype(np.float32)
        # fix length
        if step.shape[0] < exp_feats:
            step = np.pad(step, (0, exp_feats - step.shape[0]), constant_values=0.0)
        elif step.shape[0] > exp_feats:
            step = step[:exp_feats]
        hist.append(step)
        if len(hist) > seq_len:
            hist = hist[-seq_len:]

        if len(hist) == seq_len:
            X = np.stack(hist, axis=0).reshape(1, seq_len, exp_feats)
            y_pred_scaled = model.predict(X, verbose=0)[0]
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(1,-1))[0]
            if last_vals is not None:
                y_pred = smooth_direction(last_vals, y_pred)
            last_vals = y_pred
            preds6.append(y_pred)
        else:
            # before we reach seq_len, keep the seed or zeros
            if last_vals is None:
                preds6.append(np.zeros(len(TARGETS)))
            else:
                preds6.append(last_vals)

    preds6 = np.vstack(preds6)
    out = df[["timestamp"]].copy()
    for j, b in enumerate(TARGETS):
        out[b] = np.maximum(preds6[:, j], 0.0)
    return out

def main():
    print("=== Auto Forecast Run ===")

    # fixed paths
    raw_input = "inputdata.csv"
    expanded_input = "inputdata_expanded.csv"
    models_dir = "lstm_models"
    out_dir = "lstm_predictions"
    total_days = 58

    os.makedirs(out_dir, exist_ok=True)

    # 1. Expand the input file automatically
    print("Expanding input data using replicate_inputdata.py ...")
    os.system(f"python replicate_inputdata.py {raw_input} {expanded_input}")

    # 2. Load metadata for seq_len
    meta_path = os.path.join(models_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print("metadata.json not found in lstm_models", file=sys.stderr)
        sys.exit(1)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    seq_len = int(meta["seq_len"])

    # 3. Read the expanded input
    df = pd.read_csv(expanded_input)
    df["timestamp"] = find_datetime(df)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = ensure_four(df)
    if len(df) < 20:
        print("Warning: less than one week of data. Forecast may be poor.", file=sys.stderr)

    start_ts = df["timestamp"].min()
    grid = pd.DataFrame({"timestamp": build_time_grid(start_ts, total_days)})
    grid = add_covars(grid, start_ts=start_ts)

    # 4. Base week (input) values
    base_week = pick_base_week(df)[["timestamp"] + TARGETS]
    base_week[TARGETS] = base_week[TARGETS].apply(pd.to_numeric, errors="coerce").ffill()

    # 5. Predict each scenario
    out = {"timestamp": grid["timestamp"].values}
    for g in GROUPS:
        model_path = os.path.join(models_dir, f"{g}_lstm.keras")
        scaler_path = os.path.join(models_dir, f"{g}_targets_scaler.npy")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print(f"Skip {g}: missing model or scaler.")
            continue
        model = keras.models.load_model(model_path)
        scaler = load_scaler_dict(scaler_path)
        pred_g = recursive_forecast(grid, base_week, model, scaler, seq_len)
        for b in TARGETS:
            out[f"{b}_{g}"] = pred_g[b].values

    out_df = pd.DataFrame(out)
    for b in TARGETS:
        for g in GROUPS:
            col = f"{b}_{g}"
            if col in out_df.columns:
                out_df[col] = np.maximum(out_df[col].astype(float), 0.0)

    csv_path = os.path.join(out_dir, "predictions_7_14_58_scenarios.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # 6. Plot input week + predictions using day numbers
    fig, axes = plt.subplots(3, 2, figsize=(14,10), sharex=True)
    axes = axes.ravel()
    days = (out_df["timestamp"] - start_ts).dt.days + 1

    for i, b in enumerate(TARGETS):
        ax = axes[i]
        # input week
        if b in df.columns:
            ax.plot((df["timestamp"] - start_ts).dt.days + 1, df[b], "k.", label="input week")
        # predictions
        for g in GROUPS:
            col = f"{b}_{g}"
            if col in out_df.columns:
                ax.plot(days, out_df[col], label=g)
        ax.set_title(b)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.axvline(7, linestyle="--", linewidth=1)
        ax.axvline(14, linestyle="--", linewidth=1)
        ax.set_xlim(0, total_days + 1)
        ax.set_xlabel("Day")

    plt.tight_layout()
    img_path = os.path.join(out_dir, "forecast_with_input.png")
    plt.savefig(img_path, dpi=150)
    print(f"Saved plot: {img_path}")

    # 7. Print validation MAE summary
    ve_path = os.path.join(models_dir, "validation_errors.json")
    if os.path.exists(ve_path):
        with open(ve_path, "r") as f:
            ve = json.load(f)
        print("\nValidation MAE ranges:")
        for b in TARGETS:
            vals = [ve[g][b] for g in GROUPS if g in ve and b in ve[g]]
            if vals:
                print(f"  {b}: {min(vals):.3f} .. {max(vals):.3f}")
    else:
        print("validation_errors.json not found")

if __name__ == "__main__":
    main()