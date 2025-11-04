import pandas as pd

# Define cutoff date
cutoff_date = pd.Timestamp("2025-08-08")

# Files to clean
files = ["SoilSensorLog.csv"]

for file in files:
    # Load
    df = pd.read_csv(file)

    # Try to find timestamp column
    timestamp_col = None
    for col in df.columns:
        if "time" in col.lower():
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError(f"No timestamp column found in {file}")

    # Convert and filter
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df_clean = df[df[timestamp_col] < cutoff_date].copy()

    # Save cleaned file
    clean_name = file.replace(".csv", "_clean.csv")
    df_clean.to_csv(clean_name, index=False)
    print(f"✅ Cleaned {file}: {len(df)} → {len(df_clean)} rows saved as {clean_name}")
