import pandas as pd

def replicate_inputdata(input_path="inputdata.csv", output_path="inputdata_expanded.csv"):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Detect timestamp or date column
    time_col = None
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError("No timestamp/date column found in inputdata.csv")

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    bases = ["humidity", "ph", "ec", "n", "p", "k"]
    groups = ["soil", "soil2", "soil3"]

    # Find available base columns (without soil suffix)
    base_only = [b for b in bases if b in df.columns]
    if base_only:
        print("Detected base columns without soil suffix, replicating to all soil groups.")
        for g in groups:
            for b in base_only:
                df[f"{g}_{b}"] = df[b]

    # Detect any soil_* columns already there
    existing_groups = []
    for g in groups:
        if any(f"{g}_" in c for c in df.columns):
            existing_groups.append(g)

    if not existing_groups:
        raise ValueError("No recognizable soil columns found in inputdata.csv")

    # Replicate from the first existing group into the missing ones
    g_src = existing_groups[0]
    for g in groups:
        if g not in existing_groups:
            print(f"Replicating data from {g_src} to missing group {g}")
            for b in bases:
                src_col = f"{g_src}_{b}"
                if src_col in df.columns:
                    df[f"{g}_{b}"] = df[src_col]

    # Keep only relevant columns in order
    ordered_cols = [time_col] + [f"{g}_{b}" for g in groups for b in bases if f"{g}_{b}" in df.columns]
    df = df[ordered_cols]

    # Save correctly named expanded file
    df.to_csv(output_path, index=False)
    print(f"Saved expanded inputdata with correct naming: {output_path}")

    return df


if __name__ == "__main__":
    replicate_inputdata("inputdata.csv", "inputdata_expanded.csv")
