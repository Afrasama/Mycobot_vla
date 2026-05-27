import os
import glob
import pandas as pd
import numpy as np

# Find all CSV files in data/plots/
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_pattern = os.path.join(base_dir, "data", "plots", "*.csv")
csv_files = glob.glob(csv_pattern)

print(f"Found {len(csv_files)} historical run records.")

all_records = []
for file in csv_files:
    filename = os.path.basename(file)
    # Extract timestamp from filename e.g. distance_vs_attempt_20260519_123722.csv
    parts = filename.replace(".csv", "").split("_")
    timestamp = "_".join(parts[-2:])
    
    try:
        df = pd.read_csv(file)
        if df.empty:
            continue
        
        # Get the final attempt's distance
        final_attempt = df.iloc[-1]
        all_records.append({
            "timestamp": timestamp,
            "filename": filename,
            "total_attempts": len(df),
            "final_distance_m": final_attempt["final_distance_m"]
        })
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if not all_records:
    print("No records found.")
    exit()

df_summary = pd.DataFrame(all_records)

print("\n=== SYSTEM PERFORMANCE STATS ===")
total_runs = len(df_summary)
successful_runs = df_summary[df_summary["final_distance_m"] < 0.10]
success_rate = (len(successful_runs) / total_runs) * 100

print(f"Total simulated runs analyzed: {total_runs}")
print(f"Success Rate (placed within 0.10m): {success_rate:.1f}%")
print(f"Average final distance to goal: {df_summary['final_distance_m'].mean() * 1000:.2f} mm")
print(f"Median final distance to goal: {df_summary['final_distance_m'].median() * 1000:.2f} mm")
print(f"Best placement accuracy: {df_summary['final_distance_m'].min() * 1000:.2f} mm")
print(f"Worst placement accuracy: {df_summary['final_distance_m'].max() * 1000:.2f} mm")

print("\n=== ATTEMPT EFFICIENCY ===")
print(f"Average attempts to succeed/terminate: {df_summary['total_attempts'].mean():.2f}")
print(f"Max attempts recorded: {df_summary['total_attempts'].max()}")

print("\n=== RUN BY RUN SUMMARY ===")
print(df_summary.sort_values(by="timestamp").to_string(index=False))
