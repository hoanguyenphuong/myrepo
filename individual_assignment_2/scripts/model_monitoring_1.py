# Purpose: Run immediately after inference (before labels)
# Checks: Data drift (features) + Prediction drift (pd_default)

import os, glob
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import ks_2samp

# to call this script: 
# option 1: Specify both snapshot date and model name
#     python scripts/model_monitoring_1.py --snapshotdate "2024-10-01" --modelname "credit_model_2024_09_01.pkl"
#
# option 2: Specify only snapshot date (the latest model in model_bank/ will be used)
#     python scripts/model_monitoring_1.py --snapshotdate "2024-10-01"


# ========== Utility: PSI ==========
def psi(expected, actual, bins=10, epsilon=1e-6):
    quantiles = np.percentile(expected, np.linspace(0, 100, bins + 1))
    expected_perc, _ = np.histogram(expected, bins=quantiles)
    actual_perc, _ = np.histogram(actual, bins=quantiles)
    expected_perc = np.clip(expected_perc / len(expected), epsilon, 1)
    actual_perc = np.clip(actual_perc / len(actual), epsilon, 1)
    return np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))

# ========== Utility: Get latest model ==========
def get_latest_model(model_dir="model_bank/"):
    """Return the latest model filename (.pkl) from the model_bank directory."""
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith(".pkl")],
        reverse=True
    )
    if not model_files:
        raise FileNotFoundError("No model found in model_bank/")
    latest_model = os.path.join(model_dir, model_files[0])
    print(f"⚙️ Using latest model as baseline: {latest_model}")
    return latest_model
    
def main(snapshot_date, modelname=None):
    print(f"\n--- DATA MONITORING FOR {snapshot_date} ---")
    # --- Resolve baseline model ---
    if not modelname or modelname.strip() == "":
        baseline_model_path = get_latest_model()
        # Extract the full date part (e.g., "2024_09_01") from filename
        parts = baseline_model_path.split("_")
        date_part = "_".join(parts[-3:]).replace(".pkl", "")  # e.g., "2024_09_01"
        baseline_snapshot = date_part.replace("_", "-")       # → "2024-09-01"
    else:
        baseline_model_path = os.path.join("model_bank", modelname)
        parts = modelname.split("_")
        date_part = "_".join(parts[-3:]).replace(".pkl", "")  # e.g., "2024_09_01"
        baseline_snapshot = date_part.replace("_", "-")       # → "2024-09-01"
    
    print(f"📊 Comparing against baseline model trained on {baseline_snapshot}\n")
    gold_features_path = "datamart/gold/feature_store/"
    gold_predictions_path = "datamart/gold/model_predictions/"

    # --- Load latest features and predictions ---
    latest_features = glob.glob(f"{gold_features_path}/**{snapshot_date.replace('-', '_')}*.parquet", recursive=True)
    latest_preds = glob.glob(f"{gold_predictions_path}*/**{snapshot_date.replace('-', '_')}*.parquet", recursive=True)

    if not latest_features or not latest_preds:
        raise FileNotFoundError(f"Missing feature or prediction data for {snapshot_date}")

    features = pd.read_parquet(latest_features[0])
    preds = pd.read_parquet(latest_preds[0])

    # --- Load baseline (from baseline_snapshot derived from modelname or latest model) ---
    baseline_preds = glob.glob(f"{gold_predictions_path}*/**{baseline_snapshot.replace('-', '_')}*.parquet", recursive=True)
    baseline_features = glob.glob(f"{gold_features_path}/**{baseline_snapshot.replace('-', '_')}*.parquet", recursive=True)

    if not baseline_preds or not baseline_features:
        print(f"⚠️ Baseline data ({baseline_snapshot}) not found — skipping drift comparison.")
        return

    base_pred = pd.read_parquet(baseline_preds[0])
    base_feat = pd.read_parquet(baseline_features[0])

    # --- Merge and Align Columns ---
    common_features = [c for c in features.columns if c in base_feat.columns and c not in ["Customer_ID", "snapshot_date"]]

    # --- Compute Drift for all numeric features ---
    drift_records = []
    for col in common_features:  # limit to top 10 for efficiency
        if np.issubdtype(features[col].dtype, np.number):
            try:
                psi_val = psi(base_feat[col].dropna().values, features[col].dropna().values)
                ks_stat, ks_p = ks_2samp(base_feat[col].dropna(), features[col].dropna())
                drift_records.append((col, psi_val, ks_stat, ks_p))
            except Exception as e:
                print(f"⚠️ Skipped {col}: {e}")

    drift_df = pd.DataFrame(drift_records, columns=["feature", "psi", "ks_stat", "ks_pvalue"])
    
    # --- Add score drift (model output) ---
    psi_score = psi(base_pred["pd_default"].values, preds["pd_default"].values)
    ks_score, ks_p_score = ks_2samp(base_pred["pd_default"].values, preds["pd_default"].values)
    drift_df.loc[len(drift_df)] = ["model_score", psi_score, ks_score, ks_p_score]

    print("\nFeature & Prediction Drift Summary:")
    print(drift_df.sort_values("psi", ascending=False))

    # --- Save monitoring results ---
    out_dir = "datamart/gold/model_monitoring/model_monitoring_1/"
    os.makedirs(out_dir, exist_ok=True)
    filepath = f"{out_dir}/data_drift_{snapshot_date.replace('-', '_')}.parquet"
    drift_df.to_parquet(filepath, index=False)
    print(f"✅ Drift metrics saved to {filepath}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--modelname", required=False)
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)