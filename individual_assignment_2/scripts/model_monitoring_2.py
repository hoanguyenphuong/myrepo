import os, glob
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import ks_2samp

def psi(expected, actual, bins=10, epsilon=1e-6):
    quantiles = np.percentile(expected, np.linspace(0, 100, bins + 1))
    expected_perc, _ = np.histogram(expected, bins=quantiles)
    actual_perc, _ = np.histogram(actual, bins=quantiles)
    expected_perc = np.clip(expected_perc / len(expected), epsilon, 1)
    actual_perc = np.clip(actual_perc / len(actual), epsilon, 1)
    return np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))

def main(snapshot_date):
    print(f"\n--- MODEL PERFORMANCE MONITORING FOR {snapshot_date} ---")
    gold_path = "datamart/gold/model_predictions/"
    label_path = "datamart/gold/label_store/"

    # --- Load predictions ---
    latest_preds = glob.glob(f"{gold_path}*/**{snapshot_date.replace('-', '_')}*.parquet", recursive=True)
    if not latest_preds:
        raise FileNotFoundError(f"No predictions found for {snapshot_date}")
    preds = pd.read_parquet(latest_preds[0])

    # --- Load labels ---
    labels = pd.read_parquet(f"{label_path}/gold_label_store_{snapshot_date.replace('-', '_')}.parquet")

    df = preds.merge(labels, on=["Customer_ID", "snapshot_date"], how="inner")
    y_true, y_pred = df["label"], df["pd_default"]

    # --- Compute metrics ---
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
    ks_stat, ks_p = ks_2samp(y_pred[y_true == 0], y_pred[y_true == 1])

    # --- PSI (score drift vs baseline) ---
    baseline_snapshot = "2024-09-01"
    baseline_preds = glob.glob(f"{gold_path}*/**{baseline_snapshot.replace('-', '_')}*.parquet", recursive=True)
    if baseline_preds:
        expected = pd.read_parquet(baseline_preds[0])["pd_default"].values
        psi_val = psi(expected, y_pred.values)
    else:
        psi_val = np.nan

    print(f"AUC={auc:.3f}, F1={f1:.3f}, KS={ks_stat:.3f}, PSI={psi_val:.3f}")

    # --- Save metrics ---
    metrics = pd.DataFrame({
        "snapshot_date": [snapshot_date],
        "auc": [auc],
        "f1": [f1],
        "ks_stat": [ks_stat],
        "ks_p": [ks_p],
        "psi": [psi_val],
        "timestamp": [datetime.now()]
    })

    out_dir = "datamart/gold/model_monitoring/model_monitoring_2/"
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = f"{out_dir}/model_monitoring_history.parquet"

    if os.path.exists(metrics_path):
        old = pd.read_parquet(metrics_path)
        metrics = pd.concat([old, metrics], ignore_index=True)

    metrics.to_parquet(metrics_path, index=False)
    print(f"✅ Metrics appended to {metrics_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()
    main(args.snapshotdate)