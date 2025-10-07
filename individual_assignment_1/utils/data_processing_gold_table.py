import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# ======================================================
# PART A — LABEL STORE (from lab 2)
# ======================================================

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    partition_name = f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_loan_daily_directory, partition_name)
    df = spark.read.parquet(filepath)
    print(f"[Label Store] loaded from: {filepath}, row count: {df.count()}")

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table
    partition_name = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_label_store_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print("[Label Store] saved to:", filepath)
    return df


# ======================================================
# PART B — FEATURE STORE
# ======================================================

def process_features_gold_table(snapshot_date_str, silver_base_directory, gold_feature_store_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    tag = snapshot_date_str.replace('-', '_')

    # === Load 3 silver tables ===
    print("\n=== [Feature Store] Loading Silver tables ===")
    silver_click_path = os.path.join(silver_base_directory, f"feature_clickstream/silver_clickstream_{tag}.parquet")
    silver_attr_path = os.path.join(silver_base_directory, f"feature_attributes/silver_attributes_{tag}.parquet")
    silver_fin_path = os.path.join(silver_base_directory, f"feature_financials/silver_financials_{tag}.parquet")

    df_click = spark.read.parquet(silver_click_path)
    df_attr = spark.read.parquet(silver_attr_path)
    df_fin = spark.read.parquet(silver_fin_path)

    print("Loaded clickstream:", df_click.count(), "rows")
    print("Loaded attributes:", df_attr.count(), "rows")
    print("Loaded financials:", df_fin.count(), "rows")

    # === Aggregate clickstream (median up to snapshot_date) ===
    print("\n=== Aggregating clickstream (median up to snapshot_date) ===")
    df_click = df_click.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df_attr = df_attr.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df_fin = df_fin.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    click_cols = [c for c in df_click.columns if c.startswith("fe_")]

    # ✅ Fix ambiguous snapshot_date by renaming df_fin's snapshot_date
    df_fin_temp = df_fin.select("Customer_ID", col("snapshot_date").alias("snapshot_date_fin"))

    df_click_agg = (
        df_click.join(df_fin_temp, "Customer_ID", "inner")
        .filter(col("snapshot_date") <= col("snapshot_date_fin"))
        .groupBy("Customer_ID")
        .agg(*[
            F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}_median")
            for c in click_cols
        ])
    )

    # Add has_clickstream_data
    df_click_agg = df_click_agg.withColumn(
        "has_clickstream_data",
        F.when(F.lit(True), F.lit(1)).otherwise(F.lit(0))
    )

    # === Merge attribute + financials ===
    print("\n=== Merging attributes + financials ===")
    df_feat = df_fin.join(df_attr.drop("snapshot_date"), on="Customer_ID", how="inner")

    # === Join aggregated clickstream ===
    df_feat = df_feat.join(df_click_agg, on="Customer_ID", how="left")

    # === Fill missing clickstream data ===
    df_feat = df_feat.fillna(0, subset=[c for c in df_feat.columns if c.startswith("fe_")])
    df_feat = df_feat.fillna({"has_clickstream_data": 0})

    print("Merged feature store shape:", df_feat.count(), "rows ×", len(df_feat.columns), "cols")

    # === Remove Annual_Income (high VIF), scale Credit_Utilization_Ratio ===
    print("\n=== Feature Selection ===")
    if "Annual_Income" in df_feat.columns:
        df_feat = df_feat.drop("Annual_Income")

    if "Credit_Utilization_Ratio" in df_feat.columns:
        mean_ratio = df_feat.select(F.mean(col("Credit_Utilization_Ratio"))).collect()[0][0]
        std_ratio = df_feat.select(F.stddev(col("Credit_Utilization_Ratio"))).collect()[0][0]
        df_feat = df_feat.withColumn(
            "Credit_Utilization_Ratio_scaled",
            (col("Credit_Utilization_Ratio") - F.lit(mean_ratio)) / F.lit(std_ratio)
        ).drop("Credit_Utilization_Ratio")

    # === PCA on clickstream features (explaining ~90%) ===
    print("\n=== Applying PCA on clickstream features ===")
    from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA as SparkPCA

    click_cols_final = [c for c in df_feat.columns if c.startswith("fe_")]

    if click_cols_final:
        assembler = VectorAssembler(inputCols=click_cols_final, outputCol="clickstream_vec")
        df_vec = assembler.transform(df_feat).select("Customer_ID", "clickstream_vec")

        scaler = StandardScaler(inputCol="clickstream_vec", outputCol="clickstream_scaled", withMean=True, withStd=True)
        scaler_model = scaler.fit(df_vec)
        df_scaled = scaler_model.transform(df_vec)

        # Run PCA (select enough components for ~90%)
        pca_model = SparkPCA(k=len(click_cols_final), inputCol="clickstream_scaled", outputCol="clickstream_pca").fit(df_scaled)
        var_ratio = pca_model.explainedVariance.toArray().cumsum()
        k_opt = int(np.argmax(var_ratio >= 0.9) + 1)
        print(f"PCA components retained: {k_opt} (explains {var_ratio[k_opt - 1] * 100:.2f}%)")

        pca_final = SparkPCA(k=k_opt, inputCol="clickstream_scaled", outputCol="clickstream_pca_final").fit(df_scaled)
        df_pca = pca_final.transform(df_scaled).select("Customer_ID", "clickstream_pca_final")

        df_feat = df_feat.join(df_pca, on="Customer_ID", how="left")

        # Drop original clickstream columns
        for c in click_cols_final:
            df_feat = df_feat.drop(c)

    # === Feature Engineering ===
    print("\n=== Feature Engineering ===")
    df_feat = (
        df_feat
        .withColumn("emi_to_salary_ratio", col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
        .withColumn("debt_to_balance_ratio", col("Outstanding_Debt") / col("Monthly_Balance"))
        .withColumn("delay_ratio", col("Num_of_Delayed_Payment") / (col("Num_of_Loan") + F.lit(1e-6)))
    )

    # === Save Feature Store ===
    print("\n=== Saving Gold Feature Store ===")
    partition_name = f"gold_feature_store_{tag}.parquet"
    filepath = os.path.join(gold_feature_store_directory, partition_name)
    df_feat.write.mode("overwrite").parquet(filepath)
    print("[Feature Store] saved to:", filepath)
    return df_feat


# ======================================================
# MAIN ENTRY POINT
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_date", type=str, required=True)
    parser.add_argument("--silver_dir", type=str, default="datamart/silver/")
    parser.add_argument("--gold_label_dir", type=str, default="datamart/gold/label_store/")
    parser.add_argument("--gold_feature_dir", type=str, default="datamart/gold/feature_store/")
    parser.add_argument("--dpd", type=int, default=30)
    parser.add_argument("--mob", type=int, default=6)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("GoldTablesAll").getOrCreate()

    # (A) Label Store
    process_labels_gold_table(
        snapshot_date_str=args.snapshot_date,
        silver_loan_daily_directory=os.path.join(args.silver_dir, "loan_daily/"),
        gold_label_store_directory=args.gold_label_dir,
        spark=spark,
        dpd=args.dpd,
        mob=args.mob
    )

    # (B) Feature Store
    process_features_gold_table(
        snapshot_date_str=args.snapshot_date,
        silver_base_directory=args.silver_dir,
        gold_feature_store_directory=args.gold_feature_dir,
        spark=spark
    )

    print("\n🎯 GOLD completed for Label Store + Feature Store")