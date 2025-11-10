import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import pyspark
import pyspark.sql.functions as F

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# ======================================================
# 1️⃣ Initialize Spark Session
# ======================================================
spark = pyspark.sql.SparkSession.builder \
    .appName("dev_full_pipeline") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ======================================================
# 2️⃣ Config: snapshot date range
# ======================================================
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current_date = datetime(start_date.year, start_date.month, 1)
    result = []
    while current_date <= end_date:
        result.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    return result

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("📆 Processing snapshot dates:", dates_str_lst)

# ======================================================
# 🥉 BRONZE LAYER — Raw Data Loading
# ======================================================
bronze_root_directory = "datamart/bronze/"
os.makedirs(bronze_root_directory, exist_ok=True)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(
        date_str, bronze_root_directory, spark
    )

# ======================================================
# 🥈 SILVER LAYER — Data Cleaning & Transformation
# ======================================================
silver_root_directory = "datamart/silver/"
os.makedirs(silver_root_directory, exist_ok=True)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(
        date_str,
        bronze_root_directory,
        silver_root_directory,
        spark
    )

# ======================================================
# 🥇 GOLD LAYER — Label & Feature Store Creation
# ======================================================

# ---------- (a) LABEL STORE ----------
gold_label_store_directory = "datamart/gold/label_store/"
os.makedirs(gold_label_store_directory, exist_ok=True)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(
        date_str,
        os.path.join(silver_root_directory, "loan_daily/"),
        gold_label_store_directory,
        spark,
        dpd=30,
        mob=6
    )


# ---------- (b) FEATURE STORE ----------
gold_feature_store_directory = "datamart/gold/feature_store/"
os.makedirs(gold_feature_store_directory, exist_ok=True)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_features_gold_table(
        date_str,
        silver_root_directory,
        gold_feature_store_directory,
        spark
    )

# ======================================================
# ✅ Merge Feature Store + Label Store (final join)
# ======================================================
label_files = glob.glob(os.path.join(gold_label_store_directory, "*.parquet"))
feature_files = glob.glob(os.path.join(gold_feature_store_directory, "*.parquet"))

if not label_files:
    raise FileNotFoundError("❌ No label store parquet found in gold/label_store/")
if not feature_files:
    raise FileNotFoundError("❌ No feature store parquet found in gold/feature_store/")

df_label = spark.read.parquet(*label_files)
df_feat = spark.read.parquet(*feature_files)

df_final = df_feat.join(df_label, on="Customer_ID", how="inner")

print("\n🎯 Final Feature + Label joined dataset")
print("Row count:", df_final.count(), "| Columns:", len(df_final.columns))
df_final.show(5, truncate=False)

print("\n✅ Pipeline completed successfully!")