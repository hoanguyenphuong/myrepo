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

from pyspark.sql.functions import col, when, regexp_replace, split, explode, trim, array_contains
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql import SparkSession

# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def _median_on_valid(df, colname, valid_cond=None, rel_acc=0.001):
    """
    Compute median using approxQuantile on valid subset (if provided).
    """
    base = df
    if valid_cond is not None:
        base = df.where(valid_cond)
    vals = base.approxQuantile(colname, [0.5], rel_acc)
    return vals[0] if vals and vals[0] is not None else None


# ==============================================================
# MAIN SILVER PROCESSOR
# ==============================================================

def process_silver_table(snapshot_date_str: str,
                         bronze_base_directory: str,
                         silver_base_directory: str,
                         spark: SparkSession):
    """
    Process SILVER layer for ALL FOUR tables:
      A) Loans (Label Store)  -> use teacher's logic
      B) Feature tables:
         - feature_clickstream
         - features_attributes
         - features_financials
    Outputs Parquet under datamart/silver/*.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    snap_tag = snapshot_date_str.replace("-", "_")

    # ------------------------------------------------------------------
    # PART A — LOAN DAILY (Label Store)  — Code của thầy
    # ------------------------------------------------------------------
    print("\n=== [A] SILVER: loan_daily (teacher's logic) ===")
    bronze_lms_dir = os.path.join(bronze_base_directory, "lms/")
    silver_loan_dir = _ensure_dir(os.path.join(silver_base_directory, "loan_daily/"))

    bronze_loan_file = os.path.join(bronze_lms_dir, f"bronze_loan_daily_{snap_tag}.csv")
    df_loan = spark.read.csv(bronze_loan_file, header=True, inferSchema=True)
    print(f"loaded from: {bronze_loan_file} row count:", df_loan.count())

    # enforce schema / data type
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for c, t in column_type_map.items():
        df_loan = df_loan.withColumn(c, col(c).cast(t))

    # augment: mob, installments_missed, first_missed_date, dpd
    df_loan = df_loan.withColumn("mob", col("installment_num").cast(IntegerType()))
    df_loan = df_loan.withColumn("installments_missed",
                                 F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df_loan = df_loan.withColumn(
        "first_missed_date",
        F.when(col("installments_missed") > 0,
               F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df_loan = df_loan.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0,
               F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    silver_loan_file = os.path.join(silver_loan_dir, f"silver_loan_daily_{snap_tag}.parquet")
    df_loan.write.mode("overwrite").parquet(silver_loan_file)
    print("saved to:", silver_loan_file)

    # ------------------------------------------------------------------
    # PART B — FEATURE TABLES (Feature Store)
    # ------------------------------------------------------------------

    # ---------- 1) feature_clickstream ----------
    print("\n=== [B1] SILVER: feature_clickstream ===")
    bronze_click_dir = os.path.join(bronze_base_directory, "feature_clickstream/")
    silver_click_dir = _ensure_dir(os.path.join(silver_base_directory, "feature_clickstream/"))

    bronze_click_file = os.path.join(bronze_click_dir, f"bronze_clickstream_{snap_tag}.csv")
    click_df = spark.read.csv(bronze_click_file, header=True, inferSchema=True)

    click_df = (click_df
                .withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
                .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
                )

    silver_click_file = os.path.join(silver_click_dir, f"silver_clickstream_{snap_tag}.parquet")
    click_df.write.mode("overwrite").parquet(silver_click_file)
    print("saved to:", silver_click_file)

    # ---------- 2) features_attributes ----------
    print("\n=== [B2] SILVER: features_attributes ===")
    bronze_attr_dir = os.path.join(bronze_base_directory, "feature_attributes/")
    silver_attr_dir = _ensure_dir(os.path.join(silver_base_directory, "feature_attributes/"))

    bronze_attr_file = os.path.join(bronze_attr_dir, f"bronze_attributes_{snap_tag}.csv")
    attr_df = spark.read.csv(bronze_attr_file, header=True, inferSchema=True)

    # Clean data
    attr_df = (attr_df
               .drop("Name", "SSN")
               .withColumn("Age", regexp_replace("Age", "_", ""))
               .withColumn("Age", col("Age").cast(FloatType()))
               .withColumn("Occupation",
                           when(F.length(F.regexp_replace("Occupation", "_", "")) == 0, "Unknown")
                           .otherwise(col("Occupation")))
               .withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
               .withColumn("Occupation", col("Occupation").cast(StringType()))
               .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
               )

    # Replace out-of-range ages by median (values <18 or >74)
    median_age = _median_on_valid(attr_df.where(col("Age").isNotNull()), "Age")
    attr_df = attr_df.withColumn(
        "Age",
        when((col("Age") < 18) | (col("Age") > 74), F.lit(median_age)).otherwise(col("Age"))
    )

    # One-hot encode Occupation (1/0)
    occupations = [r["Occupation"] for r in attr_df.select("Occupation").distinct().collect()]
    for occ in occupations:
        safe = occ.replace(" ", "_")
        attr_df = attr_df.withColumn(f"Occupation_{safe}", when(col("Occupation") == F.lit(occ), F.lit(1)).otherwise(F.lit(0)))

    silver_attr_file = os.path.join(silver_attr_dir, f"silver_attributes_{snap_tag}.parquet")
    attr_df.write.mode("overwrite").parquet(silver_attr_file)
    print("saved to:", silver_attr_file)

    # ---------- 3) features_financials ----------
    print("\n=== [B3] SILVER: features_financials ===")
    bronze_fin_dir = os.path.join(bronze_base_directory, "feature_financials/")
    silver_fin_dir = _ensure_dir(os.path.join(silver_base_directory, "feature_financials/"))

    bronze_fin_file = os.path.join(bronze_fin_dir, f"bronze_financials_{snap_tag}.csv")
    fin_df = spark.read.csv(bronze_fin_file, header=True, inferSchema=True)

    # Clean specials / placeholders
    fin_df = (fin_df
              # strings to Unknown / clean
              .withColumn("Payment_Behaviour", when(col("Payment_Behaviour") == "!@9#%8", "Unknown").otherwise(col("Payment_Behaviour")))
              .withColumn("Credit_Mix", when(col("Credit_Mix") == "_", "Unknown").otherwise(col("Credit_Mix")))
              .withColumn("Type_of_Loan", when(col("Type_of_Loan").isNull(), "Unknown").otherwise(col("Type_of_Loan")))
              # remove underscores & cast float
              .withColumn("Annual_Income", regexp_replace("Annual_Income", "_", "").cast(FloatType()))
              .withColumn("Num_of_Loan", regexp_replace("Num_of_Loan", "_", "").cast(FloatType()))
              .withColumn("Num_of_Delayed_Payment", regexp_replace("Num_of_Delayed_Payment", "_", "").cast(FloatType()))
              .withColumn("Outstanding_Debt", regexp_replace("Outstanding_Debt", "_", "").cast(FloatType()))
              .withColumn("Amount_invested_monthly", regexp_replace("Amount_invested_monthly", "_", "").cast(FloatType()))
              .withColumn("Credit_History_Age", regexp_replace("Credit_History_Age", " Months", "").cast(FloatType()))
              # ids / dates
              .withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
              .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
              )

    # Replace sentinel values with median (per flow)
    # Num_Bank_Accounts: -1 -> median
    fin_df = fin_df.withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast(FloatType()))
    nba_median = _median_on_valid(fin_df, "Num_Bank_Accounts", valid_cond=(col("Num_Bank_Accounts") >= 0))
    fin_df = fin_df.withColumn("Num_Bank_Accounts",
                               when(col("Num_Bank_Accounts") == -1, F.lit(nba_median)).otherwise(col("Num_Bank_Accounts")))

    # Num_of_Loan: -100 -> median
    nol_median = _median_on_valid(fin_df, "Num_of_Loan", valid_cond=(col("Num_of_Loan") >= 0))
    fin_df = fin_df.withColumn("Num_of_Loan",
                               when(col("Num_of_Loan") == -100, F.lit(nol_median)).otherwise(col("Num_of_Loan")))

    # Delay_from_due_date: {-1,-2,-3,-4,-5} -> median
    dfd_median = _median_on_valid(fin_df, "Delay_from_due_date", valid_cond=(col("Delay_from_due_date") >= 0))
    fin_df = fin_df.withColumn("Delay_from_due_date",
                               when(col("Delay_from_due_date").isin(-1, -2, -3, -4, -5), F.lit(dfd_median))
                               .otherwise(col("Delay_from_due_date")).cast(FloatType()))

    # Num_of_Delayed_Payment: -3 -> median
    nodp_median = _median_on_valid(fin_df, "Num_of_Delayed_Payment", valid_cond=(col("Num_of_Delayed_Payment") >= 0))
    fin_df = fin_df.withColumn("Num_of_Delayed_Payment",
                               when(col("Num_of_Delayed_Payment") == -3, F.lit(nodp_median))
                               .otherwise(col("Num_of_Delayed_Payment")))

    # Convert types per flow (keep as float64-equivalent => FloatType in Spark)
    numeric_cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance"
    ]
    for c in numeric_cols:
        if c in fin_df.columns:
            fin_df = fin_df.withColumn(c, col(c).cast(FloatType()))

    # Handle skewed numeric (log1p) — đúng danh sách bạn chốt
    skew_cols = [
        "Annual_Income", "Num_Bank_Accounts", "Num_of_Loan",
        "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
        "Interest_Rate", "Num_Credit_Card", "Total_EMI_per_month"
    ]
    for c in skew_cols:
        if c in fin_df.columns:
            fin_df = fin_df.withColumn(c, F.log1p(col(c)))

    # Encode categorical:
    # 1) Type_of_Loan: split & multi-hot (1/0)
    if "Type_of_Loan" in fin_df.columns:
        # build unique loan types
        tokens = (fin_df
                  .select(explode(split(col("Type_of_Loan"), r",\s*")).alias("loan_type"))
                  .withColumn("loan_type", trim(col("loan_type")))
                  .where(col("loan_type").isNotNull())
                  .distinct()
                  .collect())
        uniq_loan_types = [t["loan_type"] for t in tokens if t["loan_type"]]

        arr = split(col("Type_of_Loan"), r",\s*")
        for lt in uniq_loan_types:
            safe = lt.replace(" ", "_")
            fin_df = fin_df.withColumn(f"Type_of_Loan__{safe}", when(array_contains(arr, lt), F.lit(1)).otherwise(F.lit(0)))

    # 2) Credit_Mix: ordinal {Poor:0, Standard:1, Good:2, Unknown:-1}
    if "Credit_Mix" in fin_df.columns:
        fin_df = (fin_df
                  .withColumn("Credit_Mix",
                              when(col("Credit_Mix") == "Poor", F.lit(0))
                              .when(col("Credit_Mix") == "Standard", F.lit(1))
                              .when(col("Credit_Mix") == "Good", F.lit(2))
                              .otherwise(F.lit(-1)).cast(FloatType()))
                  )

    # 3) Payment_of_Min_Amount: binary {Yes:1, No:0, NM:-1}
    if "Payment_of_Min_Amount" in fin_df.columns:
        fin_df = (fin_df
                  .withColumn("Payment_of_Min_Amount",
                              when(col("Payment_of_Min_Amount") == "Yes", F.lit(1))
                              .when(col("Payment_of_Min_Amount") == "No", F.lit(0))
                              .otherwise(F.lit(-1)).cast(FloatType()))
                  )

    # 4) Payment_Behaviour: one-hot (1/0)
    if "Payment_Behaviour" in fin_df.columns:
        behaviours = [r["Payment_Behaviour"] for r in fin_df.select("Payment_Behaviour").distinct().collect()]
        for pb in behaviours:
            safe = pb.replace(" ", "_")
            fin_df = fin_df.withColumn(f"Payment_Behaviour__{safe}",
                                       when(col("Payment_Behaviour") == F.lit(pb), F.lit(1)).otherwise(F.lit(0)))

    silver_fin_file = os.path.join(silver_fin_dir, f"silver_financials_{snap_tag}.parquet")
    fin_df.write.mode("overwrite").parquet(silver_fin_file)
    print("saved to:", silver_fin_file)

    print("\n🎯 SILVER completed for loans + 3 feature tables.")
    return


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--bronze_dir", type=str, default="datamart/bronze/")
    parser.add_argument("--silver_dir", type=str, default="datamart/silver/")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("SilverAllTables").getOrCreate()
    process_silver_table(
        snapshot_date_str=args.snapshot_date,
        bronze_base_directory=args.bronze_dir,
        silver_base_directory=args.silver_dir,
        spark=spark
    )
