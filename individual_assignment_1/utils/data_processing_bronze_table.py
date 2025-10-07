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

def process_bronze_table(snapshot_date_str, bronze_root_directory, spark):
    """
    Unified Bronze Layer ingestion for both:
      1️⃣ Label Store (loan_daily)
      2️⃣ Feature Store (clickstream, attributes, financials)
    - Only loads raw data (no cleaning)
    - Saves each as bronze snapshots under datamart/bronze/
    """

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # ========= 1️⃣ LABEL STORE (LMS) =========
    print("\n📥 Loading Label Store (loan_daily)...")
    csv_file_path = "data/lms_loan_daily.csv"

    # Load + filter only snapshot_date = current date
    df_lms = (
        spark.read.csv(csv_file_path, header=True, inferSchema=True)
        .filter(col("snapshot_date") == snapshot_date)
    )
    print(f"{snapshot_date_str} row count:", df_lms.count())

    bronze_lms_directory = os.path.join(bronze_root_directory, "lms/")
    os.makedirs(bronze_lms_directory, exist_ok=True)

    # Save to bronze/lms/
    lms_file = os.path.join(
        bronze_lms_directory,
        f"bronze_loan_daily_{snapshot_date_str.replace('-', '_')}.csv",
    )
    df_lms.toPandas().to_csv(lms_file, index=False)
    print(f"✅ Saved → {lms_file}")

    # ========= 2️⃣ FEATURE STORE TABLES =========
    feature_sources = {
        "clickstream": "data/feature_clickstream.csv",
        "attributes": "data/features_attributes.csv",
        "financials": "data/features_financials.csv",
    }

    for name, path in feature_sources.items():
        print(f"\n📥 Loading Feature Table: {name} ...")
        df = spark.read.csv(path, header=True, inferSchema=True)

        output_dir = os.path.join(bronze_root_directory, f"feature_{name}/")
        os.makedirs(output_dir, exist_ok=True)

        out_file = os.path.join(
            output_dir,
            f"bronze_{name}_{snapshot_date_str.replace('-', '_')}.csv",
        )

        df.toPandas().to_csv(out_file, index=False)
        print(f"✅ Saved raw {name} data → {out_file}")

    print("\n🎉 Bronze ingestion complete (no transformation applied).")
    return


# Allow manual testing
if __name__ == "__main__":
    spark = pyspark.sql.SparkSession.builder.appName("BronzeLayer").getOrCreate()
    process_bronze_table("2023-01-01", "datamart/bronze/", spark)
