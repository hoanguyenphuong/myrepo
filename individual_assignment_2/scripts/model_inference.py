import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# To call this script: 
# option 1: Specify both snapshot date and model name
#     python scripts/model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"
#
# option 2: Specify only snapshot date (the script will automatically use the latest model in model_bank/)
#     python scripts/model_inference.py --snapshotdate "2024-09-01"


# --- Utility: Get the latest trained model from model_bank ---
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
    
def main(snapshotdate, modelname=None):
    print('\n\n---starting job---\n\n')

    # Use the latest model if no model name is provided
    if not modelname or modelname.strip() == "":
        modelname = os.path.basename(get_latest_model())  # only filename, not path
        
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = os.path.join(config["model_bank_directory"], config["model_name"])
    
    pprint.pprint(config)
    

    # --- load model artefact from model bank ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    # --- load feature store ---
    feature_store_path = "datamart/gold/feature_store/"
    feature_files = glob.glob(os.path.join(feature_store_path, "*.parquet"))
    features_store_sdf = spark.read.parquet(*feature_files)
    
    
    # extract feature store
    features_sdf = features_store_sdf.filter(col("snapshot_date") == config["snapshot_date"])
    print("Extracted features for:", config["snapshot_date"])
    
    features_pdf = features_sdf.toPandas()


    # --- preprocess data for modeling ---
    feature_cols = model_artefact['feature_cols']
    X_inference = features_pdf[feature_cols]
    
    # apply transformer - standard scaler
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference = transformer_stdscaler.transform(X_inference)
    
    print('X_inference', X_inference.shape[0])


    # --- model prediction inference ---
    # load model
    model = model_artefact["model"]
    
    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["pd_default"] = y_inference
    

    # --- save model inference to datamart gold table ---
    # save under snapshot-based folder instead of model-based folder
    snapshot_folder_name = f"credit_model_{config['snapshot_date_str'].replace('-', '_')}"
    gold_directory = f"datamart/gold/model_predictions/{snapshot_folder_name}/"
    os.makedirs(gold_directory, exist_ok=True)
    
    output_filename = f"{snapshot_folder_name}_predictions_{config['snapshot_date_str'].replace('-', '_')}.parquet"
    filepath = os.path.join(gold_directory, output_filename)
    
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print(f"✅ Saved to: {filepath}")
    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=False, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
