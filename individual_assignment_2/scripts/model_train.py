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

from pyspark.sql.functions import col, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import add_months


# to call this script: python scripts/model_train.py --snapshotdate "2024-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    model_train_date_str = snapshotdate
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8
    
    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_ratio"] = train_test_ratio 
    pprint.pprint(config)
    
    # --- get label ---
    label_store_path = "datamart/gold/label_store/"
    label_files = glob.glob(os.path.join(label_store_path, "*.parquet"))
    label_store_sdf = spark.read.parquet(*label_files)
    
    # Filter to the full modeling window (train/test + OOT)
    labels_sdf = label_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    print("labels_sdf rows:", labels_sdf.count())

    # --- get features ---
    feature_store_path = "datamart/gold/feature_store/"
    feature_files = glob.glob(os.path.join(feature_store_path, "*.parquet"))
    features_store_sdf = spark.read.parquet(*feature_files)
    
    # Keep the same date window as labels (important for consistent merging)
    features_sdf = features_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    print("features_sdf rows:", features_sdf.count())
    
    # 1) Harmonize dtypes for join keys
    from pyspark.sql.types import StringType, DateType
    labels_sdf = labels_sdf.withColumn("Customer_ID", col("Customer_ID").cast(StringType())) \
                           .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    features_sdf = features_sdf.withColumn("Customer_ID", col("Customer_ID").cast(StringType())) \
                               .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    
    # 2) Sanity checks (keep these prints for now)
    print("Labels schema:"); labels_sdf.printSchema()
    print("Features schema:"); features_sdf.printSchema()
    
    print("Labels dates:"); labels_sdf.groupBy("snapshot_date").count().orderBy("snapshot_date").show(10, False)
    print("Features dates:"); features_sdf.groupBy("snapshot_date").count().orderBy("snapshot_date").show(10, False)
    
    print("Distinct label customers:", labels_sdf.select("Customer_ID").distinct().count())
    print("Distinct feature customers:", features_sdf.select("Customer_ID").distinct().count())
    
    # 3) How many rows would an *inner* join produce?
    probe_inner = labels_sdf.join(features_sdf, ["Customer_ID", "snapshot_date"], "inner")
    print("Inner-join row count:", probe_inner.count())
    print("Left-join row count (labels):", labels_sdf.count())

    # 4)
    label_ids = labels_sdf.select("Customer_ID").distinct()
    feature_ids = features_sdf.select("Customer_ID").distinct()

    print("Label count:", label_ids.count())
    print("Feature count:", feature_ids.count())
    print("Common IDs:", label_ids.intersect(feature_ids).count())

    # ensure both join keys have the same types and formats
    labels_sdf = labels_sdf.withColumn("Customer_ID", col("Customer_ID").cast("string")) \
                           .withColumn("snapshot_date", to_date(col("snapshot_date")))
    
    features_sdf = features_sdf.withColumn("Customer_ID", col("Customer_ID").cast("string")) \
                               .withColumn("snapshot_date", to_date(col("snapshot_date")))

    # Check types
    print("Label schema:")
    labels_sdf.printSchema()
    print("Feature schema:")
    features_sdf.printSchema()
    
    # Check sample dates
    print("Label snapshot_date distincts:")
    labels_sdf.select("snapshot_date").distinct().orderBy("snapshot_date").show(10)
    print("Feature snapshot_date distincts:")
    features_sdf.select("snapshot_date").distinct().orderBy("snapshot_date").show(10)

    # --- prepare data for modeling ---
    data_pdf = labels_sdf.join(features_sdf, on=["Customer_ID", "snapshot_date"], how="left").toPandas()

    # define feature columns first
    exclude_cols = ["Customer_ID", "snapshot_date", "Delay_from_due_date", "label", "label_def"]
    feature_cols = [c for c in data_pdf.columns if c not in exclude_cols]

    # --- ensure only numeric columns (within feature_cols only) ---
    numeric_cols = [c for c in feature_cols if np.issubdtype(data_pdf[c].dtype, np.number)]
    non_numeric = [c for c in feature_cols if c not in numeric_cols]
    if non_numeric:
        print("⚠️ Dropping non-numeric columns:", non_numeric)
    feature_cols = numeric_cols

    print("\n=== FEATURE COLUMNS USED IN MODEL ===")
    print(f"Total features: {len(feature_cols)}")
    print(feature_cols)

    # # fill missing feature values with 0 before split
    # data_pdf[feature_cols] = data_pdf[feature_cols].fillna(0)

    # split data into train - test - oot
    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]

    # now X_train, X_test, X_oot should use only numeric columns
    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"], 
        test_size=1 - config["train_test_ratio"],
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"]
    )
    
    
    print('X_train', X_train.shape[0])
    print('X_test', X_test.shape[0])
    print('X_oot', X_oot.shape[0])
    print('y_train', y_train.shape[0], round(y_train.mean(),2))
    print('y_test', y_test.shape[0], round(y_test.mean(),2))
    print('y_oot', y_oot.shape[0], round(y_oot.mean(),2))
    
    # set up standard scalar preprocessing
    scaler = StandardScaler()
    
    transformer_stdscaler = scaler.fit(X_train) # Q which should we use? train? test? oot? all?
    
    # transform data
    X_train_processed = transformer_stdscaler.transform(X_train)
    X_test_processed = transformer_stdscaler.transform(X_test)
    X_oot_processed = transformer_stdscaler.transform(X_oot)
    
    print('X_train_processed', X_train_processed.shape[0])
    print('X_test_processed', X_test_processed.shape[0])
    print('X_oot_processed', X_oot_processed.shape[0])
    
    pd.DataFrame(X_train_processed)
    
    
    # --- train model ---
    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)
    
    # Define the hyperparameter space to search
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)
    
    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=100,  # Number of iterations for random search
        cv=3,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )
    
    # Perform the random search
    random_search.fit(X_train_processed, y_train)
    
    # Output the best parameters and best score
    print("Best parameters found: ", random_search.best_params_)
    print("Best AUC score: ", random_search.best_score_)
    
    # Evaluate the model on the train set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
    train_auc_score = roc_auc_score(y_train, y_pred_proba)
    print("Train AUC score: ", train_auc_score)
    
    # Evaluate the model on the test set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
    test_auc_score = roc_auc_score(y_test, y_pred_proba)
    print("Test AUC score: ", test_auc_score)
    
    # Evaluate the model on the oot set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_oot_processed)[:, 1]
    oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
    print("OOT AUC score: ", oot_auc_score)
    
    print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
    print("Test GINI score: ", round(2*test_auc_score-1,3))
    print("OOT GINI score: ", round(2*oot_auc_score-1,3))
    
    
    # --- prepare model artefact to save ---
    model_artefact = {}
    
    model_artefact['model'] = best_model
    model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
    model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc_score
    model_artefact['results']['auc_test'] = test_auc_score
    model_artefact['results']['auc_oot'] = oot_auc_score
    model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
    model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
    model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
    model_artefact['hp_params'] = random_search.best_params_
    
    # Save feature column list for inference
    model_artefact['feature_cols'] = feature_cols

    pprint.pprint(model_artefact)
    
    
    # --- save artefact to model bank ---
    # create model_bank dir
    model_bank_directory = "model_bank/"
    
    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)
    
    # Full path to the file
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')
    
    # Write the model to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)
    
    print(f"Model saved to {file_path}")

    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')



if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
