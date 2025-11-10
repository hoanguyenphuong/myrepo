from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'hoa',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='credit_default_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for credit default prediction',
    schedule_interval='0 0 1 * *',  # Run on the 1st of each month at midnight
    start_date=datetime(2024, 9, 1),
    catchup=True,
    tags=['ml', 'credit_default'],
) as dag:

    # --- Data Dependency Checks ---
    dep_check_source_data = EmptyOperator(task_id='dep_check_source_data')

    # --- Bronze / Silver / Gold Layers ---
    bronze_layer = BashOperator(
        task_id='bronze_layer',
        bash_command='python utils/data_processing_bronze_table.py --snapshotdate "{{ ds }}"'
    )

    silver_layer = BashOperator(
        task_id='silver_layer',
        bash_command='python utils/data_processing_silver_table.py --snapshotdate "{{ ds }}"'
    )

    gold_layer = BashOperator(
        task_id='gold_layer',
        bash_command='python utils/data_processing_gold_table.py --snapshotdate "{{ ds }}"'
    )

    # --- Label and Feature Completion ---
    label_store_completed = EmptyOperator(task_id='label_store_completed')
    feature_store_completed = EmptyOperator(task_id='feature_store_completed')

    # --- Model AutoML Training Branch ---
    model_automl_start = EmptyOperator(task_id='model_automl_start')

    model_1_automl = BashOperator(
        task_id='model_1_automl',
        bash_command='python scripts/model_train_1.py --snapshotdate "{{ ds }}"'
    )

    model_2_automl = BashOperator(
        task_id='model_2_automl',
        bash_command='python scripts/model_train_2.py --snapshotdate "{{ ds }}"'
    )

    model_automl_completed = EmptyOperator(task_id='model_automl_completed')

    # --- Model Inference Branch ---
    model_inference_start = EmptyOperator(task_id='model_inference_start')

    model_1_inference = BashOperator(
        task_id='model_1_inference',
        bash_command='python scripts/model_inference_1.py --snapshotdate "{{ ds }}"'
    )

    model_2_inference = BashOperator(
        task_id='model_2_inference',
        bash_command='python scripts/model_inference_2.py --snapshotdate "{{ ds }}"'
    )

    model_inference_completed = EmptyOperator(task_id='model_inference_completed')

    # --- Monitoring (Before / After Labels) ---
    model_monitor_before_label_start = EmptyOperator(task_id='model_monitor_before_label_start')

    model_1_monitor_before_label = BashOperator(
        task_id='model_1_monitor_before_label',
        bash_command='python scripts/model_monitor_before_label_1.py --snapshotdate "{{ ds }}"'
    )

    model_2_monitor_before_label = BashOperator(
        task_id='model_2_monitor_before_label',
        bash_command='python scripts/model_monitor_before_label_2.py --snapshotdate "{{ ds }}"'
    )

    model_monitor_before_label_completed = EmptyOperator(task_id='model_monitor_before_label_completed')

    model_monitor_after_label_start = EmptyOperator(task_id='model_monitor_after_label_start')

    model_1_monitor_after_label = BashOperator(
        task_id='model_1_monitor_after_label',
        bash_command='python scripts/model_monitor_after_label_1.py --snapshotdate "{{ ds }}"'
    )

    model_2_monitor_after_label = BashOperator(
        task_id='model_2_monitor_after_label',
        bash_command='python scripts/model_monitor_after_label_2.py --snapshotdate "{{ ds }}"'
    )

    model_monitor_after_label_completed = EmptyOperator(task_id='model_monitor_after_label_completed')

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    # ETL flow
    dep_check_source_data >> bronze_layer >> silver_layer >> gold_layer
    gold_layer >> [label_store_completed, feature_store_completed]

    # Label branch (training)
    [label_store_completed, feature_store_completed] >> model_automl_start
    model_automl_start >> [model_1_automl, model_2_automl] >> model_automl_completed

    # Feature branch (inference)
    feature_store_completed >> model_inference_start
    model_inference_start >> [model_1_inference, model_2_inference] >> model_inference_completed

    # Monitoring (before + after label)
    model_inference_completed >> model_monitor_before_label_start
    model_monitor_before_label_start >> [model_1_monitor_before_label, model_2_monitor_before_label] >> model_monitor_before_label_completed

    [model_monitor_before_label_completed, label_store_completed] >> model_monitor_after_label_start
    model_monitor_after_label_start >> [model_1_monitor_after_label, model_2_monitor_after_label] >> model_monitor_after_label_completed
