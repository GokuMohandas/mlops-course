from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account
from great_expectations_provider.operators.great_expectations import (
    GreatExpectationsOperator,
)

from airflow.decorators import dag
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from config import config
from tagifai import main

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}
PROJECT_ID = "made-with-ml-359923"  # REPLACE
SERVICE_ACCOUNT_KEY_JSON = "/Users/goku/Downloads/made-with-ml-359923-9df280204d63.json"  # REPLACE
GE_ROOT_DIR = Path(config.BASE_DIR, "tests", "great_expectations")


def _extract_from_dwh():
    """Extract labeled data from
    our BigQuery data warehouse and
    save it locally."""
    # Establish connection to DWH
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_JSON)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    # Query data
    query_job = client.query(
        """
        SELECT *
        FROM mlops_course.labeled_projects"""
    )
    results = query_job.result()
    results.to_dataframe().to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)


@dag(
    dag_id="mlops",
    description="MLOps tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def mlops():
    """MLOps workflows."""
    extract_from_dwh = PythonOperator(
        task_id="extract_data",
        python_callable=_extract_from_dwh,
    )
    validate = GreatExpectationsOperator(
        task_id="validate",
        checkpoint_name="labeled_projects",
        data_context_root_dir=GE_ROOT_DIR,
        fail_task_on_validation_failure=True,
    )
    optimize = PythonOperator(
        task_id="optimize",
        python_callable=main.optimize,
        op_kwargs={
            "args_fp": Path(config.CONFIG_DIR, "args.json"),
            "study_name": "optimization",
            "num_trials": 1,
        },
    )
    train = PythonOperator(
        task_id="train",
        python_callable=main.train_model,
        op_kwargs={
            "args_fp": Path(config.CONFIG_DIR, "args.json"),
            "experiment_name": "baselines",
            "run_name": "sgd",
        },
    )

    # Define DAG
    extract_from_dwh >> validate >> optimize >> train


# Run DAGs
ml = mlops()
