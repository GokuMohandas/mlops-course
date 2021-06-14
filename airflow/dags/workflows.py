from pathlib import Path

from great_expectations_provider.operators.great_expectations import (
    GreatExpectationsOperator,
)

from airflow.decorators import dag
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago
from app import cli
from tagifai import config

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}


@dag(
    dag_id="dataops",
    description="Data related operations.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["dataops"],
)
def dataops():
    """
    Workflows to validate data and create features.
    """

    # Extract data from DWH, blob storage, etc.
    extract_data = BashOperator(
        task_id="extract_data",
        bash_command=f"cd {config.BASE_DIR} && dvc pull",
    )

    # Validate data
    validate_projects = GreatExpectationsOperator(
        task_id="validate_projects",
        checkpoint_name="projects",
        data_context_root_dir="great_expectations",
        fail_task_on_validation_failure=True,
    )
    validate_tags = GreatExpectationsOperator(
        task_id="validate_tags",
        checkpoint_name="tags",
        data_context_root_dir="great_expectations",
        fail_task_on_validation_failure=True,
    )

    # Compute features
    compute_features = PythonOperator(
        task_id="compute_features",
        python_callable=cli.compute_features,
        op_kwargs={"params_fp": Path(config.CONFIG_DIR, "params.json")},
    )

    # Feature store
    END_TS = ""
    cache_to_feature_store = BashOperator(
        task_id="cache_to_feature_store",
        bash_command=f"cd {config.BASE_DIR}/features && feast materialize-incremental {END_TS}",
    )

    # Task relationships
    extract_data >> [validate_projects, validate_tags] >> compute_features >> cache_to_feature_store


def _evaluate_model():
    return "improved"


@dag(
    dag_id="mlops",
    description="ML modeling related operations.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def mlops():
    """
    Optimization, training and evaluation of models.
    """

    # Extract features
    extract_features = PythonOperator(
        task_id="extract_features",
        python_callable=cli.get_historical_features,
    )

    # Optimization
    optimization = BashOperator(
        task_id="optimization",
        bash_command="tagifai optimize",
    )

    # Training
    train_model = BashOperator(
        task_id="train_model",
        bash_command="tagifai train-model",
    )

    # Evaluate
    evaluate_model = BranchPythonOperator(  # BranchPythonOperator returns a task_id or [task_ids]
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    # Improved or regressed
    improved = BashOperator(
        task_id="improved",
        bash_command="echo IMPROVED",
    )
    regressed = BashOperator(
        task_id="regressed",
        bash_command="echo REGRESSED",
    )

    # Deploy
    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command="echo 1",  # push to GitHub to kick off deployment workflows
    )

    # Reset references for monitoring
    set_monitoring_references = BashOperator(
        task_id="set_monitoring_references",
        bash_command="echo 1",  # tagifai set-monitoring-references
    )

    # Notifications (use appropriate operators, ex. EmailOperator)
    notify_teams = BashOperator(task_id="notify_teams", bash_command="echo 1")
    file_report = BashOperator(task_id="file_report", bash_command="echo 1")

    # Task relationships
    extract_features >> optimization >> train_model >> evaluate_model >> [improved, regressed]
    improved >> [set_monitoring_references, deploy_model, notify_teams]
    regressed >> [notify_teams, file_report]


# Define DAGs
dataops_dag = dataops()
mlops_dag = mlops()
