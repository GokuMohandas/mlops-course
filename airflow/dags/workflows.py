from pathlib import Path

from great_expectations_provider.operators.great_expectations import (
    GreatExpectationsOperator,
)

from airflow.decorators import dag
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago
from app import cli, config

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}


@dag(
    dag_id="data",
    description="Feature creating operations.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["dataops"],
)
def data():
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

    # Cache (feature store, database, warehouse, etc.)
    END_TS = ""
    cache = BashOperator(
        task_id="cache_to_feature_store",
        bash_command=f"cd {config.BASE_DIR}/features && feast materialize-incremental {END_TS}",
    )

    # Task relationships
    extract_data >> [validate_projects, validate_tags] >> compute_features >> cache


def _evaluate_model():
    return "improved"


@dag(
    dag_id="model",
    description="Model creating operations.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def model():
    """
    Model creating tasks such as optimization, training, evaluation and serving.
    """

    # Extract features
    extract_features = PythonOperator(
        task_id="extract_features",
        python_callable=cli.get_historical_features,
    )

    # Optimization
    optimization = BashOperator(
        task_id="optimization",
        bash_command="echo `tagifai optimize`",
    )

    # Train model
    train = BashOperator(
        task_id="train",
        bash_command="echo `tagifai train-model`",
    )

    # Evaluate model
    evaluate = BranchPythonOperator(  # BranchPythonOperator returns a task_id or [task_ids]
        task_id="evaluate",
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

    # Serve model
    serve = BashOperator(
        task_id="serve",  # push to GitHub to kick off serving workflows
        bash_command="echo served model",  # or to a purpose-built model server, etc.
    )

    # Notifications (use appropriate operators, ex. EmailOperator)
    report = BashOperator(task_id="report", bash_command="echo filed report")

    # Task relationships
    extract_features >> optimization >> train >> evaluate >> [improved, regressed]
    improved >> serve
    regressed >> report


def _update_policy_engine():
    return "improve"


@dag(
    dag_id="update",
    description="Model updating operations.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def update():
    """
    Model updating tasks such as monitoring, retraining, etc.
    """
    # Monitoring (inputs, predictions, etc.)
    # Considers thresholds, windows, frequency, etc.
    monitoring = BashOperator(
        task_id="monitoring",
        bash_command="echo monitoring",
    )

    # Update policy engine (continue, improve, rollback, etc.)
    update_policy_engine = BranchPythonOperator(
        task_id="update_policy_engine",
        python_callable=_update_policy_engine,
    )

    # Policies
    _continue = BashOperator(
        task_id="continue",
        bash_command="echo continue",
    )
    inspect = BashOperator(
        task_id="inspect",
        bash_command="echo inspect",
    )
    improve = BashOperator(
        task_id="improve",
        bash_command="echo improve",
    )
    rollback = BashOperator(
        task_id="rollback",
        bash_command="echo rollback",
    )

    # Compose retraining dataset
    # Labeling, QA, augmentation, upsample poor slices, weight samples, etc.
    compose_retraining_dataset = BashOperator(
        task_id="compose_retraining_dataset",
        bash_command="echo compose retraining dataset",
    )

    # Retrain (initiates model creation workflow)
    retrain = BashOperator(
        task_id="retrain",
        bash_command="echo retrain",
    )

    # Task relationships
    monitoring >> update_policy_engine >> [_continue, inspect, improve, rollback]
    improve >> compose_retraining_dataset >> retrain


# Define DAGs
data_dag = data()
model_dag = model()
update_dag = update()
