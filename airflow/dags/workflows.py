from pathlib import Path

import pandas as pd
from great_expectations_provider.operators.great_expectations import (
    GreatExpectationsOperator,
)

from airflow.decorators import dag
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.python_operator import PythonOperator
from airflow.providers.airbyte.operators.airbyte import (
    AirbyteTriggerSyncOperator,
)
from airflow.utils.dates import days_ago
from config import config
from tagifai import main, utils

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}


# Define DAG
@dag(
    dag_id="temp",
    description="DataOps tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["dataops"],
)
def temp():
    extract_projects = AirbyteTriggerSyncOperator(
        task_id="extract_projects",
        airbyte_conn_id="airbyte",
        connection_id="0a402218-61e9-4f0f-a87e-8be310fd0b23",
        asynchronous=False,
        timeout=3600,
        wait_seconds=3,
    )
    extract_tags = AirbyteTriggerSyncOperator(
        task_id="extract_tags",
        airbyte_conn_id="airbyte",
        connection_id="2ac58ee0-dd5b-4feb-918c-cd649061a860",
        asynchronous=False,
        timeout=3600,
        wait_seconds=3,
    )

    # Define DAG
    [extract_projects, extract_tags]


def _extract(ti):
    """Extract from source (ex. DB, API, etc.)
    Our simple ex: extract data from a URL
    """
    pass


def _load(ti):
    """Load into data system (ex. warehouse)
    Our simple ex: load extracted data into a local file
    """
    projects = ti.xcom_pull(key="projects", task_ids=["extract"])[0]
    tags = ti.xcom_pull(key="tags", task_ids=["extract"])[0]
    utils.save_dict(d=projects, filepath=Path(config.DATA_DIR, "projects.csv"))
    utils.save_dict(d=tags, filepath=Path(config.DATA_DIR, "tags.csv"))


def _transform(ti):
    """Transform (ex. using DBT inside DWH)
    Our simple ex: using pandas to remove missing data samples
    """
    projects = ti.xcom_pull(key="projects", task_ids=["extract"])[0]
    df = pd.DataFrame(projects)
    df = df[df.tag.notnull()]  # drop rows w/ no tag
    utils.save_dict(d=df.to_dict(orient="records"), filepath=Path(config.DATA_DIR, "projects.csv"))


# Define DAG
@dag(
    dag_id="DataOps",
    description="DataOps tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["dataops"],
)
def dataops():
    extract = PythonOperator(task_id="extract", python_callable=_extract)
    validate_projects = GreatExpectationsOperator(
        task_id="validate_projects",
        checkpoint_name="projects",
        data_context_root_dir="tests/great_expectations",
        fail_task_on_validation_failure=True,
    )
    validate_tags = GreatExpectationsOperator(
        task_id="validate_tags",
        checkpoint_name="tags",
        data_context_root_dir="tests/great_expectations",
        fail_task_on_validation_failure=True,
    )
    load = PythonOperator(task_id="load", python_callable=_load)
    transform = PythonOperator(task_id="transform", python_callable=_transform)
    validate_transforms = GreatExpectationsOperator(
        task_id="validate_transforms",
        checkpoint_name="projects",
        data_context_root_dir="tests/great_expectations",
        fail_task_on_validation_failure=True,
    )

    # Define DAG
    (extract >> [validate_projects, validate_tags] >> load >> transform >> validate_transforms)


def _offline_evaluation():
    """Compare offline evaluation report
    (overall, fine-grained and slice metrics).
    And ensure model behavioral tests pass.
    """
    return True


def _online_evaluation():
    """Run online experiments (AB, shadow, canary) to
    determine if new system should replace the current.
    """
    passed = True
    if passed:
        return "deploy"
    else:
        return "inspect"


# Define DAG
@dag(
    dag_id="MLOps",
    description="MLOps tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def mlops():
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
    offline_evaluation = PythonOperator(
        task_id="offline_evaluation",
        python_callable=_offline_evaluation,
    )
    online_evaluation = BranchPythonOperator(
        task_id="online_evaluation",
        python_callable=_online_evaluation,
    )
    deploy = BashOperator(
        task_id="deploy",
        bash_command="echo update model endpoint w/ new artifacts",
    )
    inspect = BashOperator(
        task_id="inspect",
        bash_command="echo inspect why online experiment failed",
    )
    (optimize >> train >> offline_evaluation >> online_evaluation >> [deploy, inspect])


# Run DAGs
temp_ops = temp()
data_ops = dataops()
ml_ops = mlops()
